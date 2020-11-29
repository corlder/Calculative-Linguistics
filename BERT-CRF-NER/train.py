import torch
import torch.nn as nn
import numpy as np
from tqdm import trange
import config
from model import BertNER
from metrics import f1_score, bad_case
from transformers import BertTokenizer
import sys


def train_epoch(train_loader, model, optimizer, scheduler, epoch):
    # set model to training mode
    model.train()
    # step number in one epoch: 336
    train_losses = 0
    for idx, batch_samples in enumerate(train_loader):
        # print(idx,end ="")
        batch_data, batch_token_starts, batch_labels = batch_samples
        batch_masks = batch_data.gt(0)  # get padding mask
        # compute model output and loss
        loss = model((batch_data, batch_token_starts),
                     token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)[0]
        # print(loss)
        train_losses += loss.item()
        # clear previous gradients, compute gradients of all variables wrt loss
        model.zero_grad()
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
        # performs updates using calculated gradients
        optimizer.step()
        scheduler.step()
    print("Epoch: ", epoch, ", train loss: ", float(train_losses) / len(train_loader))
    sys.stdout.flush()


def train(train_loader, dev_loader, model, optimizer, scheduler, model_dir):
    """train the model and test model performance"""
    # reload weights from restore_dir if specified
    if model_dir is not None and config.load_before:
        print("start to load model")
        model = BertNER.from_pretrained(model_dir)
        model.to(config.device)
        print("--------Load model from ", model_dir, "--------")
    best_val_f1 = 0.0
    patience_counter = 0
    # start training
    for epoch in range(1, config.epoch_num + 1):
        train_epoch(train_loader, model, optimizer, scheduler, epoch)
        val_metrics = evaluate(dev_loader, model)
        val_f1 = val_metrics['f1']
        print("Epoch: ", epoch, ", dev loss: ", val_metrics['loss'], ", f1 score: ", val_f1)
        improve_f1 = val_f1 - best_val_f1
        if improve_f1 > 1e-5:
            best_val_f1 = val_f1
            model.save_pretrained(model_dir)
            print("--------Save best model!--------")
            if improve_f1 < config.patience:
                patience_counter += 1
            else:
                patience_counter = 0
        else:
            patience_counter += 1
        # Early stopping and logging best f1
        if (patience_counter >= config.patience_num and epoch > config.min_epoch_num) or epoch == config.epoch_num:
            print("Best val f1: ", best_val_f1)
            break
    print("Training Finished!")
    sys.stdout.flush()


def evaluate(dev_loader, model, mode = 'dev'):
    # set model to evaluation mode
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(config.bert_model,do_lower_case=True,skip_special_tokens=True)
    id2label = config.id2label
    true_tags = []
    pred_tags = []
    sent_data = []
    dev_losses = 0

    with torch.no_grad():
        for idx, batch_samples in enumerate(dev_loader):
            batch_data, batch_token_starts, batch_tags = batch_samples
            sent_data.extend([[tokenizer.convert_ids_to_tokens(idx.item()) for idx in indices
                               if (idx.item() > 0 and idx.item() != 101)] for indices in batch_data])
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            loss = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)[0]
            dev_losses += loss.item()
            # shape: (batch_size, max_len, num_labels)
            batch_output = model((batch_data, batch_token_starts),
                                 token_type_ids=None, attention_mask=batch_masks)[0]

            batch_output = batch_output.detach()
            batch_mask = batch_tags.gt(-1)
            batch_tags = batch_tags.to('cpu').numpy()
            # batch_mask = batch_tags.gt(-1)
            batch_pred_tags = model.crf.decode(batch_output,mask=batch_mask)
            # print(batch_pred_tags)
            # pred_tags.extend(batch_pred_tags)
            pred_tags.extend([[id2label.get(idx) for idx in indices] for indices in batch_pred_tags])
            
            # pred_tags.extend([[id2label.get(idx) sdfor idx in indices] for indices in np.argmax(batch_output, axis=2)])
            true_tags.extend([[id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
    
    assert len(pred_tags) == len(true_tags)
    assert len(sent_data) == len(true_tags)
    # logging loss, f1 and report
    metrics = {}
    if mode == 'dev':
        f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1'] = f1
    else:
        bad_case(true_tags, pred_tags, sent_data)
        f1_labels, f1 = f1_score(true_tags, pred_tags, mode)
        metrics['f1_labels'] = f1_labels
        metrics['f1'] = f1
    metrics['loss'] = float(dev_losses) / len(dev_loader)
    return metrics
