import time
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optimizer
import matplotlib.pyplot as plt
import pandas as pd

def get_loss_fn():
    """
    Return the loss function you will use to train the model.

    Hint: use nn.CrossEntropyLoss; be sure to ignore the padding index (you
    can hardcode this value)
    """
    return nn.CrossEntropyLoss(ignore_index=0)


def calculate_loss(scores, labels, loss_fn):
    """
    Calculate the loss.

    Input:
        - scores: output scores from the model
        - labels: true labels
        - loss_fn: loss function

    """
    if scores.dim() == 3:
        batch_size, num_classes, seq_len = scores.size()
        scores_flat = scores.permute(0, 2, 1).contiguous().view(-1, num_classes)
        labels_flat = labels.contiguous().view(-1)
    else:
        scores_flat = scores
        labels_flat = labels
    loss = loss_fn(scores_flat, labels_flat)
    return loss


def get_optimizer(net, lr, weight_decay):
    """
    Return the optimizer (Adam) you will use to train the model.

    Input:
        - net: model
        - lr: initial learning_rate
        - weight_decay: weight_decay in optimizer
    """
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    return {'optimizer': optimizer, 'scheduler': scheduler}


def train_model(net, trn_loader, val_loader, optim, num_epoch=15, collect_cycle=30,
        device='cpu', verbose=True):
    """
    Train the model.

    Input:
        - net: model
        - trn_loader: dataloader for training data
        - val_loader: dataloader for validation data
        - optim: optimizer
        - num_epoch: number of epochs to train
        - collect_cycle: how many iterations to collect training statistics
        - device: device to use
        - verbose: whether to print training details
    Return:
        - best_model: the model that has the best performance on validation data
        - stats: training statistics
    """
    train_loss, train_loss_ind, val_loss, val_loss_ind = [], [], [], []
    num_itr = 0
    best_model, best_accuracy = None, 0

    optimizer = optim['optimizer']
    scheduler = optim['scheduler']
    
    
    loss_fn = get_loss_fn()
    if verbose:
        print('------------------------ Start Training ------------------------')
    t_start = time.time()
    for epoch in range(num_epoch):
        for inputs, pos_ids in trn_loader:
            inputs = {key: value.to(device) if key!='seq_lens' else value for key, value in inputs.items()}
            pos_ids = pos_ids.to(device)
            outputs = net(**inputs)
            loss = calculate_loss(outputs, pos_ids, loss_fn)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            num_itr += 1
            
            if num_itr % collect_cycle == 0:
                train_loss.append(loss.item())
                train_loss_ind.append(num_itr)
                
        if verbose:
            print('Epoch No. {0}--Iteration No. {1}-- batch loss = {2:.4f}'.format(
                epoch + 1,
                num_itr,
                loss.item()
                ))

        accuracy, loss = get_validation_performance(net, loss_fn, val_loader, device)
        val_loss.append(loss)
        val_loss_ind.append(num_itr)
        if verbose:
            print("Validation accuracy: {:.4f}".format(accuracy))
            print("Validation loss: {:.4f}".format(loss))
        if accuracy > best_accuracy:
            best_model = copy.deepcopy(net)
            best_accuracy = accuracy
    
    t_end = time.time()
    if verbose:
        print('Training lasted {0:.2f} minutes'.format((t_end - t_start)/60))
        print('------------------------ Training Done ------------------------')
    stats = {'train_loss': train_loss,
             'train_loss_ind': train_loss_ind,
             'val_loss': val_loss,
             'val_loss_ind': val_loss_ind,
             'accuracy': best_accuracy,
    }

    return best_model, stats

def get_validation_performance(net, loss_fn, data_loader, device):
    """
    Evaluate model performance.
    Input:
        - net: model
        - loss_fn: loss function
        - data_loader: data to evaluate, i.e. val or test
        - device: device to use
    Return:
        - accuracy: accuracy on validation set
        - loss: loss on validation set
    """
    net.eval()
    y_true = []
    y_pred = []
    total_loss = []

    with torch.no_grad():
        for inputs, pos_ids in data_loader:
            inputs = {key: value.to(device) if key!='seq_lens' else value for key, value in inputs.items()}
            pos_ids = pos_ids.to(device)
            outputs = net(**inputs)
            loss = calculate_loss(outputs, pos_ids, loss_fn)
            pred = outputs.argmax(dim=1)
            total_loss.append(loss.item())
            y_true.append(pos_ids)
            y_pred.append(pred)
    correct = 0
    total = 0
    for i in range(len(y_true)):
        mask = (y_true[i] != 0)
        correct += ((y_pred[i] == y_true[i]) & mask).sum().item()
        total += mask.sum().item()
    accuracy = correct / total if total > 0 else 0
    total_loss = sum(total_loss) / len(total_loss)
    return accuracy, total_loss