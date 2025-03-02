'''
Author: Yang Jialong
Date: 2024-11-11 17:33:56
LastEditTime: 2025-02-25 16:34:05
Description: 早停机制
'''
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, save_path="."):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        
    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}, validation_loss: {val_loss} < best_score: {-1 * self.best_score}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    
    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        #模型保存
        if self.save_path is not None:
            if isinstance(model, torch.nn.DataParallel):
                torch.save(model.module.state_dict(), self.save_path + 'model.pth')
            else:
                torch.save(model.state_dict(), self.save_path + 'model.pth')
        self.val_loss_min = val_loss