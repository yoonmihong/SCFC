import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score_r2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss_min_r2 = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, val_r2, model):

        score = -val_loss
        score_r2 = -val_r2

        if self.best_score is None and self.best_score_r2 is None:
            self.best_score = score
            self.best_score_r2 = score_r2
            self.save_checkpoint(val_loss, val_r2, model)
        elif (score < self.best_score - self.delta):# or (score_r2 < self.best_score_r2 - self.delta):
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score_r2 = score_r2
            self.save_checkpoint(val_loss, val_r2, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_r2, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            # self.trace_func(f'Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}). Validation std decreased ({self.val_loss_min_r2:.3f} --> {val_r2:.3f}). Saving model ...')
             self.trace_func(f'Validation loss decreased ({self.val_loss_min:.3f} --> {val_loss:.3f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        self.val_loss_min_r2 = val_r2
