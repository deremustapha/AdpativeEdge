import numpy as np

class EarlyStopping:
    def __init__(self, patience=7, delta=0):

        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.early_stop = False
        self.val_min = float('inf')
    
    def __call__(self, val):
        if val < self.val_min:
            self.val_min = val
            self.counter = 0
        elif val > self.val_min+self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return self.early_stop
        return False


# class EarlyStoppingGeneralization:
#     def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print)
#         self.patience = patience
#         self.verbose = verbose
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.val_loss_min = float('inf') #np.Inf
#         self.delta = delta
#         self.path = path
#         self.trace_func = trace_func
    
#     def __call__(self, val_loss, model):
#         score = -val_loss

#         if self.best_score is None:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#         elif score < self.best_score + self.delta:
#             self.counter += 1
#             self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_score = score
#             self.save_checkpoint(val_loss, model)
#             self.counter = 0