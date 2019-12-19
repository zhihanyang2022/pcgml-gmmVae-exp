import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
import os
import shutil

class Callback(): 
    sd = {}
    def on_train_begin(self): pass  # make sure to reset self.sd because it is a class attribute
    def on_epoch_begin(self): pass
    def on_batch_begin(self): pass
    def on_loss_begin(self): pass
    def on_backward_begin(self): pass
    def on_backward_end(self): pass
    def on_step_end(self): pass
    def on_batch_end(self): pass
    def on_epoch_end(self): pass
    def on_train_end(self): pass

class CallbackHandler(Callback): 
    
    def __init__(self, cbs):
        self.cbs = cbs
    
    def __call__(self, cb_category:str):
        self.cbs = sorted(self.cbs, key=lambda cb : cb._order)
        for cb in self.cbs: getattr(cb, cb_category)()
    
    def on_train_begin(self): self('on_train_begin')
        
    def on_epoch_begin(self): self('on_epoch_begin')
    
    def on_batch_begin(self): self('on_batch_begin')
        
    def on_loss_begin(self): self('on_loss_begin')
        
    def on_backward_begin(self): self('on_backward_begin')
        
    def on_backward_end(self): self('on_backward_end')
        
    def on_step_end(self): self('on_step_end')
        
    def on_batch_end(self): self('on_batch_end')
    
    def on_epoch_end(self): self('on_epoch_end')
        
    def on_train_end(self): self('on_train_end')

def ifoverwrite(overwrite:bool, file_navigator:str, file_navigator_type:str)->None:
    
    if overwrite:
        
        if file_navigator_type == 'path':
            file_dir = ''.join([folder + '/' for folder in file_navigator.split('/')[:-1]])
            if not (os.path.isfile(file_navigator_type) or os.path.isdir(file_dir)):
                os.makedirs(file_dir, exist_ok=True)
            
        elif file_navigator_type == 'dir':
            file_dir = file_navigator
            if os.path.isdir(file_dir):  # so that overwrite also works when there's nothing to overwrite
                shutil.rmtree(file_dir)
                os.makedirs(file_dir, exist_ok=False)
            
    else:
        assert not (os.path.isfile(file_navigator) or os.path.isdir(file_navigator)), \
        AssertionError(f'{file_navigator} already exists. To overwrite it, pass True to argument `overwrite`.')

class TensorboardCreator(Callback):
    _order=0
    
    def __init__(self, log_dir, overwrite:bool):
        self.log_dir = log_dir
        ifoverwrite(overwrite, log_dir, 'dir')
    
    def on_train_begin(self):
        self.sd.update({'writer':SummaryWriter(log_dir=self.log_dir, flush_secs=2, max_queue=2)})
        
    def on_train_end(self):
        self.sd['writer'].flush()
        self.sd['writer'].close()
        del self.sd['writer']

class MetricLogger(Callback):
    """Log (in a list) and visualize metric values over time."""
    _order=1
    
    def __init__(self, metric_name:str, group:str, on_tensorboard:bool):
        
        self.metric_name = metric_name
        self.group = group
        self.on_tensorboard = on_tensorboard

    def on_train_begin(self):
        self.sd[f'last_{self.metric_name}'] = None
        self.sd[f'{self.metric_name}s'] = []

    def on_epoch_begin(self):
        self.total = 0
        self.num_examples = 0
    
    def on_batch_end(self):
        self.total += self.sd[f'{self.metric_name}_b']
        self.num_examples += self.sd['batch_size'] 
        
    def on_epoch_end(self):
        
        last = self.total / self.num_examples
        self.sd[f'last_{self.metric_name}'] = last
        self.sd[f'{self.metric_name}s'].append(last)
        
        if self.on_tensorboard:
            self.sd['writer'].add_scalar(
                f'{self.group}/{self.metric_name}', 
                self.sd[f'last_{self.metric_name}'], 
                self.sd['epoch']
            )

class MetricsSaver(Callback):
    """
    Log metric values over time into a csv that can be loaded and visualized within a jupyter notebook.
    
    Depend on MetricRecorder to work properly.
    """
    _order=2
    
    def __init__(self, metrics_to_save:list, csv_path:str, overwrite:bool):
        """
        :param metrics_to_save: a list of names of the metrics to log
            - make sure that an accumulator for each metric is available in self.sd
            - make sure to add an 's' to each metric name
            - do not add 'epochs' to this list, since it will be obvious during plotting
        """
        self.metrics_to_save = metrics_to_save
        self.csv_path = csv_path
        ifoverwrite(overwrite, csv_path, 'path')
    
    def on_train_end(self):
        loss_df = pd.DataFrame(np.array([self.sd[m] for m in self.metrics_to_save]).T)
        loss_df.columns = self.metrics_to_save
        loss_df.to_csv(self.csv_path)

class Debugger(Callback):
    _order=999
    def __init__(self, on:bool): self.on = on
    def on_train_begin(self): 
        if self.on: print('finished on_train_begin')
    def on_epoch_begin(self): 
        if self.on: print('finished on_epoch_begin')
    def on_batch_begin(self): 
        if self.on: print('finished on_batch_begin')
    def on_loss_begin(self): 
        if self.on: print('finished on_loss_begin')
    def on_backward_begin(self): 
        if self.on: print('finished on_backward_begin')
    def on_backward_end(self): 
        if self.on: print('finished on_backward_end')
    def on_step_end(self): 
        if self.on: print('finished on_step_end')
    def on_batch_end(self): 
        if self.on: print('finished on_batch_end')
    def on_epoch_end(self): 
        if self.on: print('finished on_epoch_end')
    def on_train_end(self): 
        if self.on: print('finished on_train_end')

class MetricsPrinter(Callback):
    _order=2
    
    def __init__(self, metrics_to_print):
        self.metrics_to_print = metrics_to_print
    
    def on_epoch_end(self):
        for m in self.metrics_to_print:
            print(f'{m}: {self.sd[m]}', end='|')
        print('')

class ModelSaver(Callback):
    _order=2
    
    def __init__(self, model_path:str, overwrite:bool): 
        self.model_path = model_path
        ifoverwrite(overwrite, model_path, 'path')
    
    def on_train_end(self):
        torch.save(self.sd['model'].state_dict(), self.model_path)

