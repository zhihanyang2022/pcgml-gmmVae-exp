import torch

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        def identity(x, y): return x, y
        self.func = identity if func is None else func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))
               
def preprocess(x, y):
    return x.double().to(torch.device('cuda')), y.double().to(torch.device('cuda'))



    