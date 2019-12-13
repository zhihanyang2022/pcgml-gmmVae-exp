import functools
import inspect
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def set_2d_labels(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)

def set_3d_labels(ax, xlabel, ylabel, zlabel):
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_zlabel(zlabel)
    
def ifnone(a, b):
    return b if a is None else a
    
def set_title_and_labels_for_dim(dim):
    def set_title_and_labels(f):
        @functools.wraps(f)
        def new_f(*args, **kwargs):
            output = f(*args, **kwargs)
            
            # https://docs.python.org/3/library/inspect.html#inspect.getcallargs
            # https://docs.python.org/3/library/inspect.html#inspect.Signature.bind
            # https://docs.python.org/3/library/inspect.html#inspect.BoundArguments
            sig = inspect.signature(f)
            bind = sig.bind(*args, **kwargs); 
            bind.apply_defaults()  # not useful since it overwrites passed-in values
            f_args = bind.arguments
            
            fig = f_args.get('self').fig
            ax = fig.axes[f_args.get('pos')-1]
            
            if not f_args.get('overlay'):
            
                labels = f_args.get('labels')
                if dim == 2: 
                    labels = ifnone(labels, ('x', 'y'))
                    set_2d_labels(ax, *labels)
                elif dim == 3: 
                    labels = ifnone(labels, ('x', 'y', 'z'))
                    set_3d_labels(ax, *labels)

                title = f_args.get('title')
                ax.set_title(ifnone(title, ''))
               
                ax.grid(f_args.get('grid'))
            
            return output
        return new_f
    return set_title_and_labels

def check_figure_overflow(f):
    @functools.wraps(f)
    def new_f(*args, **kwargs):
        f_args = inspect.getcallargs(f, *args, **kwargs)  # deprecated
        num_slots = f_args.get('self').nrows * f_args.get('self').ncols
        assert f_args.get('pos') <= num_slots, AssertionError('Subplot outside available slots.')
        return f(*args, **kwargs)
    return new_f

class GraphPaper(object):
    
    def __init__(self, height, width, nrows, ncols):
        self.fig = plt.figure(figsize=(width, height))
        self.nrows, self.ncols = nrows, ncols
       
    @set_title_and_labels_for_dim(2)
    @check_figure_overflow
    def scatter_2d(
        self, 
        pos:int,
        xs:np.array, ys:np.array,
        color:str=None, dot_size:int=1, label:str=None, overlay:bool=False,
        title:str=None, labels:tuple=None,
        grid:bool=False
    )->None:
        ax = self.fig.add_subplot(self.nrows, self.ncols, pos)
        ax.scatter(xs, ys, s=dot_size, label=label)
        return ax
    
    @set_title_and_labels_for_dim(2)
    @check_figure_overflow
    def plot_2d(
        self, 
        pos:int,
        xs:np.array, ys:np.array,
        label:str=None, overlay:bool=False,
        title:str=None, labels:tuple=None, 
        grid:bool=False
    )->None:
        ax = self.fig.add_subplot(self.nrows, self.ncols, pos)
        ax.plot(xs, ys, label=label)
        return ax
    
    @check_figure_overflow
    def heatmap_2d(
        self,
        pos:int,
        mat:np.array,
        ax_off:bool=False, title:str=None,
    ):
        ax = self.fig.add_subplot(self.nrows, self.ncols, pos)
        ax.matshow(mat)
        ax.axis('off' if ax_off else 'on')
        return ax
                
    @set_title_and_labels_for_dim(3)
    @check_figure_overflow
    def scatter_3d(
        self, 
        pos:int,  
        xs:np.array, ys:np.array, zs:np.array, 
        color:str=None, dot_size:int=1, label:str=None, overlay:bool=False,
        title:str=None, labels:tuple=None,
        grid:bool=False
    )->None:
        if overlay: ax = self.fig.axes[pos-1]
        else: ax = self.fig.add_subplot(self.nrows, self.ncols, pos, projection='3d')
        ax.scatter(xs, ys, zs, s=dot_size, label=label, color=color)
        return ax
        
    def show(self, grid_for_all=False, legend_for_all=False):
        if grid_for_all: 
            for ax in self.fig.axes: ax.grid(True)
        if legend_for_all: 
            for ax in self.fig.axes: ax.legend()
        self.fig.tight_layout()
        self.fig.show()
        
    def save(self, png_name, dpi=300):
        self.fig.savefig(png_name, dpi=dpi)
        
    