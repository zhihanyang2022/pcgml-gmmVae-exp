from IPython.display import clear_output
from ipywidgets.widgets import *

import numpy as np

def ifnone(a, b): return a if (a is not None) else b

def hbox(label:str=None, widgets:list=None):
    if label is not None: return HBox([Label(label)])
    else: return HBox(widgets)

def button(description:str, func) -> Button:
    button = Button(description=description)
    button.on_click(func)
    return button

def intslider(default, max, orientation='vertical'):
    w = IntSlider(
        value=default,
        min=0,
        max=max,
        step=1,
        description='',
        disabled=False,
        continuous_update=False,
        orientation=orientation,
        readout=True,
        readout_format='d',
    )
    w.layout.height='95%'
    return w

def output(): 
    return Output() #layout={'border': '1px solid grey'})

def fmap_shape_from_tuple(input_shape:tuple, out_channels:int, kernel_size:int, stride:int, padding:int, dilation:int=1, downsize=True, output_padding=0) -> tuple:
    """
    Compute the size of feature maps given the input size.

    param: input_size: a tuple with format (c, h, w)
    Reference:
    * https://pytorch.org/docs/stable/nn.html#conv2d
    * https://pytorch.org/docs/stable/nn.html#maxpool2d
    """
    if downsize:
        formula = lambda x : np.floor((x + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)
    else:
        formula = lambda x : np.ceil((x - 1) * stride - 2 * padding + kernel_size + output_padding)

    h, w = input_shape[1], input_shape[2]
    h_out, w_out = formula(h), formula(w)
    shape_out = (int(out_channels), int(h_out), int(w_out))
    return shape_out

class VAEDesigner():

    def __init__(self, num_layers:int=3, input_shape=(3, 28, 28), downsize=True, paddings=None):
        
        self.num_layers = num_layers
        self.input_shape = input_shape
        self.downsize = downsize
        self.paddings = paddings
        
        if self.downsize: mode = 'DOWNSAMPLING'
        else: mode = 'UPSAMPLING'
        
        # hboxs of titles
        self.title_kernels  = hbox(label=f'Number of kernels of each {mode} layer')
        self.title_strides  = hbox(label=f'Size of strides of each {mode} layer')
        self.title_padding = hbox(label='Size of paddings for input arrays')
        self.title_shapes   = hbox(label=f'Output shape of each {mode} layer')        
        
        # sliders
        sliders_nkernels = [intslider(default=12, max=128) for _ in range(3)]
        sliders_skernels = [intslider(default=3, max=5) for _ in range(3)]
        sliders_strides  = [intslider(default=2, max=10) for _ in range(3)]
        sliders_paddings = [intslider(default=0, max=10, orientation='horizontal')]
        
        # hboxs of sliders
        self.hbox_nkernels = hbox(widgets=sliders_nkernels)
        self.hbox_skernels = hbox(widgets=sliders_skernels)
        self.hbox_strides = hbox(widgets=sliders_strides)
        self.hbox_padding = hbox(widgets=sliders_paddings)
        
        # buttons
        button_add_layer     = button(description='ADD LAYER', func=self.add_layer)
        button_remove_layer  = button(description='REMOVE LAYER', func=self.remove_layer)
        button_update_shapes = button(description='UPDATE SHAPES', func=self.update_output_widgets)
        self.hbox_buttons = hbox(widgets=[button_add_layer, button_remove_layer, button_update_shapes])
        
        self.hbox_shapes = hbox(widgets=[output() for i in range(self.num_layers)])
    
        self.update_layouts_for_hboxs()
        self.refresh_interface()
        

    # fundamental functions
    
    
    def cloop_thru_hboxs(self, 
                       hboxs_names:list, 
                       yield_per_step:bool=True, 
                       func=None):
        """
        Loop through widgets in several HBox's concurrently (hence the name cloop).
        
        For example, looping through [1, 2, 3] and [4, 5, 6] concurrently means to return 
        the following tuples on at a time (1, 4), (2, 5) and (3, 6).
        
        param: hboxs_names: the list of HBox's to be looped through
        param: yield_per_step: whether something need to be yielded at each iteration, default True
        param: func: if passed in, something is done (inplace) at each iteration, default None
        """
        
        def identity(i, w): pass
        func = ifnone(func, identity)

        list_of_tuples_of_children = [getattr(self, name, None).children for name in hboxs_names]
        assert None not in list_of_tuples_of_children, print('Children cannot be None.')

        for i, batch in enumerate(zip(*list_of_tuples_of_children)):
            if len(list_of_tuples_of_children) == 1: 
                func(i, batch[0])
                yield batch[0]
            else: 
                for w in batch: func(i, w) 
                yield batch
                
                
    def forpass(self, iterator): 
        """Loop through any generator without receiving its outputs; useful for inplace operations."""
        for _ in iterator: pass
                        
        
    def set_layout_for_box(self, 
                           box:HBox, 
                           height:str=None, 
                           align_items:str=None, 
                           justify_content:str=None) -> None:
        """
        Specify value for the `height`, `align_items` and `justify_items` attributes of a box.

        param: box: the box whose attributes will be setted
        param: height: the vertical height of the box in px, default None
        param: align_items: the location of widgets with respect to the horizontal midline of the box
        param: justify_content: the style by which widgets are organized horizontally
        """
        if height          is not None: box.layout.height = height
        if align_items     is not None: box.layout.align_items = align_items
        if justify_content is not None: box.layout.justify_content = justify_content
            
            
    # customizable functions
            
        
    def update_layouts_for_hboxs(self) -> None:
        for hbox in [self.hbox_nkernels, self.hbox_skernels, self.hbox_strides]: 
            self.set_layout_for_box(hbox, height='170px', align_items='center', justify_content='space-around')
        for hbox in [self.hbox_padding, self.hbox_shapes]:
            self.set_layout_for_box(hbox, height='30px', align_items='center', justify_content='space-around')

            
    def update_names_for_sliders(self):
        def set_name(i, w): w.description = 'L' + str(i)
        self.forpass(self.cloop_thru_hboxs(['hbox_nkernels', 'hbox_skernels', 'hbox_strides'], yield_per_step=False, func=set_name))
        
        
    def clear_output_widgets(self):
        for out in self.cloop_thru_hboxs(['hbox_shapes']): out.clear_output(wait=True)
        
        
    def update_output_widgets(self, b=None):
        self.clear_output_widgets()
        
        input_shape = self.input_shape
        padding = self.hbox_padding.children[0].value
        for i, (out, num_kernels, kernel_size, num_strides) in enumerate(self.cloop_thru_hboxs(['hbox_shapes', 'hbox_nkernels', 'hbox_skernels', 'hbox_strides'])):
            output_shape =  fmap_shape_from_tuple(
                input_shape=input_shape, 
                out_channels=num_kernels.value, 
                kernel_size=kernel_size.value, 
                stride=num_strides.value, 
                padding=0 if self.paddings is None else self.paddings[i], 
                downsize=self.downsize, 
            )
            with out: print(output_shape)
            input_shape = output_shape
#             padding = 0
              
            
    # functions activated by buttons
        
        
    def add_layer(self, b):
        if not self.num_layers > 7:
            self.hbox_nkernels.children += (intslider(default=12, max=64), )
            self.hbox_skernels.children += (intslider(default=2, max=5), )
            self.hbox_strides.children += (intslider(default=2, max=10), )
            self.hbox_shapes.children += (output(), )
            self.num_layers += 1
            self.refresh_interface()
        
        
    def remove_layer(self, b):
        if self.num_layers > 1:
            
            hbox_nkernels = list(self.hbox_nkernels.children)
            self.hbox_nkernels = hbox(widgets=hbox_nkernels[:-1])
            
            hbox_skernels = list(self.hbox_skernels.children)
            self.hbox_skernels = hbox(widgets=hbox_skernels[:-1])

            hbox_strides = list(self.hbox_strides.children)
            self.hbox_strides = hbox(widgets=hbox_strides[:-1])
            
            hbox_shapes = list(self.hbox_shapes.children)
            self.hbox_shapes = hbox(widgets=hbox_shapes[:-1])

            self.update_layouts_for_hboxs()
            
            self.num_layers -= 1
            self.refresh_interface()
        
        
    def refresh_interface(self):
        clear_output(wait=True)
        self.update_names_for_sliders()
        self.update_output_widgets()
        display(VBox([
                self.hbox_buttons,
                self.title_shapes,
                self.hbox_shapes,
#                 self.title_padding,
#                 self.hbox_padding,
                self.title_kernels, 
                self.hbox_nkernels,
                self.hbox_skernels,
                self.title_strides,
                self.hbox_strides,
        ]))

