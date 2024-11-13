from ipywidgets import interact, fixed, interact_manual, FloatSlider, IntSlider
import math
from matplotlib import rc
rc('animation', html='jshtml')
import numpy as np

# local modules
from util import Timer, Event, normalize_image, animate, load_events, plot_3d, event_slice

import h5py


def high_pass_filter(event_data, cutoff_frequency=5):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    events_per_frame = 2e4
    with Timer('Reconstruction'):
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        for i, e in enumerate(events):
            beta = math.exp(-cutoff_frequency * (e.t - time_surface[e.y, e.x]))
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] + e.p
            time_surface[e.y, e.x] = e.t
            if i % events_per_frame == 0:
                beta = np.exp(-cutoff_frequency * (e.t - time_surface))
                image_state *= beta
                time_surface.fill(e.t)
                image_list.append(np.copy(image_state))
    return animate(image_list, 'High Pass Filter')

def leaky_integrator(event_data, beta=1.0):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    events_per_frame = 2e4
    with Timer('Reconstruction (simple)'):
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        for i, e in enumerate(events):
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] + e.p
            if i % events_per_frame == 0:
                image_list.append(np.copy(image_state))
    fig_title = 'Direct Integration' if beta == 1 else 'Leaky Integrator'
    return animate(image_list, fig_title)

#with h5py.File(f, 'r') as file:
#    def print_h5_structure(name, obj):
#        print(name, obj)       
    # Display the structure of the file
#    file.visititems(print_h5_structure)

#p <HDF5 dataset "p": shape (3780547,), type "|u1">
#t <HDF5 dataset "t": shape (3780547,), type "<i8">
#x <HDF5 dataset "x": shape (3780547,), type "<u2">
#y <HDF5 dataset "y": shape (3780547,), type "<u2">

f = 'data/00a81/events/events.h5'
#events_data = load_events(f, n_events=10000)


import h5py
import numpy as np
import pandas as pd
import os

# Specify the correct directory path where your flow files are located
""" flow_dir = 'data/00a81/flow/'

# List all .h5 files in the flow directory
flow_files = [f for f in os.listdir(flow_dir) if f.endswith('.h5')]

flow_data = []
for file in flow_files:
    file_path = os.path.join(flow_dir, file)  # Combine the directory and filename
    with h5py.File(file_path, 'r') as f:
        # Check the keys in each file to identify which dataset to load
        print(f.keys())
        # Assuming the flow data is under the 'flow' key
        flow_data.append(f['flow'][:])

# Optionally print out flow_data to check what's loaded
print(flow_data) """


with h5py.File(f, 'r', libver='latest') as file:
    t_data = file['t'][:]
    x_data = file['x'][:]
    y_data = file['y'][:]
    p_data = file['p'][:]
