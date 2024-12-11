from ipywidgets import interact, fixed, interact_manual, FloatSlider, IntSlider
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
from os.path import join, exists
from os import makedirs
import pandas as pd
import time

class Timer:
    def __init__(self, msg='Time elapsed'):
        self.msg = msg
    def __enter__(self):
        self.start = time.time()
        return self
    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        print(f'{self.msg}: {duration:.2f}s')

class Event:
    __slots__ = 't', 'x', 'y', 'p'
    def __init__(self, t, x, y, p):
        self.t = t
        self.x = x
        self.y = y
        self.p = p
    def __repr__(self):
        return f'Event(t={self.t:.3f}, x={self.x}, y={self.y}, p={self.p})'

def normalize_image(image, percentile_lower=1, percentile_upper=99):
    mini, maxi = np.percentile(image, (percentile_lower, percentile_upper))
    if mini == maxi:
        return 0 * image + 0.5  # gray image
    return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)

class EventData:
    def __init__(self, event_list, width, height):
        self.event_list = event_list
        self.width = width
        self.height = height

    def add_frame_data(self, data_folder, max_frames=100):
        timestamps = np.genfromtxt(join(data_folder, 'image_timestamps_corrected.txt'), max_rows=int(max_frames))
        frames = []
        frame_timestamps = []
        with open(join(data_folder, 'image_timestamps_corrected.txt')) as f:
            for line in f:
                fname, timestamp = line.split(' ')
                timestamp = float(timestamp)
                frame = cv2.imread(join(data_folder, fname), cv2.IMREAD_GRAYSCALE)
                if not (frame.shape[0] == self.height and frame.shape[1] == self.width):
                    continue
                frames.append(frame)
                frame_timestamps.append(timestamp)
                if timestamp >= self.event_list[-1].t:
                    break
        frame_stack = normalize_image(np.stack(frames, axis=0))
        self.frames = [f for f in frame_stack]
        self.frame_timestamps = frame_timestamps

def load_events(path_to_events, n_events=None):
    print('Loading events...')
    header = pd.read_csv(path_to_events, delim_whitespace=True, names=['width', 'height'],
                         dtype={'width': np.int32, 'height': np.int32}, nrows=1)
    width, height = header.values[0]
    print(f'width, height: {width}, {height}')
    event_pd = pd.read_csv(path_to_events, delim_whitespace=True, header=None,
                              names=['t', 'x', 'y', 'p'],
                              dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int8},
                              engine='c', skiprows=1, nrows=n_events, memory_map=True)
    event_list = []
    for event in event_pd.values:
        t, x, y, p = event
        event_list.append(Event(t, int(x), int(y), -1 if p < 0.5 else 1))
    print('Loaded {:.2f}M events'.format(len(event_list) / 1e6))
    return EventData(event_list, width, height)

def complementary_filter(event_data, cutoff_frequency=5.0, output_folder='filtered_images'):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
    events_per_frame = 5e4
    
    # Create output folder if it doesn't exist
    if not exists(output_folder):
        makedirs(output_folder)
    
    with Timer('Reconstruction'):
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        frame_idx = 0
        max_frame_idx = len(frames) - 1
        log_frame = np.log(frames[0] + 1)
        for i, e in enumerate(events):
            if frame_idx < max_frame_idx:
                if e.t >= frame_timestamps[frame_idx + 1]:
                    log_frame = np.log(frames[frame_idx + 1] + 1)
                    frame_idx += 1
            beta = math.exp(-cutoff_frequency * (e.t - time_surface[e.y, e.x]))
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] \
                                    + (1 - beta) * log_frame[e.y, e.x] + 0.1 * e.p
            time_surface[e.y, e.x] = e.t
            if i % events_per_frame == 0:
                beta = np.exp(-cutoff_frequency * (e.t - time_surface))
                image_state = beta * image_state + (1 - beta) * log_frame
                time_surface.fill(e.t)
                image_list.append(np.copy(image_state))
                # Save the image
                image_path = join(output_folder, f'filtered_image_{i // events_per_frame}.png')
                plt.imsave(image_path, normalize_image(image_state), cmap='gray')
    print('Images saved in the folder:', output_folder)

with Timer('Loading'):
    n_events = 5e5
    path_to_events = 'data/boxes_6dof/events.zip'
    event_data = load_events(path_to_events, n_events)        

event_data.add_frame_data('data/boxes_6dof')

# Run the complementary filter and save images in a new folder
complementary_filter(event_data, cutoff_frequency=5.0, output_folder='filtered_images')