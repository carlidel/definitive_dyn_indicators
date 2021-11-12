import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from tqdm.auto import tqdm
import pickle

import henon_map_cpp as hm

FORCE_CPU = True

sample_list = np.array([10, 100, 1000, 10000, 100000, 1000000])
time_list = np.array([10, 100, 1000, 10000, 100000, 1000000], dtype=int)

time_data = np.zeros((len(sample_list), len(time_list)))

for i, sample in tqdm(enumerate(sample_list), total=len(sample_list)):
    for j, time in tqdm(enumerate(time_list), total=len(time_list)):
        # Generate data
        x = np.random.uniform(0.1, 0.2, sample)
        px = np.zeros(sample)
        y = np.random.uniform(0.1, 0.2, sample)
        py = np.zeros(sample)

        epsilon = 1.0
        mu = 0.0
        
        engine = hm.henon_tracker(x, px, y, py, 0.168, 0.201, FORCE_CPU)

        # Start timer
        start = dt.datetime.now()
        # Run simulation
        engine.track(time, epsilon, mu)
        # Stop timer
        end = dt.datetime.now()
        # Calculate time
        time_data[i, j] = (end - start).total_seconds()

# Create dataframe with time data
df = pd.DataFrame(time_data, index=sample_list, columns=time_list)

# Name the index
df.index.name = 'Sample size' 

# if not present, create folder
if not os.path.exists('data'):
    os.makedirs('data')

# Save dataframe to pickle
df.to_pickle('data/henon_map_time_data' + ('_CPU' if FORCE_CPU else '_GPU') + '.pkl')

