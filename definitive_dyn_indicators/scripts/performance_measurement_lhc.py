import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt
import os
from tqdm.auto import tqdm
import pickle
import time as tm

from utils.xtrack_engine import get_lhc_mask, xtrack_engine

FORCE_CPU = True

sample_list = np.array([1, 10, 100, 1000, 10000])
time_list = np.array([1, 10, 100, 1000, 10000], dtype=int)

time_data = np.zeros((len(sample_list), len(time_list)))

mask = get_lhc_mask()
print(mask)

engine = xtrack_engine(
    mask,
    xy_wall=1.0,
    context="CPU" if FORCE_CPU else "CUDA"
)

for i, sample in tqdm(enumerate(sample_list), total=len(sample_list)):
    for j, time in tqdm(enumerate(time_list), total=len(time_list)):
        # Generate data
        x = np.random.uniform(1e-4, 2e-4, sample)
        px = np.zeros(sample)
        y = np.random.uniform(1e-4, 2e-4, sample)
        py = np.zeros(sample)
        
        # Start timer
        start = tm.perf_counter()
        # Run simulation
        engine.track(x, px, y, py, time)
        # Stop timer
        end = tm.perf_counter()
        # Calculate time
        time_data[i, j] = (end - start)

# Create dataframe with time data
df = pd.DataFrame(time_data, index=sample_list, columns=time_list)

# Name the index
df.index.name = 'Sample size'

# if not present, create folder
if not os.path.exists('data'):
    os.makedirs('data')

# Save dataframe to pickle
df.to_pickle('data/lhc_time_data' +
             ('_CPU' if FORCE_CPU else '_GPU') + '.pkl')
