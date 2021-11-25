# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from tqdm.auto import tqdm

from definitive_dyn_indicators.utils.xtrack_engine import get_lhc_mask, xtrack_engine


# %%
OUTDIR = '../tmp'


# %%
samples = 3
n_turns = 10

extent = (0.1e-3, 1e-3)
x = np.linspace(*extent, samples)
xx, yy = np.meshgrid(x, x)
x_f = xx.flatten()
y_f = yy.flatten()
px_f = np.zeros_like(x_f)
py_f = np.zeros_like(x_f)


# %%
engine = xtrack_engine(context="CUDA")


# %%
data_true = engine.track(x_f, px_f, y_f, py_f, n_turns*10)


# %%
engine = xtrack_engine(context="CUDA")


# %%
data_half = engine.track(x_f, px_f, y_f, py_f, n_turns)

for i in tqdm(range(9)):
    with open(os.path.join(OUTDIR, 'engine.pkl'), 'wb') as f:
        pickle.dump(engine, f)

    with open(os.path.join(OUTDIR, 'engine.pkl'), 'rb') as f:
        engine = pickle.load(f)

    data_maybe = engine.keep_tracking(n_turns)


# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

ax1.imshow(data_true[4].reshape(samples, samples),
           extent=extent + extent, origin='lower')
ax2.imshow(data_maybe[4].reshape(samples, samples),
           extent=extent + extent, origin='lower')
ax3.imshow((data_true[4]-data_maybe[4]).reshape(samples, samples),
           extent=extent + extent, origin='lower')


# %%
(data_true[4]-data_maybe[4]).reshape(samples, samples)


# %%
plt.figure()

plt.imshow(np.log10(np.absolute((data_true[0]-data_maybe[0])/data_true[0])).reshape(samples, samples),
           extent=extent + extent, origin='lower')

plt.colorbar(label="Relative error [log10]")


# %%



