import os
from tqdm import tqdm

GPU_ID = "3"
epsilon = 64.0
mu_list = [0.0, 0.001, 0.01, 0.1]

# omega_x, omega_y = 0.31, 0.32
# omega_x, omega_y = 0.28, 0.31
omega_x, omega_y = 0.168, 0.201

for omega_x, omega_y, max_val in tqdm([(0.28, 0.31, 0.5), (0.168, 0.201, 0.8)]):
    for mu in tqdm(mu_list):
        # run computing.py script
        print(
            f'Running computing.py with mu={mu} and epsilon={epsilon} on GPU {GPU_ID}')
        os.system(
            f"python computing.py --gpu_id {GPU_ID} --epsilon {epsilon} --mu {mu} --omega_x {omega_x} --omega_y {omega_y} --max_x {max_val} --max_y {max_val}")
