import numpy as np

def generate_random_image(seed, shape=(32, 32, 3)):
    np.random.seed(seed)
    return np.random.rand(*shape)

num_seeds = 100
random_images = [generate_random_image(seed) for seed in range(num_seeds)]