import numpy as np

samples = []
for i in range(50):
    samples.append(np.load(f"samples_500k/diffusion_samples_{i}.npy"))
final_samples = np.concatenate(samples, axis = 0)
np.save("diffusion_500k.npy", final_samples)
