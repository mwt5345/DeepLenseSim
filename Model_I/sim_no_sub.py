import matplotlib.pyplot as plt
import numpy as np
import random

from deeplense.lens import DeepLens


# Number of sims
num_sim = int(1e4)

for i in range(num_sim):
    lens = DeepLens()
    lens.make_single_halo(1e12)
    lens.make_no_sub()
    lens.make_source_light()
    lens.simple_sim()
    File = lens.image_real
    np.save('/users/mtoomey/scratch/deeplense/Model_I/no_sub/no_sub_sim_' + str(random.getrandbits(128)),File)



if False:
    plt.figure(figsize=(10,5))
    plt.subplot(2,2,1)
    plt.imshow(lens.image_real)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(np.sqrt(lens.image_real))
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(lens.poisson)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.imshow(lens.bkg)
    plt.colorbar()
    plt.show()
