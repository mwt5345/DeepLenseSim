import matplotlib.pyplot as plt
import numpy as np

from deeplense.lens import DeepLens

lens = DeepLens()
lens.make_single_halo(1e12)
lens.make_no_sub()
lens.make_source_light()
lens.simple_sim()

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
