# DeepLenseSim

Software to simulate strong lensing images for the DeepLense project.

## Data sets

Scripts used to create data sets and link to data sets are located in each of the Model_# folders.

## Installation

First install [colossus](https://bdiemer.bitbucket.io/colossus/cosmology_cosmology.html), [lenstronomy](https://github.com/sibirrer/lenstronomy), and [pyHalo](https://github.com/dangilman/pyHalo)

```console
foo@bar:~$ pip install colossus
foo@bar:~$ pip install lenstronomy==1.9.2
foo@bar:~$ git clone https://github.com/dangilman/pyHalo.git
foo@bar:~$ cd pyHalo
foo@bar:~/pyHalo$ python setup.py develop
foo@bar:~/pyHalo$ cd ..
```


Then clone this repository to your machine and install with setup.py

```console
foo@bar:~$ git clone https://github.com/mwt5345/DeepLenseSim.git
foo@bar:~$ cd DeepLenseSim
foo@bar:~/DeepLenseSim$ python setup.py install
```

## Papers
[![](https://img.shields.io/badge/arXiv-1909.07346%20-red.svg)](https://arxiv.org/abs/1909.07346) [![](https://img.shields.io/badge/arXiv-2008.12731%20-red.svg)](https://arxiv.org/abs/2008.12731) [![](https://img.shields.io/badge/arXiv-2112.12121%20-red.svg)](https://arxiv.org/abs/2112.12121)
