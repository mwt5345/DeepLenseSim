from setuptools import setup

with open("README.md",'r') as f:
    long_description = f.read()

setup(
   name='DeepLenseSim',
   version='0.1',
   description='Simulations for DeepLense Project',
   license="MIT",
   long_description=long_description,
   author='Michael W. Toomey',
   author_email='michael_toomey@brown.edu',
   packages=['deeplense'],
)
