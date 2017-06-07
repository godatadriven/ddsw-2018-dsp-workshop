# Signal Processing for Data Science

Dutch Data Science Week Workshop

### Introduction (09:00 - 10:45)
- Signal processing applications
- Convolution
- Fourier analysis

### Feature engineering (11:00 - 12:30)
- 1D signals: filter banks, pooling
- 2D signals: bag-of-words

### Feature learning (13:30 - 14:30)
- PCA / LDA
- Convolutional neural network
    
### Speech processing hackathon (15:00 - ?)
- Speaker / gender / age / nationality recognition...
- ...based on Fourier / bag-of-words / convnet

## Getting started

This repo uses conda's virtual environment for __Python 3__.

Install (mini)conda if not yet installed by following the instructions on https://conda.io/docs/install/quick.html.

Create the environment using the environment.yml config file:
```sh
$ cd /sp/for/datascience/ddsw/folder
$ conda env create -f environment.yml
$ source activate dsp
```

If that doesn't work, inspect the environment.yml and install the packages you need one by one.

Then, start a Jupyter notebook server
```sh
$ jupyter-notebook
```
and pick the notebook you want to run.

Or view a notebook as a presentation (if applicable; this is indicated by the presence of 'Slide Type' dropdowns in the top right of the notebook cells)
```sh
$ # possible presentations are intro.ipynb, feature-engineering.ipynb and feature-learning.ipynb
$ jupyter nbconvert intro.ipynb --to slides --post serve
```

## Hackathon

In the hackathon we will apply Digital Signal Processing methods to audio singals. Provided is a handful of audio samples and associated metadata of the speakers. Based on that, we'll try to recognize the speaker's identity/gender/age/etc using machine learning. See `hackathon/hackathon.ipynb` for the assignment; or a couple of my implementations in `hackathon/hackathon-answers.ipynb`.

## Lab notebooks

In `fourier-lab.ipynb` we will look at how fourier analysis can help us to find periodicity in a timeseries, and we will use FFT to identify outliers in the sunspots dataset. As a bonus exercise (more of a thought experiment), we look at how we can use FFT for extrapolation. See `fourier-lab-answers.ipynb` for possible solutions.

In `convnet-lab.ipynb` we implement an end-to-end transfer learning pipeline using Keras. We will train a CNN on the first 5 digits of MNIST, and use the features learnt at this step to classify the last 5. A student should be able to solve this exercise by following the lecture material in `convnet.ipynb`; solutions are in `convnet-lab-answers.ipynb`
