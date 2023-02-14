# python-match
Python implementation of real-time audio-to-audio alignment


The online time warping (OLTW) algorithm is implemented in this system, based on the paper MATCH: A MUSIC ALIGNMENT TOOL CHEST.

Tested on Python 3.9 & 3.10 (conda)

## Clone Repository

```bash
$ git clone git@github.com:laurenceyoon/python-match.git
$ cd python-match
```

## Setting a conda environment from environment file

```bash
$ conda env create -f environment.yml
$ conda activate match
$ jupyter notebook
```

## Setting a new conda environment from scratch

```bash
$ conda create --name match python=3.9 jupyter matplotlib
$ conda activate match

# reinstall soundfile
$ pip install --upgrade --force-reinstall soundfile

$ conda install -c conda-forge librosa
$ conda install -c conda-forge pyaudio
$ conda install -c conda-forge numpy
```
