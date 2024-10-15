# python-match
Python implementation of real-time audio-to-audio alignment


The online time warping (OLTW) algorithm is implemented in this system, based on the paper MATCH: A MUSIC ALIGNMENT TOOL CHEST.

Tested on Python 3.12 (conda)

## Clone Repository

```bash
$ git clone git@github.com:laurenceyoon/python-match.git
$ cd python-match
```

## Setting a conda environment from environment file

```bash
$ conda env create -f environment.yml
$ conda activate match
```

If some packages are not installed properly, please install them manually. You can also refer to the `requirements.txt` file.

## Run the code

```bash
$ python run_alignment.py
```

