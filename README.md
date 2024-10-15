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

## Run the alignment & evaluation

```bash
$ python run_alignment.py

...
[23:08:44.675041] Current position: 3886
100% (3887 of 3887) |###################################| Elapsed Time: 0:00:09 Time:  0:00:09
Evaluation Results: {
    "mean": 90.9091,
    "median": 66.6667,
    "std": 89.9954,
    "skewness": 6.9744,
    "kurtosis": 65.1447,
    "50ms": 0.2426,
    "100ms": 0.7426,
    "200ms": 0.9559,
    "300ms": 0.9632,
    "500ms": 0.9632,
    "1000ms": 0.9706
}
```

The evaluation results are printed at the end of the alignment process. 
The meaning of each field is as follows:

- mean: mean deviation of frame-wise aligned results (in milliseconds)
- median: median deviation of frame-wise aligned results (in milliseconds)
- std: standard deviation of frame-wise aligned results (in milliseconds)
- skewness: skewness of aligned results
- kurtosis: kurtosis of aligned results
- 50ms: ratio of successful aligned results within 50ms
- 100ms: ratio of successful aligned results within 100ms
- 200ms: ratio of successful aligned results within 200ms
- 300ms: ratio of successful aligned results within 300ms
- 500ms: ratio of successful aligned results within 500ms
- 1000ms: ratio of successful aligned results within 1000ms

