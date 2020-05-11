# CalibrateGan
```
1) Download SUN Dataset from here:
2) wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
3) tar -xzf SUN2012.tar.gz
4) Run `python3 generate_synthetic_samples.py`
5) mkdir processed/SUN2012_cleaned
6) mkdir processed/SUN2012_cleaned/input
7) mkdir processed/SUN2012_cleaned/output

Move and order the downloads to the input and output folders you specified
8) `python3 clean_paths.py` 

Train a model!
9) Set the parameters inside of the train.py function. You can set `load=True` if you want to load a prior model (then specify those paths)
10) Run `python3 train.py`

```
