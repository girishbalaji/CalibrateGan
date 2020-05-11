# CalibrateGan
```
1) Download SUN Dataset from here:
2) wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
3) tar -xzf SUN2012.tar.gz
4) Run `python3 generate_synthetic_samples.py`

5) Make the output directory where we're going to put the dataset. This step 5 and step 4 are separated just to support generalizability in case we support future dataset structures in the future

- `mkdir processed/SUN2012_cleaned`
- `mkdir processed/SUN2012_cleaned/input`
- `mkdir processed/SUN2012_cleaned/output`

Move and order the downloads to the input and output folders you specified
8) `python3 clean_paths.py` 

Train a model!
9) Set the parameters inside of the train.py function. You can set `load=True` if you want to load a prior model (then specify those paths)
10) Run `python3 train.py`

```
