# CalibrateGan
```
1) Download SUN Dataset from here:
2) wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
3) tar -xzf SUN2012.tar.gz
- move SUN2012/ to the datasets/ folder
- make sure we have a processed/ directory also

4) Then run `python3 generate_synthetic_samples.py`

5) Make the output directory where we're going to put the dataset. This step 5 and step 4 are separated just to support generalizability in case we support future dataset structures in the future

- `mkdir processed/SUN2012_cleaned`
- `mkdir processed/SUN2012_cleaned/input`
- `mkdir processed/SUN2012_cleaned/output`

Move and order the downloads to the input and output folders you specified
8) `python3 clean_paths.py` 

Train a model!
9) Set the parameters inside of the train.py function. You can set `load=True` if you want to load a prior model (then specify those paths)
10) Run `python3 train.py`

Just wanna use one of my old models to test one image?
11) (edit the paramters in test.py to the desired paths) then run `python3 test.py`

Notes:
- Make sure there is a logs folder. All the images and models will be saved there (in the directory you specify)
- In this special version I turned it, I saved some models and some other images.
- gcptrial-sat2 is the latest and greatest models and validation images. You can check them out! 

```
Note: you need tensorflow 1.15 to run these models.

