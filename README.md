# CalibrateGan
```
1) Download SUN Dataset from here:
2) wget http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
3) tar -xzf SUN2012.tar.gz
4) Run `generate_synthetic_samples.py`
5) mkdir SUN2012_cleaned
6) mkdir SUN2012_cleaned/input
7) mkdir SUN2012_cleaned/output

Move and order the downloads

8) ls -v processed/SUN2012/input/*/*.jpg | cat -n | while read n f; do mv -n "$f" "processed/SUN2012_cleaned/input/$n.jpg"; done
9) ls -v processed/SUN2012/output/*/*.jpg | cat -n | while read n f; do mv -n "$f" "processed/SUN2012_cleaned/output/$n.jpg"; done
```
