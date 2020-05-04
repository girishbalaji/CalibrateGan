from Pix2PixModel import Pix2PixModel
import tensorflow as tf
import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
import glob
import random
import pickle

def split_data(datadir, metadatafile):
    pathexp = "{}/*.jpg".format(datadir)
    print("regexpath: {}".format(pathexp))
    fnames = glob.glob(pathexp)

    basenames = list(map(os.path.basename, fnames))
    num_names = len(fnames)
    random.shuffle(basenames)

    print("Found: {} files".format(num_names))

    metadata = {}
    metadata["training_idx"] = 0
    metadata["val_idx"] = 0
    metadata["test_idx"] = 0
    metadata["training_data"] = basenames[:int(0.7 * num_names)]
    metadata["val_data"] = basenames[int(0.7 * num_names): int(0.9 * num_names)]
    metadata["test_data"] = basenames[int(0.9 * num_names): ]
    pickle.dump(metadata, metadatafile)

def save_metadata(metadata, metadatapath):
    with open(metadatapath, 'wb') as metadatafile:
        pickle.dump(metadata, metadatafile)

def load_metadata(metadatapath):
    with open(metadatapath, 'rb') as metadatafile:
        metadata = pickle.load(metadatafile)
    return metadata

def load_batch(batch_size, metadatapath, input_data_dir, output_data_dir, part="training"):
    metadata = load_metadata(metadatapath)
    curr_idx = metadata[part+"_idx"]
    total_ims = len(metadata[part+"_data"])

    paths = np.array(metadata[part+"_data"])
    indices = range(curr_idx, curr_idx + batch_size)
    curr_paths = paths.take(indices, mode='wrap')

    metadata[part+"_idx"] = (curr_idx + batch_size) % total_ims
    save_metadata(metadata, metadatapath)

    def get_input_im(path):
        path = "{}/{}".format(input_data_dir, path)
        im = plt.imread(path)
        print("aah", im.shape)
        return im

    def get_output_im(path):
        path = "{}/{}".format(output_data_dir, path)
        return plt.imread(path)
    
    input_ims = np.array([get_input_im(p) for p in curr_paths])
    output_ims = np.array([get_output_im(p) for p in curr_paths])
    return input_ims, output_ims



if __name__ == "__main__":
    ### PARAMS
    input_data_dir = "processed/SUN2012_cleaned/input"
    output_data_dir = "processed/SUN2012_cleaned/output"
    metadatapath = "logs/SUN2012_metadata.txt"
    batch_size = 64
    ###
    if not os.path.exists(metadatapath):
        with open(metadatapath, 'wb') as metadatafile:
            print("Couldn't find metadata, loading {} metadata...".format(input_data_dir))
            split_data(input_data_dir, metadatafile)

    metadata = load_metadata(metadatapath)

    # Demo code to show dataset
    # im1s, im2s = load_batch(5, metadatapath, input_data_dir,output_data_dir, "test")

    # plt.figure()
    # #subplot(r,c) provide the no. of rows and columns
    # f, axarr = plt.subplots(5,2) 

    # # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    # for i in range(5):
    #     axarr[i,0].imshow(im1s[i])
    #     axarr[i,1].imshow(im2s[i])
    # f.suptitle("Testing data: left - synthetically blurred; right - clean")
    # plt.savefig("logs/demo/test_imags.png")

    
    batch_size = 10
    im1s, im2s = load_batch(10, metadatapath, input_data_dir, output_data_dir, "training")
    
    #print("AAHH: ", np.array(im1s).shape)
    
    model = Pix2PixModel()
    model.train_on_batch(im1s, im2s, batch_size)



