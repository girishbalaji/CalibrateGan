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
        return im / 255.

    def get_output_im(path):
        path = "{}/{}".format(output_data_dir, path)
        return plt.imread(path) / 255.
    
    input_ims = np.array([get_input_im(p) for p in curr_paths])
    output_ims = np.array([get_output_im(p) for p in curr_paths])
    return input_ims, output_ims


def save_performance_ims(model, trialname, im1s, im2s, part="training", batch_size=4):
    date = datetime.datetime.now()
    out_ims = model.predict(im1s)
    plt.figure()
    #subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(batch_size,3) 

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    for i in range(batch_size):
        axarr[i,0].imshow(im1s[i])
        axarr[i,1].imshow(im2s[i])
        axarr[i,2].imshow(out_ims[i])
    f.suptitle("left: synthetically blurred; middle: original; right: generated")
    plt.savefig("logs/{}/{}/{}_{}_predict.png".format(trialname, "ims", part, date))

def save_loss(loss, trialname, part="training"):
    date = datetime.datetime.now()
    f = open("logs/{}/{}_losses.txt".format(trialname, part), "a")
    f.write("{},{}\n".format(date, str(loss)))
    f.close()


if __name__ == "__main__":
    ### PARAMS
    input_data_dir = "processed/SUN2012_cleaned/input"
    output_data_dir = "processed/SUN2012_cleaned/output"
    metadatapath = "logs/SUN2012_metadata.txt"
    
    TRAIN = True
    LOAD = False

    trialname = "trial4"
    batch_size = 64
    eval_batch_size = 40

    genpath = "logs/GenModel_2020-05-05 07:47:15.209846.h5"
    discpath = "logs/DiscModel_2020-05-05 07:47:15.209846.h5"
    ###
    if not os.path.exists(metadatapath):
        with open(metadatapath, 'wb') as metadatafile:
            print("Couldn't find metadata, loading {} metadata...".format(input_data_dir))
            split_data(input_data_dir, metadatafile)

    metadata = load_metadata(metadatapath)
    
    try:
        os.makedirs("logs/{}".format(trialname))
    except OSError as e:
        pass
    try:
        os.makedirs("logs/{}/{}".format(trialname, "ims"))
    except OSError as e:
        pass
    try:
        os.makedirs("logs/{}/{}".format(trialname, "models"))
    except OSError as e:
        pass


    model = Pix2PixModel()
    if LOAD:
        model.load_model(genpath, discpath)
    
    if TRAIN:
        for i in range(400):
            print("Training iter: {}".format(i))
            im1s, im2s = load_batch(batch_size, metadatapath, input_data_dir, output_data_dir, "training")
            gan_loss, disc_loss = model.train_on_batch(im1s, im2s, batch_size)
            save_loss((gan_loss, disc_loss), trialname, part="training")
            print("Gan loss: {}; Disc loss: {}".format(gan_loss, disc_loss))

            if i % 20 == 0:
                model.save("logs/{}/{}".format(trialname, "models"))
                im1s, im2s = load_batch(eval_batch_size, metadatapath, input_data_dir, output_data_dir, "val")
                eval_loss = model.eval_on_batch(im1s, im2s, eval_batch_size)
                save_loss(eval_loss, trialname, "val")
                save_performance_ims(model, trialname, im1s, im2s, "val")
                
    else:
        save_val_example(model, trialname)
        


    

    


