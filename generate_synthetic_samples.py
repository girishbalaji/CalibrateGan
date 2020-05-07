import argparse
import sys
import os
from imgaug.augmenters import blur
from imageio import imwrite as imsave
from matplotlib import pyplot as plt
import cv2
from multiprocessing import Pool

def resize(sample):
    nrow, ncol,_ = sample.shape
    mindim = min(nrow, ncol)
    resized = sample[(nrow - mindim) // 2 : (nrow + mindim) // 2,
                    (ncol - mindim) //2 : (ncol + mindim) // 2,:]
    resized = cv2.resize(resized, (256,256))
    return resized

def process_directory(args):
    curr_dir_name, raw_image_dir, model_input_dir, model_output_dir = args
    
    print("Processing: {}...".format(curr_dir_name))
    motion_blur_kernel = blur.MotionBlur((21,41))

    curr_model_input_dir = "{}/{}".format(model_input_dir, curr_dir_name)
    curr_model_output_dir = "{}/{}".format(model_output_dir, curr_dir_name)
    cur_raw_image_dir = "{}/{}".format(raw_image_dir, curr_dir_name)

    try:
        os.makedirs(curr_model_input_dir)
        os.makedirs(curr_model_output_dir)
    except OSError as exc:
        pass

    for rootName, dirNames, fileNames in os.walk(cur_raw_image_dir):
        dirNames.sort()
        fileNames.sort()

        for fileName in fileNames:
            full_path = os.path.join(rootName, fileName)

            # Gets {letter}_{name}_{og imname}.jpg
            new_name = "_".join(full_path.split("/")[3:])
            print(new_name)
            if (fileName.endswith(".jpg")):
                curr_im = plt.imread(full_path)
                if (len(curr_im.shape) != 3 or curr_im.shape[2] != 3):
                    continue
                resized = resize(curr_im)
                for i in range(3):
                    blurred = motion_blur_kernel.augment_image(resized)
                    model_input_im_path = "{}/{}_{}".format(curr_model_input_dir, i, new_name)
                    model_output_im_path = "{}/{}_{}".format(curr_model_output_dir, i, new_name)
                    imsave(model_input_im_path, blurred)
                    imsave(model_output_im_path, resized)

def main():
    SOURCE_DIR = "datasets"
    OUTPUT_DIR = "processed"

    parser = argparse.ArgumentParser()
    # parser.add_argument('-s', '--size', type=int, required=True)
    # parser.add_argument('-n', '--name', type=str, required=True)
    

    args = parser.parse_args()
    dataset_name = "SUN2012"
    RAW_IMAGE_DIR = "{}/{}/{}".format(SOURCE_DIR, dataset_name, "Images")

    model_input_dir = "{}/{}/{}".format(OUTPUT_DIR, "SUN2012", "input")
    model_output_dir = "{}/{}/{}".format(OUTPUT_DIR, "SUN2012", "output")

    print("Raw image dir: {}".format(RAW_IMAGE_DIR))
    
    all_dirs = os.listdir(RAW_IMAGE_DIR)
    gen_arg = lambda x: (x, RAW_IMAGE_DIR, model_input_dir, model_output_dir)
    args = [gen_arg(x) for x in all_dirs]
    with Pool(16) as p:
        p.map(process_directory, args)



        






































if __name__ == '__main__':
    main()