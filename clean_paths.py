import glob
import os
import shutil


def main():
    DATASET = "SUN2012"


    input_ims = glob.glob("processed/SUN2012/input/*/*.jpg")
    output_ims = glob.glob("processed/SUN2012/output/*/*.jpg")

    sorted(input_ims)
    sorted(output_ims)

    print("Found: {} input ims; {} output ims".format(len(input_ims), len(output_ims)))

    try:
        os.makedirs("processed/SUN2012_cleaned")
    except OSError:
        pass
    try:
        os.makedirs("processed/SUN2012_cleaned/input")
    except OSError:
        pass
    try:
        os.makedirs("processed/SUN2012_cleaned/output")
    except OSError:
        pass

    for i in range(len(input_ims)):
        if i % 20 == 0:
            print("{} / {} images".format(i, len(input_ims)))

        oginputpath = input_ims[i]
        ogoutputpath = output_ims[i]

        targetinputpath = "processed/SUN2012_cleaned/input/{}.jpg".format(i)
        targetoutputpath = "processed/SUN2012_cleaned/output/{}.jpg".format(i)

        shutil.move(oginputpath, targetinputpath)
        shutil.move(ogoutputpath, targetoutputpath)






if __name__ == '__main__':
    main()