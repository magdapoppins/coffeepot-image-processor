import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import pandas as pd
from os import listdir, walk, remove
import csv
from sklearn import datasets, svm, metrics, utils


def convert_to_bw(path, name):
    # for example convert_to_bw("Images\\", "ONE.JPG")
    image_file = Image.open(path+name) # Open colour image
    image_file = image_file.convert("L") # Convert the image to b&w
    image_file.save(path+"BW_"+name)
    remove(path+name)
    return path+"BW_"+name

def convert_to_png(path, name):
    image_file = Image.open(path+name)
    image_file.save(path + name[:-4]+".png")
    remove(path+name)
    return name[:-4]+".png"

def process_image(image_file, count):
    """Process Image

    :image_file: <str> Name of file to be processed.
    :count: <str> The processed file will be saved by this count (as name) 
    :return: <list<float>> The image px values as a numpy array 

    Example:
    for i in len(images):
        process_image(images[i], i)
        
    """
    OUTPUT_PATH = "./Processed/"
    INPUT_PATH = "./Images/"
    WIDTH = 100
    HEIGHT = 100
    EXT = ".jpg"

    #image_path = INPUT_PATH + image_file
    
    # Step 0: Convert to png and to b&w
    png_name = convert_to_png(INPUT_PATH, image_file)
    image_path = convert_to_bw(INPUT_PATH, png_name)

    # Step 1: image_file = "pic.jpg" Open with Pillow
    im1 = Image.open(image_path)

    # If not width == height, resize (check params..)
    width_1, height_1 = im1.size
    if width_1 > height_1:
        diff = width_1-height_1
        im1.crop((diff/2, 0, diff/2, height_1))
    elif width_1 < height_1:
        diff = height_1-width_1
        im1.crop((0, diff/2, width_1, diff/2))
    
    # Step 2: Use antialias filter to apply resize
    im2 = im1.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

    # Save
    im2.save(OUTPUT_PATH+count+EXT)

    # Return numpy array
    img_np = np.array(list(im2.getdata(band=0)), float)
    img_np = img_np.reshape([10000, 1])
    return img_np

def process_all():
    """
    Process All
    Example:
    process_all()

    Requires that the same directory contains folders Images and Processed. Will give you 100x100px, grayscale, cropped variants of your pics.
        
    """

    images = []
    image_numpys = np.empty([10000 , 1])
    IMAGE_PATH = './Images'

    for (dirpath, dirnames, filenames) in walk(IMAGE_PATH):
        images.extend(filenames)

    for i in range(len(images)):
        if images[i].find("BW_") == -1:
            nump = process_image(images[i], str(i))
            nump = np.transpose(nump)
            #image_numpys.append(nump)
            if i == 0:
                image_numpys = nump
            else:
                image_numpys = np.concatenate((image_numpys, nump))

    np.savetxt("data2.csv", image_numpys, fmt="%.2f", delimiter=",")

    # df = pd.DataFrame(data=image_numpys, index=None)
    # df.to_csv("coffeecsv.csv", header=None, index=None)

    print("Successfully processed {} files.".format(len(images)))
    print("You get a numpy array with length {}.".format(len(image_numpys)))


def load_data(filepath):
    saveddata = np.loadtxt(filepath,delimiter=',')
    images=[]
    flat_data = []

    # need to still add target and descr
    target=[]
    descr = []


    fig = plt.figure()

    count = 0
    cols = 3
    n_images = len(saveddata)

    for i in saveddata:
        # add flat data to array
        flat_data.append(i)
        # reshape the image from 10000*1 to 100*100
        i = i.reshape([100,100])
        images.append(i)

        # plot it
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), count + 1)
        plt.axis('off')
        plt.imshow(i, cmap=plt.cm.gray, interpolation='none')
        count = count + 1

    # plot it
    fig.set_size_inches(np.array(fig.get_size_inches()) * (n_images-1))
    fig.tight_layout()
    # uncomment this to show image
    #plt.show()

    return utils.Bunch(data=flat_data, target=target, target_names=np.arange(10),images=images, DESCR=descr)




# uncomment this when u want to process images
#process_all()

#load data
dataset = load_data("./data2.csv")

print("hello there :)")