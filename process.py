from PIL import Image
import numpy as np
import pandas as pd
from os import listdir, walk

def process_image(image_file, count):
    """Process Image

    :image_file: <str> Name of file to be processed.
    :count: <str> The processed file will be saved by this count (as name) 
    :return: <list<float>> The image px values as a numpy array 

    Example:
    for i in len(images):
        process_image(images[i], i)
        
    """
    OUTPUT_PATH = "Processed\\"
    INPUT_PATH = "Images\\"
    WIDTH = 100
    HEIGHT = 100
    EXT = ".jpg"
    
    # Step 1: image_file = "pic.jpg" Open with Pillow
    image_path = INPUT_PATH + image_file
    im1 = Image.open(image_path)

    # If not width == height, resize (check params..)
    width_1, height_1 = im1.size
    if width_1 > height_1:
        diff = width_1-height_1
        im1.crop((0, diff/2, width_1, diff/2))
    elif width_1 < height_1:
        diff = height_1-width_1
        im1.crop((0, 0, diff/2, diff/2))
    
    # Step 2: Use antialias filter to apply resize
    im2 = im1.resize((WIDTH, HEIGHT), Image.ANTIALIAS)

    # Step 3: Convert to b&w
    im3 = im2.convert('LA')

    # Save
    im2.save(OUTPUT_PATH+count+EXT)

    # Return numpy array
    img_np = np.array(list(im3.getdata(band=0)), float)
    img_np.shape = (im3.size[1], im3.size[0])
    print(img_np.shape)
    return img_np

def process_all():
    """Process All
    Example:
    process_all()

    Requires that the same directory contains folders Images and Processed. Will give you 100x100px, grayscale, cropped variants of your pics.
        
    """

    images = []
    image_numpys = []
    IMAGE_PATH = 'C:\\Samples\\ImageProcessor\\Images'

    for (dirpath, dirnames, filenames) in walk(IMAGE_PATH):
        images.extend(filenames)

    for i in range(len(images)):
        nump = process_image(images[i], str(i))
        image_numpys.append(nump)


    # file = open('coffeearray.txt', 'w')
    # file.write(str(image_numpys))

    np.savetxt("foo.csv", image_numpys[0], delimiter=",")

    # df = pd.DataFrame(data=image_numpys, index=None)
    # df.to_csv("coffeecsv.csv", header=None, index=None)

    print("Successfully processed {} files.".format(len(images)))
    print("You get a numpy array with length {}.".format(len(image_numpys)))   