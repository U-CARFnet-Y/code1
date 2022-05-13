
import tensorflow as tf
import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from PIL import Image
from cv2 import *
"""
Description:
This code is the first network U-CARfnet data processing, including:
1. Read data from the directory
2. Data processing (normalization and size modification)
3. Make one-to-one correspondence between training pictures and labels
4. Read and process test data
The structure of the directory in which data is stored
Data1 is a picture of the training UNet network
The training data is stored in data/train/, where the training image is stored in image, and the corresponding training label is stored in label. Note that the name of the image must be corresponding to the meaning, and the name is numbered. Such as:
PNG in image corresponds to 1.png in label
The test/ generated image data is stored in data/test/, where the original image is stored in the data/test/test folder, and the processed data is stored in the testResult folder (it does not exist at the beginning of use, and the testResult folder is generated after the execution of the program).

"""

def mask_proccess(mask):
    for i in range(256):
        for j in range(256):
            if mask[i, j]>0.001:
                mask[i, j] = 0
            else:
                mask[i, j] = 1
    return mask

def trainGenerator(batch_size, train_path, image_folder,
                   mask_folder, image_num=1448, target_size=(256, 256)):
    # 对文件进行检索，和编号
    image_dir = os.path.join(train_path, image_folder)
    mask_dir = os.path.join(train_path, mask_folder)
    i = 0
    while True:
        image_batch = []
        mask_batch = []
        image_count = i
        for batch in range(batch_size):
            image = io.imread(os.path.join(image_dir, "%s.png" % image_count), as_gray=True)
            image = trans.resize(image, target_size)
            image = np.reshape(image, image.shape + (1,))
            image_list = image.tolist()
            image_batch.append(image_list)
            mask = io.imread((os.path.join(mask_dir, "%s.png" % image_count)), as_gray=True)
            mask = trans.resize(mask, target_size)
            mask = mask_proccess(mask)
            mask = np.reshape(mask, mask.shape + (1,))
            mask_list = mask.tolist()
            mask_batch.append(mask_list)
            image_count += 1
            if image_count >= image_num:
                image_count = 0

        image_batch = np.array(image_batch)
        mask_batch = np.array(mask_batch)
        i += 1
        if i >= image_num: i = 0
        yield (image_batch, mask_batch)

def testGenerator(test_path,num_image = 1448,target_size = (256,256),as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%s.png" % i), as_gray=as_gray)
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img

def testGenerator1(test_path, test_folder, label_folder, num_image = 1448, target_size = (256, 256), as_gray=True):
    test_image_dir = os.path.join(test_path, test_folder)
    test_mask_dir = os.path.join(test_path, label_folder)
    print("-" * 50)
    print("check_point")
    print("-" * 50)
    while True:
        for i in range(num_image):
            img = io.imread(os.path.join(test_image_dir, "%s.png" % i), as_gray= as_gray)
            mask = io.imread(os.path.join(test_mask_dir, "%s_predict.png" % i), as_gray=as_gray)

            img = trans.resize(img, target_size)
            img = np.reshape(img, img.shape + (1,))
            img = np.reshape(img, (1,) + img.shape)
            mask = mask / 255.
            mask = trans.resize(mask, target_size)
            mask = np.reshape(mask, mask.shape + (1,))
            mask = np.reshape(mask, (1,) + mask.shape)
            mask = mask_proccess(mask)
            yield (img, mask)

def saveResult(save_path, save_dict, npyfile, more_name=""):
    try:
        os.mkdir(os.path.join(save_path, save_dict))
    except:
        print("this dir has been exist")
    save_dir = os.path.join(save_path, save_dict)
    for i, item in enumerate(npyfile):
        img = item[:, :, 0]
        # img[img > 0.5] = 1
        # img[img <= 0.5] = 0
        file_name = "%d" + more_name + ".png"
        io.imsave(os.path.join(save_dir, (file_name % i)), img)

if __name__=='__main__':
    trainGenerator(2, "./data1/train", "image", "label" )
