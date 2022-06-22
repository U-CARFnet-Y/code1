import numpy as np
import os
import skimage.io as io
import skimage.transform as trans
from PIL import Image

def trainGenerator(batch_size, train_path, image_folder, mask_folder, image_num=30, target_size=(256, 256)):

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
            mask = np.reshape(image, mask.shape + (1,))
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

def testGenerator(test_path,num_image = 30,target_size = (256,256),as_gray=True):
    for i in range(num_image):
        img = io.imread(os.path.join(test_path, "%s.png" % i), as_gray=as_gray)
        img = trans.resize(img, target_size)
        img = np.reshape(img, img.shape+(1,))
        img = np.reshape(img, (1,) + img.shape)
        yield img

def saveResult(save_path,save_dict, npyfile):
    try:
        os.mkdir(os.path.join(save_path, save_dict))
    except:
        print("this dir has been exist")
    save_dir = os.path.join(save_path,save_dict)
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        # img[img > 0.5] = 1
        # img[img <= 0.5] = 0
        io.imsave(os.path.join(save_dir, "%d_predict.png" % i), img)



if __name__=='__main__':
    trainGenerator(2, "./data2/train", "image", "label" )
    testGenerator("./data1/test/test")
    saveResult("./data2/test/","testResult")