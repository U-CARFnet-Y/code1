from U_CARFnet_train import *
from data import *


img_num = 0
for root, dirnames, files in os.walk('./data1/train/image'):
    for file in files:
        img_num += 1
test_num = 0
for root, dirnames, files in os.walk('./data1/test/test'):
    for file in files:
        test_num += 1
'''
model = U_CARFnetrain(batch=2, img_num=img_num,
                    epochs=50, load_model='U_CARFnet.hdf5')
'''

model = U_CARFnet(pretrained_weights='U_CARFnet.hdf5')
imageGene = testGenerator(test_path="./data1/train/image", num_image=img_num)
results = model.predict_generator(imageGene, img_num, verbose=1)
saveResult("./data2/train", "U_CARFnet", results)

testGene = testGenerator(test_path="./data1/test/test",num_image=test_num)
results = model.predict_generator(testGene, test_num, verbose=1)
saveResult("./data2/test", "U_CARFnet", results)

if __name__=='__main__':
    print(img_num)
    print(test_num)
