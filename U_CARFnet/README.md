## Implementation of the use of code
---

### directory.
1. [Environment]
2. [How to train]
3. [How to predict]

### Environment
tensorflow-gpu==2.4.0
### How to train.
#### a、Introduction to the data.py file.
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

#### b、Introduction to the attention.py file.
1、This file is built for the attention module (no need to run).

#### c、Introduction to the U_CARFnet_train.py file.
1、Run U_CARFnet_train.py to start training.
2、The training generates the U_CARFnet.hdf5 file.
#### d、Introduction to the U_CARFnet_trainer.py file. 
1、Run U_CARFnet_train.py to call U_CARFnet.hdf5 to forecast and generate forecast results.

### How to predict
1、Run U_CARFnet_train.py and put the forecast results in. /data2/test/U_CARFnet.

