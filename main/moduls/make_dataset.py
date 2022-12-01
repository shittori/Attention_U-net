import numpy as np
import random

def load_X():
    import glob
    import cv2
    import os
    import random
    import numpy as np
    train = list()
    path = 'D:/2021CVSLab/AIST/1283k-2-1/img_glass_128_32_10'
    img_list = glob.glob(path + '/*')
    file_names = os.listdir(path)
    # img_list = glob.glob('D:/2021CVSLab/AIST/dataAugumenntaion/Data_Augmentation/x64/Release/5/img_glass_128_16_10/*')
    tes=cv2.imread(img_list[0],cv2.IMREAD_GRAYSCALE)
    h,w=tes.shape
    random.seed(0)
    random.shuffle(img_list)
    img_list = img_list[:10000]
    for img in img_list:
        im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        train.append(im)
    train = np.array(train)
    '''defolt'''
    train = train.reshape(-1, h, w, 1).astype('float32') / 255.

    print('***TrainShape***',train.shape)
    # print(type(train))
    return train, file_names, path
def load_Y():
    import glob
    import tensorflow.keras.preprocessing.image as keras_Image
    import cv2
    import itertools
    train = list()
    img_list = glob.glob('D:/2021CVSLab/AIST/1283k-2-1/img_mask_128_32_10/*')
    random.seed(0)
    random.shuffle(img_list)
    img_list = img_list[:10000]
    mask=list()
    mask_append=mask.append
    tes = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
    h, w = tes.shape
    # train_append=train.append
    for img in img_list:
        im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        # im = cv2.imread(img, -1)
        # cv2.imshow('ss',im)
        # cv2.Waitkey()
        # print('sss')
        x = np.zeros([h, w, 3])
        all_num = itertools.product(range(128), repeat=2)
        for i, j in all_num:
                if im[i, j] == 0:
                    x[i, j, 0] = 255.0

                elif im[i, j] == 128:
                    x[i, j, 1] = 255.0

                elif im[i, j] == 255:
                    x[i, j, 2] = 255.0
        mask_append(x)

        # train_append(im)
    train =np.array(mask)
    '''適宜サイズ変える'''
    train = train.reshape(-1, h, w, 3).astype('float32') / 255.


    print(train.shape)
    return train
def load_3y():
    def load_Y():
        import glob
        import tensorflow.keras.preprocessing.image as keras_Image
        import cv2
        import itertools
        import random
        import numpy as np

        img_list = glob.glob('data/processed/*')
        tes = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
        h, w, _ = tes.shape
        random.seed(0)
        random.shuffle(img_list)
        img_list = img_list[:100]
        mask = list()
        mask_append = mask.append
        for img in img_list:
            im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            x = np.zeros([h, w, 3])
            all_num = itertools.product(range(128), repeat=2)
            for i, j in all_num:
                if im[i, j] == 0:
                    x[i, j, 0] = 255.0

                elif im[i, j] == 128:
                    x[i, j, 1] = 255.0

                elif im[i, j] == 255:
                    x[i, j, 2] = 255.0
            mask_append(x)

            # train_append(im)
        train = np.array(mask)
        '''適宜サイズ変える'''
        train = train.reshape(-1, w, h, 3).astype('float32') / 255.

        print(train.shape)
        return train

def load_valid():
    import glob
    import cv2
    import numpy as np
    import random
    import os
    train = list()
    mask = list()
    path = 'D:/2021CVSLab/AIST/1283k-3-1/img_glass_128_32_10'

    img_list = glob.glob(path+'/*')
    file_names = os.listdir(path)
    mask_list = glob.glob(path+'/*')
    tes = cv2.imread(img_list[0],-1)
    h, w, _ = tes.shape
    random.seed(0)
    random.shuffle(img_list)
    random.seed(0)
    random.shuffle(mask_list)
    img_list = img_list[:1000]
    mask_list = mask_list[:1000]
    for img in img_list:
        im = cv2.imread(img,cv2.IMREAD_GRAYSCALE)

        train.append(im)
    for img in mask_list:
        im = cv2.imread(img, -1)
        # x = np.zeros([h, w, 3])
        # for i in range(h):
        #     for j in range(w):
        #         if im[i, j] == 0:
        #             x[i, j, 0] = 255
        #
        #         elif im[i, j] == 128:
        #             x[i, j, 1] = 255
        #
        #         elif im[i, j] == 255:
        #             x[i, j, 2] = 255
        mask.append(im)
        # mask.append(im)

    train = np.array(train)
    mask = np.array(mask)
    # train = train[:1000]
    # mask = mask[:1000]
    train = train.reshape(-1, h, h, 1).astype('float32') / 255.
    mask = mask.reshape(-1, h, w, 3).astype('float32') / 255.
    '''defolt'''
    # train = train.reshape(-1, 256, 256, 1).astype('float32') / 255.
    # train = train[...,tf.newaxis].astype('float32')/255.

    print(train.shape)
    print(type(train))
    print(mask.shape)
    print(type(mask))
    return train, mask, file_names, path


def load_test():
    import glob
    import tensorflow.keras.preprocessing.image as keras_Image
    import os
    import cv2
    import numpy as np
    path = os.path.dirname(__file__)
    path2 = os.path.dirname(path)
    path2 = os.path.dirname(path2)
    path3 = path2 + '/data/test/3k-3-1_sidecut'
    # path3 = path2 + '/data/test/3k_2_1_128_testorigin'
    file_names = os.listdir(path3)
    train = list()
    img_list = glob.glob(path3+'/*')
    tes = cv2.imread(img_list[0], cv2.IMREAD_GRAYSCALE)
    h, w = tes.shape
    for img in img_list:
        im = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
        # cv2.imshow('sample',im)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        train.append(im)
    train = np.array(train)
    train = train.reshape(-1, h, w, 1).astype('float32') / 255.
    print(train.shape)
    return train, file_names
# 値を-1から1に正規化する関数
def normalize_x(image):
    image = image / 127.5 - 1
    return image


# 値を0から1に正規化する関数
def normalize_y(image):
    image = image / 255.
    return image


# 値を0から255に戻す関数
def denormalize_y(image):
    image = image * 255.
    return image

#理想的な正規化
def normalize(x,axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    return (x - x_min) / (x_max - x_min),x_min,x_max


# 閾値を設ける
def step(image_y):

    for i in range(128):
        for j in range(128):
            if image_y[i][j] >= 127:
                image_y[i][j]=255.0
            else:
                image_y[i][j]=0.0

    return image_y
def step_three(image_y):
    Buffer=0
    for i in range(128):
        for j in range(128):
            for k in range(3):
                Buffer=image_y[i][j][k]

            if Buffer:
                for l in range(3):
                    image_y[i][j][l]=0
            elif 85< Buffer/3 < 171:
                for l in range(3):
                    image_y[i][j][l]=128
            elif 171<= Buffer/3:
                for l in range(3):
                    image_y[i][j][l]=255
            Buffer=0
    return image_y
def mean_step(image_y,th1,th2):
    for i in range(128):
        for j in range(128):
            if image_y[i][j] >th1[i][j] or image_y[i][j]>th2[i][j]:
                image_y[i][j]=255.0
            else:
                image_y[i][j]=0.0

    return image_y
if __name__=='main':
    load_X()
    load_3y()
    load_test()
    load_valid()
    normalize()
    normalize_x()
    normalize_y()
    step()
    mean_step()