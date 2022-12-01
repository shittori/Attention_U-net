# https://www.kaggle.com/code/xxc025/attention-u-net/notebook
#
#
from tensorflow import reduce_sum
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.layers import Flatten
def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (reduce_sum(y_true_f) + reduce_sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss = 1.0 - dsc(y_true, y_pred)
    return loss


def bce_dice_loss(y_true, y_pred):
    loss =  CategoricalCrossentropy()(y_true,y_pred)
    return loss

def dice_coef_loss(y_true,y_pred):
  y_true_f=K.flatten(y_true)
  y_pred_f=K.flatten(y_pred)
  intersection=K.sum(y_true_f*y_pred_f)
  return 1.-(2.*intersection)/(K.sum(y_true_f*y_true_f)+K.sum(y_pred_f*y_pred_f))


def dice_coef_crossentropy(y_true,y_pred):

    loss = CategoricalCrossentropy(y_true,y_pred) + dice_coef_loss(y_true,y_pred)
    return loss


class attention_unet():
    def __init__(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.img_shape = (self.img_rows, self.img_cols, 1)
        self.df = 32
        self.uf = 32

    def __call__(self):
        def conv2d(layer_input, filters):
            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)

            d = layers.BatchNormalization()(d)
            d = layers.LeakyReLU()(d)
            # d = layers.Activation('relu')(d)

            d = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)

            d = layers.BatchNormalization()(d)
            d = layers.LeakyReLU()(d)
            # d = layers.Activation('relu')(d)

            return d

        def deconv2d(layer_input, filters):
            u = layers.UpSampling2D((2, 2))(layer_input)
            u = layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(u)
            u = layers.BatchNormalization()(u)
            u = layers.LeakyReLU()(u)
            # u = layers.Activation('relu')(u)

            return u

        def attention_block(F_g, F_l, F_int):
            '''xはエンコーダ部，
                gがデコーダー部'''
            g = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)

            g = layers.BatchNormalization()(g)

            x = layers.Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)

            x = layers.BatchNormalization()(x)
            #       print(g.shape)
            #       print(x.shape)
            psi = layers.Add()([g, x])
            psi = layers.Activation('relu')(psi)

            psi = layers.Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
            psi = layers.BatchNormalization()(psi)
            psi = layers.Activation('sigmoid')(psi)
            '''***
            Multiply():入力のリスト同士の積を計算する．入力はすべて同じレイヤーを持ったテンソルの一つで，1つのテンソルを返す．Shapeは等しい
            '''
            return layers.Multiply()([F_l, psi])

        inputs = layers.Input(shape=self.img_shape)

        conv1 = conv2d(inputs, self.df)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)

        conv2 = conv2d(pool1, self.df * 2)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)

        conv3 = conv2d(pool2, self.df * 4)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)

        conv4 = conv2d(pool3, self.df * 8)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)

        conv5 = conv2d(pool4, self.df * 16)

        up6 = deconv2d(conv5, self.uf * 8)
        attention1 = attention_block(up6, conv4, self.uf * 8)
        up6 = layers.Concatenate()([up6, attention1])
        conv6 = conv2d(up6, self.uf * 8)

        up7 = deconv2d(conv6, self.uf * 4)
        attention2 = attention_block(up7, conv3, self.uf * 4)
        up7 = layers.Concatenate()([up7, attention2])
        conv7 = conv2d(up7, self.uf * 4)

        up8 = deconv2d(conv7, self.uf * 2)
        attention3 = attention_block(up8, conv2, self.uf * 2)
        up8 = layers.Concatenate()([up8, attention3])
        conv8 = conv2d(up8, self.uf * 2)

        up9 = deconv2d(conv8, self.uf)
        attention4 = attention_block(up9, conv1, self.uf)
        up9 = layers.Concatenate()([up9, attention4])
        conv9 = conv2d(up9, self.uf)

        outputs = layers.Conv2D(3, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(conv9)

        model = Model(inputs=inputs, outputs=outputs)
        # model = Model(inpus=inputs, outputs=[attention1,attention2,attention3,attention4,outputs])

        return model
if __name__=='__main__':
    a = np.ones((1,128,128,1))
    model = attention_unet(img_cols=128,img_rows=128)
    model = model()
    model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(lr=0.01))
    model(a)