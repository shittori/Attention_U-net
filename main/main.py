import tensorflow as tf
import os
import datetime
from moduls.utils import plots ,mkdir
from moduls.make_dataset import load_X,load_Y,load_test,load_valid, step,normalize_y,normalize_x,normalize,denormalize_y,step_three,mean_step
import numpy as np
from model.Attention_U_net import attention_unet, dice_coef_crossentropy,bce_dice_loss
from pathlib import Path
def train():
    # trainingDataフォルダ配下にimageフォルダを置いている
    X_train,file_names,image_names = load_X()
    # trainingDataフォルダ配下にimage_segmentフォルダを置いている
    Y_train = load_Y()

    x_valid,y_valid,file_names_val,image_names_val=load_valid()
    '''Hyper Param'''
    BATCH_SIZE = 10
    NUM_EPOCH = 100
    lr = 0.0001
    loss = bce_dice_loss
    Optimizer =tf.keras.optimizers.Adam(lr=lr)
    '''saved_chackpoint'''
    mk = mkdir('train')
    # checkpoint_path = mk + "training_2/model-{epoch:02d}.h5"
    MODEL_DIR = mk + 'temp'
    if not os.path.exists(MODEL_DIR):  # ディレクトリが存在しない場合、作成する。
        os.makedirs(MODEL_DIR)

    # ログ保存用
    now = datetime.datetime.now()
    f = open(mk+ "Trainlog.txt", "w")
    f.write(str(now))
    f.write("\r\n")
    # ハイパーパラメータのログ保存
    f.write("Train Used on "+ image_names +"\r\n")
    f.write("Valid Used on " + image_names_val + "\r\n")
    f.write("Train on %s Samples\r\n" % str(len(file_names)))
    f.write("Valid on %s Samples\r\n" % str(len(file_names_val)))
    f.write("batchSize: %s\r\n" % str(BATCH_SIZE))
    f.write("epochNum: %s\r\n" % str(NUM_EPOCH))
    f.write("lr: %s\r\n" % str(lr))
    f.write("loss: %s \r\n" % str(loss))
    f.write("optimizer: %s\r\n" % str(Optimizer))
    f.close()
    w,h,c,_=X_train.shape
    model = attention_unet(img_rows=128,img_cols=128)
    model = model()
    model.compile(loss=loss,
                  optimizer=Optimizer)
    # エポック数は適宜調整する
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, "model_weight-{epoch:02d}.h5"),save_best_only=False,period=5,save_weights_only=True)
    # path = mkdir('train')
    history = model.fit(X_train, Y_train,validation_data=(x_valid,y_valid), callbacks=[checkpoint], batch_size=BATCH_SIZE, epochs=NUM_EPOCH,
                        verbose=1)
    model.save_weights(mk + 'unet_weights.hdf5')

    f_m=open(mk+ 'model.sumary.txt','w')
    model.summary(print_fn=lambda x:f_m.write(x + '\r\n'))
    f_m.close()

    plots(history, mk)
    return mk


def predict(path):
    import cv2
    X_test, file_names = load_test()
    # ログ保存用
    save = Path(path).parts[-1]
    path2 = os.path.dirname(path)
    path3 = os.path.dirname(path2)
    path4 = path3 + '/predict/'
    now = datetime.datetime.now()
    mk = mkdir(switch='predict')
    # ハイパーパラメータのログ保存
    f = open(mk + "%sTestlog.txt" % save, "w")
    f.write(str(now))
    f.write("\r\n")
    f.write(mk + "Test Used on " + file_names[0]+ "\r\n")
    f.write(mk + "Test on %s Samples\r\n" % str(len(file_names)))
    f.write(mk + "Model_Param Used on %s\r\n" % str(path))
    f.close()
    '''使わない'''
    model = attention_unet(128,128)
    model =model()
    '''model構築のための流し込み'''
    model(np.zeros((1, 128, 128, 1)))
    model.load_weights(path + '/unet_weights.hdf5')

    BATCH_SIZE =1
    Y_pred = model.predict(X_test, BATCH_SIZE)
    # path = 'D:/2021CVSLab/AIST/TS_SNData/5/'
    '''
    2particle
    1pore
    '''
    '''ToDo
    list型で返されている値を分解，ソートして返す
    '''
    '''U-net++,Output4'''
    '''U-net++deep3,Output3'''
    # for i, y in enumerate(Output[:, :, :, 2]):
    for i, y in enumerate(Y_pred[:,:,:,1]):
        # testDataフォルダ配下にimageフォルダを置いている
        th1,th2=Y_pred[i,:,:,0],Y_pred[i,:,:,2]
        th1,th2=denormalize_y(th1),denormalize_y(th2)
        image_y = denormalize_y(y)
        image=step(image_y)
        image=mean_step(image,th1,th2)
        # image=step_three(image_y)
        '''loadtrain'''
        # cv2.imwrite(path + 'pred/' + 'prediction' +str(i)+ '.png', image_y)
        cv2.imwrite(mk + 'pred/' + 'prediction' + str(file_names[i]) + '.bmp', image)

if __name__ == '__main__':
    '''使用しない'''
    gpu_devices = tf.config.experimental.list_physical_devices("GPU")
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)
    # TF_ENABLE_GPU_GARBAGE_COLLECTION=False
    # path = train()
    path = 'D:/Attention_U-net/report/20220622attention/2022_6_20_18/'
    predict(path)