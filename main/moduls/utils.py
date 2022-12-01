import matplotlib.pyplot as plt
import pandas as pd
import os
import pathlib
def load_loss():
    df =pd.read_csv('D:/U-net++/result/train/2022_5_23_12/explain.csv')
    loss = df['loss']
    loss_1 = df['output_loss']
    # loss_2 = df['output_2_loss']
    # loss_3 = df['output_3_loss']
    # loss_4 = df['output_4_loss']
    val_loss=df['val_loss']
    epochs=df.iloc[:,0]
    fig2 = plt.figure()
    path = 'D:/U-net++/result/train/2022_5_23_12/'
    plt.plot(epochs, loss, 'm', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    axes = plt.gca()
    axes.set_facecolor('lightgray')
    plt.grid(color='white')
    plt.title('Training and Validation loss')
    plt.legend()
    # plt.show()
    fig2.savefig(path + "loss.png")
    fig3 = plt.figure()

    plt.plot(epochs, loss_1, 'navy', label='loss_1')
    # plt.plot(epochs, loss_2, 'tomato', label='loss_2')
    # plt.plot(epochs, loss_3, 'mediumspringgreen', label='loss_3')
    # plt.plot(epochs, loss_4, 'peru', label='loss_4')
    axes = plt.gca()
    axes.set_facecolor('lightgray')
    plt.grid(color='white')
    plt.title('Training loss')
    plt.legend()
    plt.show()
    fig3.savefig(path + "EACHloss.png")
def plots(history, path):
    df = pd.DataFrame(history.history)
    df.to_csv(path + "explain.csv")
    print(pd.DataFrame(history.history))

    loss = history.history['loss']
    loss_1=history.history['output_loss']
    # loss_2=history.history['output_2_loss']
    # loss_3=history.history['output_3_loss']
    # loss_4=history.history['output_4_loss']
    val_loss = history.history['val_loss']


    epochs = range(len(loss))

    # 2) Loss Plt
    fig2 = plt.figure()

    plt.plot(epochs, loss, 'm', label='training loss')
    plt.plot(epochs, val_loss, 'b', label='validation loss')
    axes=plt.gca()
    axes.set_facecolor('lightgray')
    plt.grid(color='white')
    plt.title('Training and Validation loss')
    plt.legend()
    # plt.show()
    fig2.savefig(path + "loss.png")
    fig3 = plt.figure()

    plt.plot(epochs, loss_1, 'navy', label='loss_1')
    # plt.plot(epochs, loss_2, 'tomato', label='loss_2')
    # plt.plot(epochs, loss_3, 'mediumspringgreen', label='loss_3')
    # plt.plot(epochs, loss_4, 'peru', label='loss_4')
    axes = plt.gca()
    axes.set_facecolor('lightgray')
    plt.grid(color='white')
    plt.title('Training loss')
    plt.legend()
    plt.show()
    fig3.savefig(path + "EACHloss.png")

def mkdir(switch):
    import datetime
    if switch == 'train':
        import glob

        now = datetime.datetime.now()
        # path = os.getcwd()
        path = os.path.dirname(__file__)
        path2 = os.path.dirname(path)
        path3 = os.path.dirname(path2)
        path4= path3 + '/result/train'
        is_file = os.path.exists(path4)
        print(is_file)
        hoge = '/' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(
            now.hour)
        is_hoge = os.path.exists(path4 + hoge)

        if is_file:
            if not is_hoge:
                os.mkdir(path4 + hoge)
        else:
            os.mkdir(os.path.dirname(path4))
            os.mkdir(os.path.dirname(path4) + hoge)

        mk = path4 + hoge + '/'
        return mk
    elif switch == 'predict':
        path = os.path.dirname(__file__)
        path2 = os.path.dirname(path)
        path3 = os.path.dirname(path2)
        path4 = path3 + '/result/predict'
        now = datetime.datetime.now()
        hoge = '/' + str(now.year) + '_' + str(now.month) + '_' + str(now.day) + '_' + str(
            now.hour)
        is_hoge = os.path.exists(path4 + hoge)
        if is_hoge==False:
            os.makedirs(path4+hoge)
            os.mkdir(path4 + hoge +'/pred')
            mk = path4 + hoge + '/'
        else:
            mk = path4 + hoge + '/'
        return mk

if __name__== '__main__':
    # mkdir()
    load_loss()
    # print('s')
    # plots()