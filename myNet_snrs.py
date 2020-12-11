import utils
import os
import os.path as osp
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import LambdaCallback,EarlyStopping,ModelCheckpoint,TensorBoard
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
import csv

def mkfile_or_exist(newfile_pth):
    if not osp.exists(newfile_pth):
        f = open(newfile_pth,'w')
        f.close()
        print(newfile_pth + " created.")
    else:
        print(newfile_pth + " already existed.")
    return

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# GPU usage setup
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def Net_snrs(select_snr):
    # hyperparameters
    lr = 0.0001
    filter_num = 60
    kernel_size = 10
    lstm_units = 128
    drop_ratio = 0.2
    lstm_drop_ratio = 0.2
    dense_units = 128
    sample_length = 64
    max_epoch = 1000
    batch_size = 400
    patience = 200

    # load data
    filename = 'pkl_data/'+str(sample_length)+'.pkl'
    # select_snr = -20
    x_train_snr,y_train_snr,x_val_snr,y_val_snr,x_test_snr,y_test_snr = utils.radioml_IQ_data_snr(filename,select_snr)


    # callbacks
    early_stopping = EarlyStopping(monitor='val_loss',patience=patience)
    best_model_path = 'result/models/DNN/'+str(sample_length)+f'/best_{select_snr}.h5'
    checkpointer = ModelCheckpoint(best_model_path,verbose=0,save_best_only=True)
    TB_dir = 'result/TB'
    tensorboard = TensorBoard(TB_dir)
    # model = utils.DetectNet(lr,(2,sample_length),filter_num,lstm_units,kernel_size,drop_ratio,lstm_drop_ratio,dense_units)
    model = utils.DNN(lr,(2,sample_length),drop_ratio)
    history = model.fit(x_train_snr,y_train_snr,epochs=max_epoch,batch_size=batch_size,verbose=0,shuffle=True,validation_data=(x_val_snr, y_val_snr),
                        callbacks=[early_stopping,checkpointer,tensorboard])
    #test stage
    model = load_model(best_model_path)
    Pd, Pf, accuracy = utils.myperformance_evaluation(x_test_snr,y_test_snr,model,select_snr)
    return Pd, Pf, accuracy

def main():
    snrs = np.arange(-20,10,2)
    # snrs = [-6,-4,-2]
    save_path = 'result/xls/DNN/' + str(64) + '/stat.xls'
    f = open(save_path, 'w')
    f_csv = csv.writer(f)
    f_csv.writerow(['snr','Pf', 'Pd', 'accuracy'])
    # break
    for snr in snrs:
        if snr == 8:
            break
        Pd, Pf, accuracy = Net_snrs(select_snr=snr)
        # print(cm)
        # break
        # import csv
        # pd_list.append(pf)
        f_csv.writerow([snr, Pd, Pf, accuracy])


    return 0
if __name__ == '__main__':
    print('main')
    main()