#https://www.kaggle.com/devm2024/transfer-learning-with-vgg-16-cnn-aug-lb-0-1712

#import
import numpy as np
import pandas as pd
import cv2
from subprocess import check_output
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold,StratifiedShuffleSplit
from os.path import join as opj
from mpl_toolkits.mplot3d import Axes3D

from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Input,Flatten,Activation
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras import initializers
from keras.optimizers import Adam,rmsprop
from keras.layers.advanced_activations import LeakyReLU,PReLU
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping

from keras.datasets import cifar10
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from keras.applications.mobilenet import MobileNet
from keras.applications.vgg19 import VGG19
from keras.layers import Concatenate,Dense,LSTM,Input,concatenate
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

def get_more_images(imgs):
    
    more_images = []
    vert_flip_imgs = []
    hori_flip_imgs = []
    ddd_flip_imgs=[]
      
    for i in range(0,imgs.shape[0]):
        a=imgs[i,:,:,0]
        b=imgs[i,:,:,1]
        c=imgs[i,:,:,2]
        
        av=cv2.flip(a,1)
        ah=cv2.flip(a,0)
        ad=cv2.flip(a,-1)
        bv=cv2.flip(b,1)
        bh=cv2.flip(b,0)
        bd=cv2.flip(b,-1)
        cv=cv2.flip(c,1)
        ch=cv2.flip(c,0)
        cd=cv2.flip(c,-1)
        vert_flip_imgs.append(np.dstack((av, bv, cv)))
        hori_flip_imgs.append(np.dstack((ah, bh, ch)))
        ddd_flip_imgs.append(np.dstack((ad,bd,cd)))
      
    v = np.array(vert_flip_imgs)
    h = np.array(hori_flip_imgs)
    d=np.array(ddd_flip_imgs)
       
    more_images = np.concatenate((imgs,v,h,d))
    
    return more_images

inputpath='D:\\ice\\data\\'#数据
#print(check_output(['ls',inputpath]))#.decode('utf8'))

train=pd.read_json(inputpath+'train.json')
test=pd.read_json(inputpath+'test.json')
target_train=train['is_iceberg']#读取


test['inc_angle']=pd.to_numeric(test['inc_angle'],errors='coerce')#then invalid parsing will be set as NaN
train['inc_angle']=pd.to_numeric(train['inc_angle'],errors='coerce')#133 NAs
train['inc_angle']=train['inc_angle'].fillna(method='pad')#没懂
X_angle=train['inc_angle']
#test['inc_angle']=pd.to_numeric(test['inc_angle'],errors='coerce')
X_test_angle=test['inc_angle']
#X_angle=np.concatenate((X_angle,X_angle,X_angle,X_angle))
#target_train=np.concatenate((target_train,target_train,target_train,target_train))

X_band_1=np.array([np.array(band).astype(np.float32).reshape(75,75)for band in train['band_1']])
X_band_2=np.array([np.array(band).astype(np.float32).reshape(75,75)for band in train['band_2']])
X_band_3=(X_band_1+X_band_2)/2#三层，一层band1，一层band2，一层取平均
#X_band_3=np.array([np.full((75,75),angel).astype(np.float32)for angel in train['inc_angle']])
X_train=np.concatenate([X_band_1[:,:,:,np.newaxis],X_band_2[:,:,:,np.newaxis],X_band_3[:,:,:,np.newaxis]],axis=-1)
#X_train=get_more_images(X_less)
X_band_test_1=np.array([np.array(band).astype(np.float32).reshape(75,75)for band in test['band_1']])
X_band_test_2=np.array([np.array(band).astype(np.float32).reshape(75,75)for band in test['band_2']])
X_band_test_3=(X_band_test_1+X_band_test_2)/2
#X_band_test_3=np.array([np.full((75,75),angel).astype(np.float32)for angel in test['inc_angle']])
X_test=np.concatenate([X_band_test_1[:,:,:,np.newaxis],X_band_test_2[:,:,:,np.newaxis],X_band_test_3[:,:,:,np.newaxis]],axis=-1)
#转换成图像


#file_path='D:\\ice\\savedata\\'
batch_size=64

gen=ImageDataGenerator(horizontal_flip=True,vertical_flip=True,width_shift_range=0.,height_shift_range=0.,channel_shift_range=0,zoom_range=0.2,rotation_range=10)#对数据进行一些操作
'''keras.preprocessing.image.ImageDataGenerator(featurewise_center=False,#使输入数据集去中心化，均值为0
    samplewise_center=False,#使输入数据每个样本均值为0 
    featurewise_std_normalization=False,#使输入除以数据集的标准差以完成标准化
    samplewise_std_normalization=False,#使每个数据输入除以自身的标准差
    zca_whitening=False,#对输入数据施加ZCA白化
    zca_epsilon=1e-6,#ZCA使用的eposilon
    rotation_range=0.,#证书，数据提升时，图片随机转动的角度
    width_shift_range=0.,#数据提升时，图片水平偏移的幅度
    height_shift_range=0.,#数据提升时，图片竖直偏移的幅度
    shear_range=0.,#剪切强度，逆时针方向的剪切变换角度
    zoom_range=0.,#随机缩放的幅度
    channel_shift_range=0.,#随即通道偏移的幅度
    fill_mode='nearest',#当进行变换时超出边界的点将根据本参数给定的方法进行处理
    cval=0.,#指定向超出边界的点填充的值
    horizontal_flip=False,#随机水平翻转
    vertical_flip=False,#随机竖直翻转
    rescale=None,#重放缩因子
    preprocessing_function=None,#将应用于每个输入的函数，在做任何修改之前运行
    data_format=K.image_data_format())#
'''
def gen_flow_for_two_inputs(X1,X2,y):
    genX1=gen.flow(X1,y,batch_size=batch_size,seed=55)
    genX2=gen.flow(X1,X2,batch_size=batch_size,seed=55)
    while True:
        X1i=genX1.next()
        X2i=genX2.next()
        yield[X1i[0],X2i[1]],X1i[1]#（输入1，输入2，y#好像是生成26个，如果对所有训练集的话，

def get_callbacks(filepath,patience=2):
    es=EarlyStopping('val_loss',patience=10,mode='min')
    msave=ModelCheckpoint(filepath,save_best_only=True)#模型保存路径，只保存验证集上性能最好的的模型
    return [es,msave]
    
def getVggAngleModel():
    input_2=Input(shape=[1],name='angle')
    angle_layer=Dense(1,)(input_2)
    base_model=VGG16(weights='imagenet',include_top=False,input_shape=X_train.shape[1:],classes=1)
    #include_top是否保留顶层的3个全连接网络，
    #include_top为false时，input_shape应为长为3的元祖，指明输入图片的形状，宽高必须要大于48，cakasses图片分类的类别数目
    x=base_model.get_layer('block5_pool').output#pooling5的输出
    
    x=GlobalMaxPooling2D()(x)#平铺了
    merge_one=concatenate([x,angle_layer])#该层接收一个列表的同shape张量，并返回它们的按照给定轴相接构成的向量。
    merge_one=Dense(512,activation='relu',name='fc2')(merge_one)#全连接层
    merge_one=Dropout(0.3)(merge_one)
    merge_one=Dense(512,activation='relu',name='fc3')(merge_one)
    merge_one=Dropout(0.3)(merge_one)
    
    predictions=Dense(1,activation='sigmoid')(merge_one)
    model=Model(input=[base_model.input,input_2],output=predictions)
    
    sgd=SGD(lr=1e-3,decay=1e-6,momentum=0.9,nesterov=True)#随机梯度下降优化器lr学习速率，decay，每次更新之后学习速率的下降，momentum，参数更新动量？，nesterov是否应用这个动量
    model.compile(loss='binary_crossentropy',optimizer=sgd,metrics=['accuracy'])
    return model
    
    
def myAngleCV(X_train,X_angle,X_test):
    K=3
    folds=list(StratifiedKFold(n_splits=K,shuffle=True,random_state=16).split(X_train,target_train))
    y_test_pred_log=0
    y_train_pred_log=0
    y_valid_pred_log=0.0*target_train
    for j,(train_idx,test_idx)in enumerate(folds):
        print('\n==============FOLD=',j)
        X_train_cv=X_train[train_idx]#训练集，验证集
        Y_train_cv=target_train[train_idx]
        X_holdout=X_train[test_idx]
        Y_holdout=target_train[test_idx]
        
        X_angle_cv=X_angle[train_idx]
        X_angle_hold=X_angle[test_idx]#分成验证集和训练集
        
        
        X_train_cv=get_more_images(X_train_cv)
        Y_train_cv=np.concatenate((Y_train_cv,Y_train_cv,Y_train_cv,Y_train_cv))
        X_angle_cv=np.concatenate((X_angle_cv,X_angle_cv,X_angle_cv,X_angle_cv))
        
        file_path='%s_aug_model_weights.hdf5'%j
        callbacks=get_callbacks(filepath=file_path,patience=5)
        gen_flow=gen_flow_for_two_inputs(X_train_cv,X_angle_cv,Y_train_cv)#队训练集中分出来的训练集进行操作
        #K=3的时候，大约17*64个图像应该，好像只是随机处理了一下没有增加图片。
        galaxyModel=getVggAngleModel()
        galaxyModel.fit_generator(gen_flow,steps_per_epoch=24,epochs=200,shuffle=True,verbose=1,validation_data=([X_holdout,X_angle_hold],Y_holdout),callbacks=callbacks)
        #steps_per_epoch表示一个epoch分成多少个batch_size
        galaxyModel.load_weights(filepath=file_path)#从HDF5文件中加载权重到当前模型，默认情况下模型结构保持不变，
        score=galaxyModel.evaluate([X_train_cv,X_angle_cv],Y_train_cv,verbose=0)
        print('train loss:',score[0])
        print('train accuracy:',score[1])
        
        score=galaxyModel.evaluate([X_holdout,X_angle_hold],Y_holdout,verbose=0)
        print('Test loss:',score[0])
        print('test accuracy',score[1])
        
        
        pred_valid=galaxyModel.predict([X_holdout,X_angle_hold])
        y_valid_pred_log[test_idx]=pred_valid.reshape(pred_valid.shape[0])
        
        temp_test=galaxyModel.predict([X_test,X_test_angle])
        y_test_pred_log+=temp_test.reshape(temp_test.shape[0])
        
        temp_train=galaxyModel.predict([X_train,X_angle])
        y_train_pred_log+=temp_train.reshape(temp_train.shape[0])
        
    y_test_pred_log=y_test_pred_log/K
    y_train_pred_log=y_train_pred_log/K
    
    print('\n Train log loss Validation= ',log_loss(target_train,y_train_pred_log))
    print('Test log loss Validation= ',log_loss(target_train,y_valid_pred_log))
    
    return y_test_pred_log

    
submission=pd.DataFrame()
submission['id']=test['id'] 
submission['is_iceberg']=0 
for i in range(5):#上一次是5，交叉验证K是3，结果0.16左右，然后这个还没有尝试
    np.random.seed(i)   
    preds=myAngleCV(X_train,X_angle,X_test)
    submission['is_iceberg']+=preds
submission['is_iceberg']/=5
submission.to_csv('D:\\ice\\savedata\\vgg_16_fold_3_seed_16_cv2_newimags.csv',index=False)
