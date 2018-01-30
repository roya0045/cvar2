from utils import kerasdatasets
import utils as ARU
import numpy as np
import copy
import datetime as dt
import time as ti
#import os
import tfvar as T
import tensorflow.contrib.layers as tcl
import keras as k
import keras.layers as kl
import keras.optimizers as ko
import functools as ft
#ARU.layer_interp(nlayers, inpu, method)

K=1#keras trainer trigger

#heads, may add dropout
a1=['conv','conv','pool']
a2=['conv','conv']
a3=['proto']
a4=['proto','proto']
a5=['conv','proto']
a6=['conv','proto','conv','proto']
a65=['proto','conv','conv']#,'pool']
a7=['proto','pool','proto','pool']
a8=['proto','conv','pool','conv']
#body
b1=['flat','dense']
b2=['flat','dense','dense']
b3=['flat','dense','dense','dense']

#################VALUES THAT CAN BE CHANGED################################

#################STRUCTURE

kerdict=a1+b2
#layers to use, use 1 from a and one from b

optim=ko.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0001)#.optimizers.Nadam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
#optimizer to use, see keras or tf docs for alternatives

################TRAINING

batchs=23
#0#10 #size of the batch

epochs=5
# number of training iterations

################LAYER SETTINGS

layer_sizes=[20,50,100,30,]
#number of "cells" per layers

window_size=[6,4]
#window size to use for conv and proto layer

pool_size=2
# size of the pooling window

act="relu"
#activation function, see keras or tf docs for more alternatives

################DATA SETTINGS

data=[3,4,1,2][0]# dataset to use, change only the slicer
#0 = mnist, 1= fashion mnist, 2= cifar 10, 3 = cifar 100 with coarse labels( can be changed)

fracto=0 
#fraction of the data to use, set to 0 for all data

################MISC
proto=0#changes the parent class (0:_conv,1: base layer)
seq=0 #use the sequential model builder instead of the API
outft=0#used to coerce output data type to match NN output in case
eval=0#perform evaluation step at the end of training
################# END OF VALUES THAT CAN BE CHANGED################################


if data==2:
    outshp=100
else:
    outshp=10
channel_order=["first","last"][1]
train,test=kerasdatasets(data)#3)# for digits#((x_train, y_train), (x_test, y_test))
arg=[{"num_classes":outshp},int,float,np.unicode][outft]

if fracto:
    train=(train[0][:(train[0].shape[0]//fracto)],train[1][:(train[1].shape[0]//fracto)])
    test=(test[0][:(test[0].shape[0]//fracto)],test[1][:(test[1].shape[0]//fracto)])
tout=k.utils.to_categorical if outft == 0 else ARU.caster
tout=ft.partial(tout,**arg) if outft==0 else ft.partial(tout,typ=arg)
tftrain=(np.expand_dims(np.float32(train[0]), 1 if channel_order=="first" else -1),tout(train[1]))
tftest=(np.expand_dims(np.float32(test[0]), 1 if channel_order=="first" else -1),tout(test[1] ))

def kertrain(*args):
    kerdict=list()
    for arg in args:
        kerdict.append(arg)
    klks={'conv':kl.Convolution2D,'pool':[kl.AveragePooling2D,kl.MaxPool2D][1],'flat':kl.Flatten,'dense':kl.Dense,'drop':kl.Dropout,"proto":[T.TFvarLayer,T.KvarLayer][proto]}
    kwargs={'conv':((layer_sizes,),{"activation":act,"data_format":"channels_{}".format(channel_order),"kernel_initializer":tcl.xavier_initializer_conv2d()}),
            'proto':((layer_sizes,),{"activation":act,"format":"NHWC" if channel_order=="last" else "NCHW"}),
            "pool":((),{"pool_size":(pool_size,pool_size),"data_format":"channels_{}".format(channel_order),}),"dense":((layer_sizes,),{"activation":act}),"flat":((),{}),"drop":(0.25,{})}
    kcw=window_size[-1]
    lss=layer_sizes[-1]
    if seq:
        model=k.Sequential()
        for ix,I in enumerate(kerdict):#sequential model
            if (len(kwargs[I][0])>0) and not(I=="drop" or I=="flat" or I=="pool"):
                try:
                    sss=(layer_sizes.pop(0),)
                except:
                    sss=(lss,)
            else:
                sss=kwargs[I][0]
            if (I=="conv")or(I=='proto'):
                try:
                    wand=window_size.pop(0)
                except:
                    want=kcw
                twand=(wand,wand)
            if ix==0:
                kwargs[I][1]['input_shape']=tftrain[0][0].shape
            if ix==len(kerdict):
                model.add(klks[I](outshp,activation="softmax"))
            else:
                if (I=="conv")or(I=='proto'):
                    model.add(klks[I](*sss,twand,**kwargs[I][1]))
                else:
                    model.add(klks[I](*sss,**kwargs[I][1]))
    else:
        kinput=kl.Input(shape=tftrain[0][0].shape,dtype=np.float32)
        KERDICT=dict()
        iv=len(kerdict)
        print(iv)
        for ix,U in enumerate(kerdict):#func api
            if (len(kwargs[U][0])>0) and not(U=="drop" or U=="flat"or U=="pool"):
                try:
                    sss=(layer_sizes.pop(0),)
                except:
                    sss=(lss,)
            else:
                sss=kwargs[U][0]
            if (U=="conv")or(U=='proto'):
                try:
                    wand=window_size.pop(0)
                except:
                    want=kcw
                twand=(wand,wand)
            print(ix,ix-iv-1,U,sss if sss else kwargs[U][0])
            if ix==0:
                if (U=="conv")or(U=='proto'):
                    KERDICT[ix-iv]= klks[U](*sss,twand,**kwargs[U][1])(kinput)
                else:
                    KERDICT[ix-iv]= klks[U](*sss,**kwargs[U][1])(kinput)
            elif ix+1==len(kerdict):
                KERDICT[ix-iv]= klks[U](outshp,activation="softmax")(KERDICT[ix-iv-1])
            else:
                if (U=="conv")or(U=='proto'):
                    KERDICT[ix-iv]= klks[U](*sss,twand,**kwargs[U][1])(KERDICT[ix-iv-1])
                else:
                    KERDICT[ix-iv]= klks[U](*sss,**kwargs[U][1])(KERDICT[ix-iv-1])
        model=k.Model(inputs=kinput,output=KERDICT[-1])
    model.compile(loss=k.losses.categorical_crossentropy,
                  optimizer=optim
                  ,metrics=['accuracy'])
    k.utils.print_summary(model)
    sess=tf.Session()
    with sess.as_default():
        sess.run(tf.global_variables_initializer())
        hardlr=(copy.deepcopy(sess.run(optim.lr))*2)
    def step_decay(epoch):
        drop = 0.5
        epochs_drop = 10.0
        lrate = hardlr * drop** floor((1+epoch)/epochs_drop)
        return lrate
    callbako=[k.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0),
              k.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=0, verbose=0, mode='min'),
              k.callbacks.LearningRateScheduler(step_decay)]
    fitres=model.fit(x=tftrain[0],y=tftrain[1],batch_size=batchs,epochs=epochs,
              validation_split=0.05,validation_data=(tftest[0],tftest[1]),
              shuffle=False,verbose=2,callbacks=callbako)
    if eval:
        evres=model.evaluate(tftest[0], y=tftest[1], batch_size=batchs, verbose=0,callbacks=callbako)
        return(fitres,evres)
    return(fitres)

if K:
    returned=dict()
    todo=[kerdict,]
    #hsto=kertrain(*kerdict)
    #print(hsto.params)#good
    #print(hsto.history)#1 value per epoch per metric
    print(kerdict)
    for parms in todo:
        if eval:
            histob,histob2=kertrain(*parms)
            returned[histob.params]=histob.history
            returned[histob2.params]=histob2.history
        else:
            histob=kertrain(*parms)
            returned[histob.params]=histob.history
    print(returned)
    tdt=dt.date.today()
    with open("protolog_{}{}-{}.txt".format(tdt.year,tdt.timetuple()[-2],ti.ctime().split(" ")[-2].replace(":","-"))) as fl:
        fl.write(returned)
