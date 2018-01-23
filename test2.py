'''
Created on Jan 15, 2018

@author: ARL
'''

from AR_NN_util.utils import kerasdatasets
import AR_NN_util.utils as ARU
import numpy as np
import os
import tfvar as T
import keras as k
import functools as ft
#ARU.layer_interp(nlayers, inpu, method)

layer_sizes=64
pool_size=2
window_size=3
batchs=1#0#10
epochs=15
channel_order=["first","last"][1]
datas=["cloth","numbah"][0]
#todo keras with tensorflow layer, mxnet gluon, chainer, maybe pure TF and cntk and mxnet symmbolic
CN,MXG,MXS,TF,CH,K=0,0,0,0,0,0#1,1,0,0,0,1
train,test=kerasdatasets(4 if datas=="cloth" else 3)#3)# for digits#((x_train, y_train), (x_test, y_test))
print(test[1][2:10])
print(test[1].shape)
outft=0
arg=[{"num_classes":10},int,float,np.unicode][outft]
tout=k.utils.to_categorical if outft == 0 else ARU.caster
tout=ft.partial(tout,**arg) if outft==0 else ft.partial(tout,typ=arg)

tftrain=(np.expand_dims(np.float32(train[0]), 1 if channel_order=="first" else -1),tout(train[1]))
tftest=(np.expand_dims(np.float32(test[0]), 1 if channel_order=="first" else -1),tout(test[1] ))

fract=0
if fract:
    fracto=500
    x_train=train[0][:(train[0].shape[0]//fracto)]
    x_test=test[0][:(test[0].shape[0]//fracto)]
    y_train=train[1][:(train[1].shape[0]//fracto)]
    y_test=test[1][:(test[1].shape[0]//fracto)]

#heads, may add randvr to add dropout
a1=['conv','pool']
a2=['conv','pool','conv','pool']
a3=['proto']
a4=['proto','proto']
a5=['conv','proto']
a6=['conv','proto','conv','proto']
a7=['proto','pool','proto','pool']
a8=['proto','conv','pool','conv']
#body
b1=['flat','dense']
b2=['flat','dense','dense']
b3=['flat','dense','dense','dense']

K,CH=1,0

print(train[0][0].shape)
if K:#func api or sequential
    import keras as k
    import keras.layers as kl
    outshp=10
    kerdict=a7+b2#a2+b2
    seq=0
    klks={'conv':kl.Convolution2D,'pool':[kl.AveragePooling2D,kl.MaxPool2D][1],'flat':kl.Flatten,'dense':kl.Dense,'drop':kl.Dropout,"proto":T.TFvarLayer}
    kwargs={'conv':((layer_sizes,(window_size,window_size)),{"activation":"relu","data_format":"channels_{}".format(channel_order)}),
            'proto':((layer_sizes,(window_size,window_size)),{"activation":"relu","format":"NHWC" if channel_order=="last" else "NCHW"}),
            "pool":((),{"pool_size":(pool_size,pool_size),"data_format":"channels_{}".format(channel_order)}),"dense":((layer_sizes,),{"activation":"relu"}),"flat":((),{}),"drop":((0.25,),{})}
    if seq:
        model=k.Sequential()
        for ix,I in enumerate(kerdict):#sequential model
            if ix==0:
                kwargs[I][1]['input_shape']=tftrain[0][0].shape
            if ix==len(kerdict):
                model.add(klks[I](outshp,activation="softmax"))
            else:
                model.add(klks[I](*kwargs[I][0],**kwargs[I][1]))
    else:
        kinput=kl.Input(shape=tftrain[0][0].shape,dtype=np.float32)
        KERDICT=dict()
        iv=len(kerdict)
        print(iv)
        for ix,U in enumerate(kerdict):#func api
            print(ix,ix-iv-1)
            if ix==0:
                KERDICT[ix-iv]= klks[U](*kwargs[U][0],**kwargs[U][1])(kinput)
            elif ix+1==len(kerdict):
                KERDICT[ix-iv]= klks[U](outshp,activation="softmax")(KERDICT[ix-iv-1])
            else:
                KERDICT[ix-iv]= klks[U](*kwargs[U][0],**kwargs[U][1])(KERDICT[ix-iv-1])
        model=k.Model(inputs=kinput,output=KERDICT[-1])
    model.compile(loss=k.losses.categorical_crossentropy,optimizer="Adam",metrics=['accuracy'])
    k.utils.print_summary(model)
    print(tftrain[0].shape)
    
    model.fit(x=tftrain[0],y=tftrain[1],batch_size=batchs,epochs=epochs,validation_data=(tftest[0],tftest[1]))
  
  
  
if CH:#https://github.com/chainer/chainer/blob/master/examples/mnist/train_mnist.py  
    channel_order="first"
    train=[]
    test=[]
    for I in range(tftrain[0].shape[0]):
        train.append((tftrain[0][I],tftrain[1][I]))
    
    for I in range(tftest[0].shape[0]):
        test.append((tftest[0][I],tftest[1][I]))

    import chainer as ch
    import chainer.links as chL
    import chainer.functions as chF
    import chainer.training.extensions as extensions
    import chainerc2d as c2
    plotch=0
    chnlks={'conv':chL.Convolution2D,'_pool':[chF.average_pooling_2d,chF.max_pooling_2d][1],'_flat':chF.flatten,'dense':chL.Linear,'_drop':chF.dropout,"proto":c2.Convar2D,"_relu":chF.relu}
    kwargs={'conv':((None,layer_sizes),{"ksize":(window_size,window_size)}),
            'proto':((layer_sizes,(window_size,window_size)),{"format":"NCHW" if channel_order=='first' else "NHWC"}),
            "_pool":(( pool_size,pool_size),{}),"dense":((None,layer_sizes,),{}),"_flat":((),{}),"_drop":((0.25,),{}),"_relu":((),{})}
    #relu=chF.relu()
    a1=['conv','pool']
    a2=['conv','_pool','conv','_pool']
    a3=['proto']
    a4=['proto','proto']
    a31=['proto','_relu']
    a41=['proto','_relu','proto','_relu']
    a5=['conv','proto','_relu']
    a6=['conv','proto','conv','proto']
    a61=['conv','proto','_relu','conv','proto','_relu']
    a7=['proto','pool','proto','pool']
    #body
    b1=['flat','dense','_relu']
    b2=['flat','dense','_relu','dense','_relu']
    b3=['flat','dense','_relu','dense','_relu','dense','_relu']

    class Tx(ch.Chain):
        def __init__(self):
            super(Tx,self).__init__()
            with self.init_scope():
                pass
            
    class custnet(ch.Chain):
        def __init__(self,chdict):
            super(custnet, self).__init__()
            for ix,chl in enumerate(chdict):
                if ix==0:
                    net=[(chl+ix,chnlks[chl](*kwargs[chl][0],**kwargs[chl][1]))]
                else:
                    net+=[(chl+ix,chnlks[chl](*kwargs[chl][0],**kwargs[chl][1]))]
            #net = [('conv1', chL.Convolution2D(1, 6, 5, 1))]
            #net += [('_sigm1', chF.Sigmoid())]
            #net += [('_mpool1', chF.MaxPooling2D(2, 2))]
            #net += [('conv2', chL.Convolution2D(6, 16, 5, 1))]
            #net += [('_sigm2', chF.Sigmoid())]
            #net += [('_mpool2', chF.MaxPooling2D(2, 2))]
            #net += [('conv3', chL.Convolution2D(16, 120, 4, 1))]
            #net += [('_sigm3', chF.Sigmoid())]
            #net += [('_mpool3', chF.MaxPooling2D(2, 2))]
            #net += [('fc4', chL.Linear(None, 84))]
            #net += [('_sigm4', chF.Sigmoid())]
            #net += [('fc5', chL.Linear(84, 10))]
            #net += [('_sigm5', chF.Sigmoid())]
            with self.init_scope():
                for n in net:
                    if not n[0].startswith('_'):
                        setattr(self, n[0], n[1])
            self.forward = net
    
        def __call__(self, x):
            for n, f in self.forward:
                if not n.startswith('_'):
                    x = getattr(self, n)(x)
                else:
                    x = f(x)
            if ch.config.train:
                return x
            return chF.softmax(x)
        
    LF=chF.softmax
    class MLP(ch.Chain):

        def __init__(self, n_units, n_out):
            super(MLP, self).__init__()
            with self.init_scope():
                # the size of the inputs to each layer will be inferred
                self.l1 = chL.Linear(None, n_units)  # n_in -> n_units
                self.l2 = chL.Linear(None, n_units)  # n_units -> n_units
                self.l3 = chL.Linear(None, n_out)  # n_units -> n_out
    
        def __call__(self, x):
            h1 = chF.relu(self.l1(x))
            h2 = chF.relu(self.l2(h1))
            return LF(self.l3(h2))
    class clp(ch.Chain):

        def __init__(self, n_units, n_out):
            super(clp, self).__init__()
            with self.init_scope():
                # the size of the inputs to each layer will be inferred
                self.c1 = chL.Convolution2D(None, n_units,ksize=(window_size,window_size))  # n_in -> n_units
                self.c2 = chL.Convolution2D(None, n_units,ksize=(window_size,window_size))  # n_in -> n_units
                #self.l1 = chL.Convolution2D(None, n_units)  # n_in -> n_units
                #self.l1 = chL.Convolution2D(None, n_units)  # n_in -> n_units
                self.l1 = chL.Linear(None, n_units)  # n_units -> n_units
                self.l2 = chL.Linear(None, n_units)  # n_units -> n_units
                self.l3 = chL.Linear(None, n_out)  # n_units -> n_out
    
        def __call__(self, x):
            h1 = chF.max_pooling_2d(self.c1(x), pool_size, )
            h2 = chF.max_pooling_2d(self.c2(h1), pool_size)
            h3 = chF.relu(self.l1(chF.relu(h2)))
            h4 = chF.relu(self.l2(h3))
            return LF(self.l3(h4))

    class clpp(ch.Chain):
        def __init__(self, n_units, n_out,two=False):
            super(clpp, self).__init__()
            with self.init_scope():
                self.two=two
                # the size of the inputs to each layer will be inferred
                self.c1 = chL.Convolution2D(None, n_units,ksize=(window_size,window_size))  # n_in -> n_units
                self.p1 = c2.Convar2D(None, n_units,(window_size,window_size))  # n_in -> n_units
                if two:
                    self.c2 = chL.Convolution2D(None, n_units,ksize=(window_size,window_size))  # n_in -> n_units
                    self.p2 = c2.Convar2D(None, n_units,(window_size,window_size))  # n_in -> n_units
                #self.l1 = chL.Convolution2D(None, n_units)  # n_in -> n_units
                #self.l1 = chL.Convolution2D(None, n_units)  # n_in -> n_units
                self.l1 = chL.Linear(None, n_units)  # n_units -> n_units
                self.l2 = chL.Linear(None, n_units)  # n_units -> n_out
                self.l3 = chL.Linear(None, n_out)  # n_units -> n_out
        def __call__(self, x):
            h1 = chF.max_pooling_2d(self.c1(x), pool_size)
            h2 = chF.relu(self.p1(h1))
            if self.two:
                h3 = chF.max_pooling_2d(self.c2(h2), pool_size)
                h4 = chF.relu(self.p2(h1))
            h5 = chF.relu(self.l1(chF.relu(h4 if self.two else h2)))
            h6 = chF.relu(self.l2(h5))
            return LF(self.l3(h6))
        
    class clv(ch.Chain):
        def __init__(self, n_units, n_out,pool=False):
            super(clv, self).__init__()
            with self.init_scope():
                self.pool=pool
                # the size of the inputs to each layer will be inferred
                self.p1 = c2.Convar2D(None, n_units,(window_size,window_size))  # n_in -> n_units
                self.p2 = c2.Convar2D(None, n_units,(window_size,window_size))  # n_in -> n_units
                self.l1 = chL.Linear(None, n_units)  # n_units -> n_units
                self.l2 = chL.Linear(None, n_units)  # n_units -> n_out
                self.l3 = chL.Linear(None, n_out)  # n_units -> n_out
        def __call__(self, x):
            if self.pool:
                h1 = chF.max_pooling_2d(self.p1(x), pool_size)
                h2 = chF.max_pooling_2d(self.p2(h1), pool_size)
            else:
                h1 = chF.relu(self.p1(x))
                h2 = chF.relu(self.p2(h1))
            h3 = chF.relu(self.l1(h2))
            h4 = chF.relu(self.l2(h3))
            return LF(self.l3(h4))
        
    chnlks={'conv':chL.Convolution2D,'pool':[chF.AveragePooling2D,chF.MaxPooling2D],'flat':chF.flatten,'dense':chL.Linear,'drop':chF.Dropout,"protp":c2.Convar2D}
    plotch=1
    #imperative, strucure data with explicit links
    #    print('GPU: {}'.format(args.gpu))
    #print('# unit: {}'.format(args.unit))
    #print('# Minibatch-size: {}'.format(args.batchsize))
    #print('# epoch: {}'.format(args.epoch))
    #print('')

    # Set up a neural network to train
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    model = chL.Classifier(clp(layer_sizes, 10))

    # Setup an optimizer
    optimizer = ch.optimizers.Adam()
    optimizer.setup(model)

    # Load the MNIST dataset
    #train, test = chainer.datasets.get_mnist()

    train_iter = ch.iterators.SerialIterator(train, batchs,shuffle=True,repeat=1, )
    test_iter = ch.iterators.SerialIterator(test, batchs,repeat=1, shuffle=True)
    # Set up a trainer
    updater = ch.training.StandardUpdater(train_iter, optimizer,)
    trainer = ch.training.Trainer(updater, (epochs, 'epoch'), out="chrez")

    # Evaluate the model with the test dataset for each epoch
    trainer.extend( extensions.Evaluator(test_iter, model, ))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    #trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    #frequency = epochs if freq == -1 else max(1, freq)
    trainer.extend(extensions.snapshot(), trigger=(epochs, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Save two plot images to the result dir
    if plotch and extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    #if args.resume:
    #    # Resume from a snapshot
    #    chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer2=0
    trainer.run()
    if trainer2:
        while train_iter.epoch < epochs:
        
            # ---------- One iteration of the training loop ----------
            train_batch = train_iter.next()
            image_train, target_train = concat_examples(train_batch, gpu_id)
        
            # Calculate the prediction of the network
            prediction_train = model(image_train)
        
            # Calculate the loss with softmax_cross_entropy
            loss = chF.softmax_cross_entropy(prediction_train, target_train)
        
            # Calculate the gradients in the network
            model.cleargrads()
            loss.backward()
        
            # Update all the trainable paremters
            optimizer.update()
            # --------------------- until here ---------------------
        
            # Check the validation accuracy of prediction after every epoch
            if train_iter.is_new_epoch:  # If this iteration is the final iteration of the current epoch
        
                # Display the training loss
                print('epoch:{:02d} train_loss:{:.04f} '.format(
                    train_iter.epoch, float(to_cpu(loss.data))), end='')
        
                test_losses = []
                test_accuracies = []
                while True:
                    test_batch = test_iter.next()
                    image_test, target_test = concat_examples(test_batch, gpu_id)
        
                    # Forward the test data
                    prediction_test = model(image_test)
        
                    # Calculate the loss
                    loss_test = chF.softmax_cross_entropy(prediction_test, target_test)
                    test_losses.append(to_cpu(loss_test.data))
        
                    # Calculate the accuracy
                    accuracy = chF.accuracy(prediction_test, target_test)
                    accuracy.to_cpu()
                    test_accuracies.append(accuracy.data)
        
                    if test_iter.is_new_epoch:
                        test_iter.epoch = 0
                        test_iter.current_position = 0
                        test_iter.is_new_epoch = False
                        test_iter._pushed_position = None
                        break
        
                print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
                    np.mean(test_losses), np.mean(test_accuracies)))
    
    