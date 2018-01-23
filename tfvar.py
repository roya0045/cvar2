'''
Created on Dec 19, 2017

@author: ARL
'''
import numpy as np
import tensorflow as tf
import keras as K
from keras.engine.topology import Layer as LYR
import keras.backend as KB
import AR_NN_util.utils as ARU
def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    "kernel, step, stride,pad,cover, dilation"
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


#imgtf=tf.convert_to_tensor()
#tfa=tf.transpose(imgtf, [0,2,3,1])#NWHC hit

class MyLayer(LYR):#K.layers._Conv

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return KB.dot(x, self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)
class TFvarLayer(K.layers.convolutional._Conv): #K.layers.convolutional._Conv):#layers.Layer):
    """
    :param num_c: number of cell in this layer
    :filter size of the filter/kernel (int or container)
    :stride size of the stride/step (int or container)
    :bigB use a bias for the kernel(False) or for the whole output(True)
    :padding type of padding to use, todo
    """
    def __init__(self, num_c, filtr,stride=1, #num_routing=3,
                 sqrt=0,V=False,format='NHWC',sizz=0,
                 w_init='glorot_uniform',KCD=False,
                 b_init='zeros',bigB=False,pad=None,activation="relu",
                 **kwargs):
        self.noutputs=num_c
        self.window=filtr
        self.stride=stride
        self.sqrt=sqrt
        self.KCD=KCD
        assert (format=="NCHW")or(format=="NHWC")
        self.format=format
        self.KINIT=w_init
        self.BINIT=b_init
        self.sizz=sizz
        self.pad=pad
        self.shape=None
        self.arrs=None
        self.ashp=None
        self.convshape=None
        super(TFvarLayer, self).__init__(2,num_c,filtr,**kwargs)
        

        
        #print('modified iimage tnsor',tfa.eval(),'modified iimage tnsor')
    def tfwindow(self,arr,nc_to_nh=False,pad="VALID",stride=1):
        """pad can be "SAME",will add 0s to get the same output shape as input
        nc_to_nh: change from "NHWC" to "NCHW"
        stride = stride for the 2 internal axis
        """
        assert pad=="VALID" or pad=="SAME"
        print(arr.shape,"inwind")
        print("convshape",self.convshape)
        if self.format=="NCHW":
            temp=tf.extract_image_patches(arr, ksizes=(1,1,*self.window), strides=(1,1,stride,stride),rates=(1,1,1,1),padding=pad) 
        elif self.format=="NHWC":
            temp=tf.extract_image_patches(arr, ksizes=(1,*self.window,1), strides=(1,stride,stride,1),rates=(1,1,1,1),padding=pad)
        print(temp.shape,"outwind") 
        if nc_to_nh and self.format=='NCHW':
            return(tf.transpose(tf.reshape(temp,(-1,*self.convshape,1,self.ch,*self.window)),perm=[0,1,2,3,6,4,5]))
        else:
            if self.format=="NCHW":
                return(tf.reshape(temp,(-1,*self.convshape,1,self.ch,*self.window)))
            elif self.format=="NHWC":
                return(tf.reshape(temp,(-1,*self.convshape,self.ch,*self.window,1)))

    def build(self, input_shape):
        print(input_shape)
        #self.arrs=input_shape
        if self.format=='NHWC':
            self.W = self.add_weight(shape=(self.noutputs,self.window[0],self.window[1],input_shape[-1]),#[self.num_c,self.chnl,self.filter[0],self.filter[1]],
                                 initializer=self.KINIT,name='W',trainable=True)
            self.ashp=(self.noutputs,self.window[0],self.window[1],input_shape[-1])
        elif self.format=='NCHW':
            self.W = self.add_weight(shape=(self.noutputs,input_shape[1],self.window[0],self.window[1]),#[self.num_c,self.chnl,self.filter[0],self.filter[1]],
                                 initializer=self.KINIT,name='W',trainable=True)
            self.ashp=(self.noutputs,input_shape[1],self.window[0],self.window[1])
        if self.KCD==1:
            self.B=self.add_weight(shape=(self.noutputs,1),
                                       initializer=self.BINIT,name='bias',trainable=True)
        elif self.KCD==2:
            self.B=self.add_weight(shape=(self.noutputs,self.ch),
                                       initializer=self.BINIT,name='bias',trainable=True)
        else:
            self.B=self.add_weight(shape=(self.noutputs,),
                                       initializer=self.BINIT,name='bias',trainable=True)#shape=(self.chnl,self.dms)

        twv=self.W.get_shape().as_list()
        if self.format=='NHWC':
            #W=tf.transpose(W, [0,2,3,1])
            self.sb=(1,1,self.W.shape[-1])
            self.WV=twv[1:3]
            self.ch=twv[-1]
            self.xi=(-3,-2)
            self.x2=(-1,-3,-2)
        elif self.format=='NCHW':
            
            self.sb=(self.W.shape[1],1,1)
            self.WV=twv[-2:]
            self.ch=twv[1]
            self.xi=(-2,-1)
            self.x2=(-2,-1,-3)
        self.built = True

    def call(self, array, training=None):#you can crunch by doing all channels
        if self.shape is None:
            self.shape=array.shape.as_list()
            print(self.shape,"shape")
            if self.format=="NCHW":
                self.convshape=ARU.convshape(self.shape[-2:], self.window)
            elif self.format=="NHWC":
                self.convshape=ARU.convshape(self.shape[1:-1], self.window)
        reshaped=self.tfwindow(array)
        if (self.arrs is None):
            self.arrs=reshaped.shape.as_list()
            print('arss',self.arrs)
        mul=(reshaped*self.W)
        size=tf.reduce_sum(self.W,axis=self.xi,keep_dims=True)#shape=(outputs, channel)
        mean=tf.reduce_sum(mul,axis=self.xi,keep_dims=True)/tf.constant(self.WV[0]*self.WV[1],shape=self.sb,dtype=tf.float32)
        i=(tf.square(mul-mean))/size
        if self.KCD:
            out=tf.reduce_sum(i,axis=self.xi)#i+B is useless
        else:
            out=tf.reduce_sum(i,axis=self.x2)#i+B is useless
        if self.sqrt:
            out=tf.sqrt(out)
        if not(self.B is None):
            try:
                out=out+self.B 
            except Exception as e:
                B=tf.reshape(self.B,(*self.B.shape,*[1 for _ in range(len(self.ashp)-len(self.B.shape)-1)]))
                out=out+B
        #print(out.shape,self.format,(self.arrs[0],self.arrs[1],self.arrs[2],self.ashp[0]))
        print(out.shape,self.format,(self.arrs[1],self.arrs[2],self.ashp[0]))
        if self.format=="NCHW":
            if self.KCD:
                return(tf.transpose(tf.reshape(out,(self.arrs[0],self.arrs[1],self.arrs[2],self.ashp[0]*self.arrs[-3])),(0,3,1,2)))
            else:
                assert out.shape[1:]==(self.arrs[1],self.arrs[2],self.ashp[0])
                print(tf.transpose(out, (0,3,1,2)).shape,"outshapenchw")
                return(tf.transpose(out, (0,3,1,2)))#tf.reshape(out,(self.arrs[0],self.ashp[0],self.arrs[1],self.arrs[2]))
        else:
            if self.KCD:
                return(tf.reshape(out,(self.arrs[0],self.arrs[1],self.arrs[2],self.ashp[0]*self.arrs[-3])))
            else:
                assert out.shape[1:]==(self.arrs[1],self.arrs[2],self.ashp[0])
                return(out)


    def compute_output_shape(self, input_shape):
        if self.format=="NCHW":
            #self.hout=get_conv_outsize(input_shape[-2], self.window[-2], s=self.stride, p=self.pad, )
            #self.wout=get_conv_outsize(input_shape[-1], self.window[-1], s=self.stride, p=self.pad, )
            if self.KCD:
                return((input_shape[0],self.noutputs*input_shape[1],*self.convshape))
            else:
                return((input_shape[0],self.noutputs,*self.convshape))
        elif self.format=="NHWC":
            #self.hout=get_conv_outsize(input_shape[-3], self.window[-2], s=self.stride, p=self.pad, )
            #self.wout=get_conv_outsize(input_shape[-2], self.window[-1], s=self.stride, p=self.pad, )
            if self.KCD:
                return((input_shape[0],*self.convshape,self.noutputs*input_shape[-1]))
            else:
                return((input_shape[0],*self.convshape,self.noutputs))

        #return tuple([None, self.num_c,self.chnl, self.dim_vector,self.num_c])
        
class KvarLayer(K.engine.topology.Layer): #K.layers.convolutional._Conv):#layers.Layer):
    """
    :param num_c: number of cell in this layer
    :filter size of the filter/kernel (int or container)
    :stride size of the stride/step (int or container)
    :bigB use a bias for the kernel(False) or for the whole output(True)
    :padding type of padding to use, todo
    """
    def __init__(self, num_c, filtr,stride=1, #num_routing=3,
                 sqrt=0,V=False,format='NHWC',sizz=0,
                 w_init='glorot_uniform',KCD=False,
                 b_init='zeros',bigB=False,pad=None,activation="relu",
                 **kwargs):
        self.noutputs=num_c
        self.window=filtr
        self.stride=stride
        self.sqrt=sqrt
        self.KCD=KCD
        self.format=format
        self.KINIT=w_init
        self.BINIT=b_init
        self.sizz=sizz
        self.pad=pad
        super(TFvarLayer, self).__init__(**kwargs)
        

        
        #print('modified iimage tnsor',tfa.eval(),'modified iimage tnsor')
    def tfwindow(self,arr,nc_to_nh=False,pad="VALID",stride=1,format="NCHW"):
        """pad can be "SAME",will add 0s to get the same output shape as input
        nc_to_nh: change from "NHWC" to "NCHW"
        stride = stride for the 2 internal axis
        """
        assert pad=="VALID" or pad=="SAME"
        if format=="NCHW":
            temp=tf.extract_image_patches(arr, ksizes=(1,1,*self.window), strides=(1,1,stride,stride),rates=(1,1,1,1),padding=pad) 
        elif format=="NHWC":
            temp=tf.extract_image_patches(arr, ksizes=(1,*self.window,1), strides=(1,stride,stride,1),rates=(1,1,1,1),padding=pad) 
        if nc_to_nh:
            return(tf.transpose(tf.reshape(temp,(-1,*temp.shape[1:3].as_list(),1,-1,*self.window)),perm=[0,1,2,3,6,4,5]))
        else:
            return(tf.reshape(temp,(-1,*temp.shape[1:3].as_list(),1,-1,*self.window)))

    def build(self, input_shape):
        self.arrs=input_shape
        if self.format=='NHWC':
            self.W = self.add_weight(shape=(self.noutputs,self.window[0],self.window[1],input_shape[-1]),#[self.num_c,self.chnl,self.filter[0],self.filter[1]],
                                 initializer=self.KINIT,name='W',trainable=True)
        elif self.format=='NCHW':
            self.W = self.add_weight(shape=(self.noutputs,input_shape[1],self.window[0],self.window[1]),#[self.num_c,self.chnl,self.filter[0],self.filter[1]],
                                 initializer=self.KINIT,name='W',trainable=True)

        self.B=self.add_weight(shape=(self.noutputs,1),
                                       initializer=self.BINIT,name='bias',trainable=True)#shape=(self.chnl,self.dms)


        if self.format=='NHWC':
            #W=tf.transpose(W, [0,2,3,1])
            self.sb=(1,1,self.W.shape[-1])
            self.WV=self.W.get_shape().as_list()[1:3]
            self.xi=(-3,-2)
            self.x2=(-1,-3,-2)
        elif self.format=='NCHW':
            self.sb=(self.W.shape[1],1,1)
            self.WV=self.W.get_shape().as_list()[-2:]
            self.xi=(-2,-1)
            self.x2=(-2,-1,-3)
        self.built = True

    def call(self, array, training=None):#you can crunch by doing all channels
        reshaped=self.tfwindow(array)
        arrs=reshaped.shape
        ashp=self.W.shape
        mul=(reshaped*self.W)
        size=tf.reduce_sum(self.W,axis=self.xi,keep_dims=True)#shape=(outputs, channel)
        mean=tf.reduce_sum(mul,axis=self.xi,keep_dims=True)/tf.constant(self.WV[0]*self.WV[1],shape=self.sb,dtype=tf.float32)
        i=(tf.square(mul-mean))/size
        if self.KCD:
            out=tf.reduce_sum(i,axis=self.xi)#i+B is useless
        else:
            out=tf.reduce_sum(i,axis=self.x2)#i+B is useless
        print(out.shape)
        if self.sqrt:
            out=tf.sqrt(out)
        if not(self.B is None):
            try:
                out=out+self.B 
            except:
                B=tf.reshape(self.B,(*self.B.shape,*[1 for _ in range(len(ashp)-len(self.B.shape)-1)]))
                out=out+B
        if self.KCD:
            return(tf.transpose(tf.reshape(out,(arrs[0],arrs[1],arrs[2],ashp[0]*arrs[-3])),(0,3,1,2)))
        else:
            assert out.shape==(arrs[0],arrs[1],arrs[2],ashp[0])
            return(out)


    def compute_output_shape(self, input_shape):
        if self.format=="NCHW":
            self.hout=get_conv_outsize(input_shape[-2], self.window[-2], s=self.stride, p=self.pad, )
            self.wout=get_conv_outsize(input_shape[-1], self.window[-1], s=self.stride, p=self.pad, )
            if self.KCD:
                return([input_shape[0],self.noutputs*input_shape[1],self.hout,self.wout])
            else:
                return([input_shape[0],self.noutputs,self.hout,self.wout])
        elif self.format=="NHWC":
            self.hout=get_conv_outsize(input_shape[-3], self.window[-2], s=self.stride, p=self.pad, )
            self.wout=get_conv_outsize(input_shape[-2], self.window[-1], s=self.stride, p=self.pad, )
            if self.KCD:
                return([input_shape[0],self.hout,self.wout,self.noutputs*input_shape[-1]])
            else:
                return([input_shape[0],self.hout,self.wout,self.noutputs])

        #return tuple([None, self.num_c,self.chnl, self.dim_vector,self.num_c])