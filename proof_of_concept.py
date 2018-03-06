

method=['opencv','pil','mpl','big_mpl'][-1]
import PIL 
if method=='opencv':
    import cv2
elif method=='mpl' or method=='big_mpl':
    import matplotlib.pyplot as plt
    import matplotlib.image as mim
import numpy as np
import utils as ARU
import tensorflow as tf
from edger import edger
import time

from utils import kerasdatasets
import scipy as sp
import imgaug as iag


class KvarLayer(): #K.layers.convolutional._Conv):#layers.Layer):
    """
    :param num_c: number of cell in this layer
    :filter size of the filter/kernel (int or container)
    :bigB use a bias for the kernel(False) or for the whole output(True)
    :padding type of padding to use, todo
    """
    def __init__(self,  weights, 
                 sqrt=0,format='NHWC',
                 KCD=False,channal=False,t2=1,cast=None,
                 **kwargs):
        self.window=weights[0].shape
        self.channal=channal
        self.W=np.array(weights)
        if format=='NHWC':
            if t2:
                self.W=tf.expand_dims(self.W,len(self.W.shape))
            else:
                self.W=np.transpose(self.W, (1,2,0))
                if self.channal:
                    self.W=tf.expand_dims(self.W, 1)
        else:
            if self.channal:
                self.W=tf.expand_dims(self.W, 1)
        self.sqrt=sqrt
        self.KCD=KCD
        assert (format=="NCHW")or(format=="NHWC")
        self.format=format
        self.cast=cast
        self.zerr=tf.constant(0,dtype=cast)
        self.one=tf.constant(1,dtype=cast)
        if self.format=='NHWC':
            self.sb=(1,1,self.W.shape[-1])
            self.xi=(-3,-2)
            self.x2=(-1,-3,-2)
        elif self.format=='NCHW':
            self.sb=(self.W.shape[1],1,1)
            self.xi=(-2,-1)
            self.x2=(-2,-1,-3)
            
    def tfwindow(self,arr,nc_to_nh=False,pad="VALID",stride=1):
        """pad can be "SAME",will add 0s to get the same output shape as input
        nc_to_nh: change from "NHWC" to "NCHW"
        stride = stride for the 2 internal axis
        """
        assert pad=="VALID" or pad=="SAME"
        if self.format=="NCHW":
            temp=tf.extract_image_patches(arr, ksizes=(1,1,*self.window), strides=(1,1,stride,stride),rates=(1,1,1,1),padding=pad) 
        elif self.format=="NHWC":
            temp=tf.extract_image_patches(arr, ksizes=(1,*self.window,1), strides=(1,stride,stride,1),rates=(1,1,1,1),padding=pad)
        if nc_to_nh and self.format=='NCHW':
            return(tf.transpose(tf.reshape(temp,(-1,*self.convshape,1,self.ch,*self.window)),perm=[0,1,2,3,6,4,5]))
        else:
            if self.format=="NCHW":
                #print((-1,*self.convshape,1,self.ch,*self.window))
                return(tf.reshape(temp,(-1,*self.convshape,1,self.ch,*self.window)))
            elif self.format=="NHWC":
                #print((-1,*self.convshape,self.ch,*self.window,1))
                return(tf.reshape(temp,(-1,*self.convshape,self.ch,*self.window,1)))

    def call(self, array,outfrmt='NCHW'):#you can crunch by doing all channels
        self.shape=array.shape
        if self.format=="NCHW":
            self.convshape=ARU.convshape(self.shape[-2:], self.window)
            self.ashp=self.window[-1]
            self.ch=self.shape[1]
        elif self.format=="NHWC":
            self.convshape=ARU.convshape(self.shape[1:-1], self.window)
            self.ch=self.shape[-1]
            if self.channal:
                self.ashp=self.ch
            else:
                self.ashp=self.W.shape[-1]
        reshaped=self.tfwindow(array)
        if self.cast:
            self.W=tf.cast(self.W,self.cast)
            reshaped=tf.cast(reshaped,self.cast)
        self.arrs=reshaped.shape
        mul=(reshaped*self.W)
        size=tf.reduce_sum(self.W,axis=self.xi,keepdims=True)#shape=(outputs, channel)
        szm=tf.constant(self.window[0]*self.window[1],shape=self.sb,dtype=self.cast)
        mean=tf.reduce_sum(mul,axis=self.xi,keepdims=True)/szm
        #print(self.W.dtype,size.dtype,self.one.dtype,self.zerr.dtype)
        size=tf.cond(tf.equal(tf.reshape(size,([1])),self.zerr)[0], lambda: tf.add(size,self.one),lambda: size)
        i=(tf.square(mul-mean))/size
        if self.KCD:
            out=tf.reduce_sum(i,axis=self.xi)
        else:
            out=tf.reduce_sum(i,axis=self.x2)
        if self.sqrt:
            out=tf.sqrt(out)
        if not(self.cast is None):
            out=tf.cast(out, self.cast)
        #print(out.shape,self.format,(self.arrs[0],self.arrs[1],self.arrs[2],self.ashp[0]))
        #print('mul',mul.shape,'size',size.shape,'mean',mean.shape,'i',i.shape,'out',out.shape)
        #print(out.shape,self.format,(self.arrs[1],self.arrs[2],self.ashp))
        if self.format=="NCHW":
            if self.KCD:
                return(tf.transpose(tf.reshape(out,(self.arrs[0],self.arrs[1],self.arrs[2],self.ashp*self.arrs[-3])),(0,3,1,2)).eval())
            else:
                assert out.shape[1:]==(self.arrs[1],self.arrs[2],self.ashp)
                #print(tf.transpose(out, (0,3,1,2)).shape,"outshapenchw")
                return(tf.transpose(out, (0,3,1,2)).eval())#tf.reshape(out,(self.arrs[0],self.ashp[0],self.arrs[1],self.arrs[2]))
        else:
            if self.KCD:
                return(tf.reshape(out,(self.arrs[0],self.arrs[1],self.arrs[2],self.ch)).eval())
            else:
                if outfrmt=='NCHW':
                    out=tf.transpose(out, (3,0,1,2))
                    assert out.shape[1:]==(self.ashp,self.arrs[1],self.arrs[2])
                elif outfrmt=='NHWC':
                    out=tf.transpose(out, (3,1,2,0))
                    assert out.shape[1:]==(self.arrs[1],self.arrs[2],self.ashp)
                return(out.eval())
    

def jpg2arr(image_path,imfrmt='CHW'):
  """
  Loads JPEG image into 3D Numpy array of shape 
  (width, height, channels)
  """
  with PIL.Image.open(image_path) as image:         
    im_arr = np.fromstring(image.tobytes(), dtype=np.uint8) 
    im_arr = im_arr.reshape((image.size[1], image.size[0], 3))  
    if imfrmt=='CHW':
        im_arr = np.transpose(im_arr, (2,0,1))                               
  return(im_arr)

FORMAT=['NCHW','NHWC'][1]
holder=[]
bp=".\\POC\\"
kerdataa=1
if kerdataa:
    for K in (4,3):
        tr,ts=kerasdatasets(K)
        for I in range(15):
            holder.append(tr[0][I])
for imgpth in ["guit.jpg","meme.jpg",
               "3pan.jpg","croc.jpg","spell.jpg",
               "cell.jpg",
               "tiger.jpg",
               "butterfly.jpg",
               ]:
    holder.append(bp+imgpth)

weights=edger([3,3],2,num_edges=None,batch_edges=3,maxpoint=5,mode=1,operator=1,seed=10102)
sess=tf.Session()
onerw=True
sample=5
norm=0
test=[3,5,8,6,7,1,10,12]
class INFLIST(list):
    def pop_inf(self,x):
        if self.__len__()==1:
            return(self.__getitem__(0))
        else:
            return(self.pop(x))

datcast=[None,np.uint8,np.float16,np.float32,np.uint16,np.float64][-1]
recast=[None,np.uint8,np.float16,np.float32,np.uint16][1]
with sess.as_default():
    for I in holder:
        if isinstance(I, str):
            if not(method=='opencv'):
                I=jpg2arr(I,imfrmt=FORMAT[1:])
            elif method=='opencv':
                og=cv2.imread(I,1)
            Itype='RGB'
            if not(method=='opencv'):
                og=PIL.Image.fromarray(I, Itype)
            if FORMAT=='NCHW':
                og=np.transpose(og, (2,0,1))
        else:
            Itype='L'
            if FORMAT=='NCHW':
                I=np.reshape(I,(1,*I.shape))
                if method=='pil':
                    og=PIL.Image.fromarray(I[0], Itype)
            elif FORMAT=='NHWC':
                I=np.reshape(I,(*I.shape,1))
                if method=='pil':
                    og=PIL.Image.fromarray(I[:,:,0], Itype)
        if method=='pil':
            og.show()
        I2=None
        weigtz=[]
        if onerw:
            acm={}
            for num in range(sample+1):
                if num==0:
                    rw= [np.array([[0, -1, 0],[-1, 4, -1],[0, -1, 0]],dtype=datcast),]
                elif num==1:
                    rw = [np.array([[0, 0, 0],[0, -4, 1],[0, 2, 1]],dtype=datcast),]
                else:
                    rw=[weights[np.random.randint(0,high=len(weights))],]
                weigtz.append(rw[0])
                print('weights',rw,'weights')
                lay=KvarLayer(rw,format=FORMAT,KCD=True,sqrt=1,cast=datcast)
                outp=lay.call(np.reshape(I,(1,*I.shape,)),outfrmt='NHWC')[0]
                conv1=np.transpose(np.array([sp.signal.correlate2d(I[:,:,ix],rw[0], mode='valid') for ix in range(I.shape[-1])]),(1,2,0))
                convp=iag.augmenters.Sequential([iag.augmenters.Convolve(matrix=rw[0])]).augment_image(I)
                rwex=np.expand_dims(np.expand_dims(rw[0],-1),-1).astype(datcast)
                if not(Itype=='L'):
                    rwex=np.transpose(np.array([np.transpose(np.array([rw[0] for _ in range(I.shape[-1])]),(1,2,0)) for _ in range(I.shape[-1])]),(1,2,3,0))
                else:
                    rwex=np.expand_dims(np.expand_dims(rw[0],-1),-1).astype(datcast)
                tfconv=tf.nn.conv2d(np.expand_dims(I.astype(np.float16),0), rwex, [1,1,1,1], padding='VALID').eval().astype(recast)
                if not(lay.KCD):
                    if norm:
                        outp *= 255.0/outp.max()
                    else:
                        outp=outp//3
                suum=rw[0].sum(axis=(1,0)).astype(datcast)
                if 0:
                    if suum==0:
                        suum+=1
                    convp = convp//suum
                    conv1 = conv1.astype(np.uint8)//suum
                    tfconv=tfconv//suum

                if Itype=='L':
                    if I2 is None:
                        I2=I[:,:,0]
                    outp=outp[:,:,0]
                    convp=convp[:,:,0]
                    conv1=conv1[:,:,0]
                    tfconv=tfconv[0,:,:,0]
                    cmapv='gray'
                else:
                    if I2 is None:
                        I2=I
                    cmapv=None
                    tfconv=tfconv[0]
                if method=='pil':
                    com=PIL.Image.fromarray(convp, Itype)
                    co1=PIL.Image.fromarray(conv1, Itype)
                    img = PIL.Image.fromarray(outp, Itype)
                    com.show('conv')
                    img.show('var')
                elif method=='opencv':
                    cv2.imshow('output',outp)
                elif method=="big_mpl":
                    acm[num]={0:I2,1:convp,2:outp,3:conv1,4:tfconv,5:rw[0]}
            if method=="big_mpl":
                acmarg=['Original','imgaug conv','weighted variance','conv with scipy.signal','tf.conv2d','kernel/weight']
                no_original=0
                many_plots=1
                if no_original:
                    start=1
                    col=2
                    row=2
                    subx=0.45
                    suby=0.5
                else:
                    start=0
                    col=3
                    row=2
                    subx=0.48
                    suby=0.55
                i=len(acm)
                il=len(acm[0])
                j=i*il
                if many_plots:
                    for ij in range(i):
                        weight=weigtz[ij]
                        plt.subplots(nrows=row, ncols=col)#, sharex, sharey, squeeze, subplot_kw, gridspec_kw)
                        for column in range(start,il):
                            data=acm[ij][column]
                            #print(data)
                            #print(data.min(),data.max(),column,data.dtype,recast,recast(255.0).dtype)
                            if column==(5+start):
                                cmapv='gray'
                            if 1:
                                data=recast((data-data.min()) * (data.dtype.type(255.0)/(data.max()-data.min())))
                            plt.subplot(row,col,column+1-start)
                            plt.imshow(data,cmap=cmapv)
                            plt.title(acmarg[column],fontdict={'family':'serif','weight':'black','style':'oblique'})
                            plt.colorbar()#mappable, cax, ax)
                        #plt.figlegend(handles=,str(weight),  loc='center')
                        plt.suptitle('kernel'+str(weight),x=subx,y=suby)
                        plt.tight_layout(pad=0.55, h_pad=2.71, w_pad=1.51)#, rect)
                        plt.show()
                else:
                    plt.subplots(nrows=len(acm), ncols=4)#, sharex, sharey, squeeze, subplot_kw, gridspec_kw)
                    for jj in range(j):
                        plt.subplot(i,il,jj+1)
                        xj=jj%il
                        vj=jj//il
                        weight=weigtz[vj]
                        plt.imshow(acm[vj][xj],cmap=cmapv)
                        if vj==0:
                            plt.title(acmarg[xj])
                    plt.show()
