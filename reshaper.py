'''
Created on Dec 13, 2017

@author: ARL
'''
import os
os.environ['PYTHONPATH'] =r'C:\Users\utilisateur\Anaconda\envs\tensorflow'
os.environ['PATH']='C:\\Users\\utilisateur\\Anaconda\\envs\\tensorflow\\;C:\\Program Files (x86)\\Graphviz2.38\\bin\\'
import numpy as np
import mxnet as mx
import mxnet.ndarray as mnd
import cntk as C
import itertools
import varitest as VT
import cntk.ops as CO
import tensorflow as tf
def indary(input):
    ishp=input.shape
    output=[]
    for I in ishp:
        pass
        
def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    "kernel, step, stride,pad,cover, dilation"
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1
def im2col_cpuV2(
        img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None,og=True,reshape=False,channel1=False):
    """
    Extract patches from an image based on the filter.

This function rearranges patches of an image and put them in the channel dimension of the output.

Patches are extracted at positions shifted by multiples of stride from the first position -pad for each spatial axis. The right-most (or bottom-most) patches do not run over the padded spatial size.

Notation: here is a notation.

n is the batch size.
c is the number of the input channels.
h and w are the height and width of the input image, respectively.
kH and kW are the height and width of the filters, respectively.
sY and sX are the strides of the filter.
pH and pW are the spatial padding sizes.
dY and dX are the dilation factors of filter application.
The output size (hO,wO)(hO,wO) is determined by the following equations when cover_all = False:

hOwO=(h+2pH−kH−(kH−1)∗(dY−1))/sY+1,=(w+2pW−kW−(kW−1)∗(dX−1))/sX+1.
hO=(h+2pH−kH−(kH−1)∗(dY−1))/sY+1,wO=(w+2pW−kW−(kW−1)∗(dX−1))/sX+1.
When cover_all = True, the output size is determined by the following equations:

hOwO=(h+2pH−kH−(kH−1)∗(dY−1)+sY−1)/sY+1,=(w+2pW−kW−(kW−1)∗(dX−1)+sX−1)/sX+1.
hO=(h+2pH−kH−(kH−1)∗(dY−1)+sY−1)/sY+1,wO=(w+2pW−kW−(kW−1)∗(dX−1)+sX−1)/sX+1.
Parameters:    
x (Variable) – Input variable of shape (n,c,h,w)(n,c,h,w).
ksize (int or pair of ints) – Size of filters (a.k.a. kernels). ksize=k and ksize=(k, k) are equivalent.
stride (int or pair of ints) – Stride of filter applications. stride=s and stride=(s, s) are equivalent.
pad (int or pair of ints) – Spatial padding width for input arrays. pad=p and pad=(p, p) are equivalent.
cover_all (bool) – If True, all spatial locations are rearranged into some output pixels. It may make the output size larger.
dilate (int or pair of ints) – Dilation factor of filter applications. dilate=d and dilate=(d, d) are equivalent.
Returns:    
Output variable whose shape is (n,c⋅kH⋅kW,hO,wO)(n,c⋅kH⋅kW,hO,wO)

Return type:    
Variable
    """
    #if not(og) and not(channel1):
    #    img=numpy.rollaxis(img, 1, len(img.shape))
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    img = np.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))

    col = np.ndarray((n, c, out_h, out_w,  kh, kw), dtype=img.dtype)
    for R in range(kh):
        jdy = R * dy #window index * dilation
        j_lim = jdy + sy * out_h
        for H in range(kw):
            idx = H * dx #window index * dilation
            i_lim = idx + sx * out_w
            col[:,:,:,:,R,H]=img[:, :, jdy:j_lim:sy, idx:i_lim:sx]#pour chaque point
    if not(channel1):
        col=np.rollaxis(col,1,-2)
    print(col.shape)
    return col  
peakw=1
scale=1
shape=(3,8,8)
window=(3,3)
step=(1,1)
pad=(0,0)
outputs=4
bias_depth=2
numimg=1
print('peakw',peakw,'scale',scale,'shape',shape,'window',window,'stride',step,'pad',pad)

####tests

#vals=shape[0]*shape[1]*shape[2]
ival=shape[0]*shape[1]*shape[2]
vals=window[-2]*window[-1]
inputest=np.reshape(np.linspace(0, 255, num=ival,  dtype=np.float32,endpoint=True),shape)
inputest=np.array([inputest for _ in range(numimg)])
outh=get_conv_outsize(shape[-2], window[-2], step[-2], pad[-2])
outw=get_conv_outsize(shape[-1], window[-1], step[-1], pad[-1])
baseline=im2col_cpuV2(inputest, window[0], window[1], step[0], step[1], pad[0], pad[1], cover_all=False, dy=1, dx=1,
        out_h=outh, out_w=outw,og=0,channel1=0)

#numpy https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html
#broadcast, tile, split ??block?
#print(np.broadcast_to(inputest,baseline.shape))
print(np.split(inputest,3,1)[0].shape)
print(np.repeat(inputest, repeats=(0,3,3), axis=1).shape)#(1,2,3)))


#mxnet https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html

mna=mnd.array(inputest,  dtype=np.float32)
print(mna.shape)
mnind=mna[:,:,0:6,0:6]
mne=mnd.array(np.empty((1,3,6,6,3,3),dtype=np.float32))
"""
print('mne',mne.shape)
for R in range(window[0]):
    print('R',R)
    jdy = R * 1 #window index * dilation
    j_lim = jdy + 1 * mna.shape[-2]-2
    for H in range(window[1]):
        print('H',H)
        idx = H * 1 #window index * dilation
        i_lim = idx + 1 * mna.shape[-1]-2
        mne[:,:,:,:,R,H]=mna[:, :, jdy:j_lim:1, idx:i_lim:1]
print(mne,mne.shape)"""
#mng=mnd.gather_nd(mna,)
#mns=mnd.scatter_nd()
#mnb=mna.broadcast_to((numimg,outh,outw,1,shape[0],*window))
#mnd.take(mna)
print(mnind,mnind.shape)

#print(mnb.shape)
#print(mnb.asnumpy()==baseline)




#cntk https://www.cntk.ai/pythondocs/cntk.ops.sequence.html
#class cntktest(C.layers.layers):
#    def __init__(self,cell,data,kern=(3,3),step=(1,1),pad=(0,0)):
#        self.out=cell
#        self.outh=get_conv_outsize(data[-2], kern[-2], step[-2], pad[-2])
#        self.outw=get_conv_outsize(data[-1], kern[-1], step[-1], pad[-1])
#C.input_variable(shape, dtype, needs_gradient, is_sparse, dynamic_axes, name)
cna=CO.element_times(inputest, 1, name='cnin')
#cwd=C.layers.layers._window(cna,axis=-1,begin=0,end=8,step=1,stride=3)
                      #axis=-filter_rank, begin=-lpad, end=-lpad+filter_shape[-filter_rank], step=1, stride=strides[-filter_rank], initial_state=None)
#cnb=CO.sequence.broadcast_as(operand, broadcast_as_operand, name)
#cnbs=CO.sequence.slice(seq, begin_index, end_index, name)
#cnbsc=CO.sequence.scatter(seq, condition, new_sequence_axis_typeinfo, name)
#print(cwd.shape)
#print(cwd.as_numpy()==baseline)


# tf uses NHWC not NCHW
#tf https://www.tensorflow.org/api_guides/python/array_ops#Shapes_and_Shaping
#code https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/array_ops.py
#https://www.tensorflow.org/api_docs/python/tf/while_loop
#tfb=tf.broadcast_static_shape(shape_x, shape_y)
#tfs=tf.split()
#tfss=tf.strided_slice(tfa,[0,0,0,0],[-1,-1,-1,-1],[1,1,1,1])#https://www.tensorflow.org/api_docs/python/tf/strided_slice
#tfe=tfb.as_numpy() == baseline
#trevs=tf.reverse_sequence(transf, seq_lengths, seq_axis, batch_axis, name, seq_dim, batch_dim)
   
print('original input',inputest[-1,-1])
#inputest=np.reshape(inputest,(1,8,8,3))#np.rollaxis(inputest, 1, 4)
print('inputest',inputest)
convbase=np.tensordot(baseline, np.ones((1,3,3,3), dtype=np.float32), axes=((-3,-2,-1),(-3,-2,-1)))
window=(3,3)
axes=[0,4]
sess=tf.Session()

with sess.as_default() as ff:
    perm=0
    imgtf=tf.convert_to_tensor(inputest, dtype=np.float32, name='input')
    imgtf2=tf.convert_to_tensor(np.rollaxis(inputest,1,4), dtype=np.float32, name='input')
    imgtf3=tf.convert_to_tensor(np.reshape(inputest,(1,8,8,3)), dtype=np.float32, name='input')
    __o=np.array([np.ones(window,dtype=np.float32)+i for i in [0,1,2]],dtype=np.float32)
    _W=tf.convert_to_tensor(np.reshape(__o,(1,*__o.shape)))
    ones=tf.ones((1,3,3,3), dtype=tf.float32)
    ones2=tf.ones((1,1,1,1), dtype=tf.float32)
    tfar=tf.reshape(imgtf,(1,8,8,3))
    tfa=tf.transpose(imgtf, [0,2,3,1])
    conv1=tf.layers.conv2d(tfa, 1, (3,3),  kernel_initializer=tf.ones_initializer(tf.float32), bias_initializer=tf.zeros_initializer(tf.float32),trainable=False)#,data_format='channels_first')
    #tfconvt=tf.layers.conv2d_transpose(tfa, 1, (3,3), kernel_initializer=tf.ones_initializer(tf.float32),bias_initializer=tf.zeros_initializer(tf.float32),trainable=False)
    tf.initialize_all_variables().run()
    tfimp=tf.extract_image_patches(tfa, ksizes=(1,*window,1), strides=(1,1,1,1),rates=(1,1,1,1),padding='VALID')
    tfrs=tf.reshape(tfimp, (*tfimp.shape[:3],1,-1,*window))
    tran=tf.transpose(tfrs, perm=[0,1,2,3,6,4,5])#[0,3,1,2,6,4,5])
    
    
    def var(array,W=_W,B=None,square=0,sqrt=0,V=False,order='NHWC'):
        #W=tf.transpose(W, [0,2,3,1])
        if order=='NHWC':
            W=tf.transpose(W, [0,2,3,1])
            xi=(-3,-2)
            x2=(-1,-3,-2)
        elif order=='NCHW':
            xi=(-2,-1)
            x2=(-2,-1,-3)
        arrs=array.shape
        ashp=W.shape
        if V:
            print(W.eval())
        print(arrs,ashp)
        mul=(array*W)
        #xi=(-3,-1)
        #xi=(-3,-2,-1)
        #x2=(-3,-2,-1)
        #x2=(-4,-3,-2)
        if V:
            print('Wsamp',W[-1,-1].eval())
            print('array*w',(mul.eval())[0])
        size=tf.reduce_sum(W,axis=xi,keep_dims=True)#shape=(outputs, channel)
        if V:
            print("sizesamp",size.shape,size.eval())
        if B is None:
            B=tf.zeros(W.shape[0:2],dtype=tf.float32)#channel
        B=tf.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        mean=tf.reduce_sum((mul),axis=xi,keep_dims=True)/size
        if V:
            print("meansamp",mean.eval()[0])
        if square:
            i=(np.square((mul)-mean)+B)/size
        else:
            i=(((mul)-mean)+B)/(size)
        if V:
            print('isamp',i.shape,i.eval()[-1,-1,])
        out=tf.reduce_sum(i+B,axis=x2)
        #out=np.rollaxis(np.sum(i+B,axis=x2),-1,1)
        print(out.shape)
        if sqrt:
            out=tf.sqrt(out)
        assert out.shape==(arrs[0],arrs[1],arrs[2],ashp[0])
        return(out)
    
    
    print('modified iimage tnsor',tfa.eval(),'modified iimage tnsor')
    def wando(arr):
        print('arr shape',arr.shape)
        temp=tf.extract_image_patches(arr, ksizes=(1,*window,1), strides=(1,1,1,1),rates=(1,1,1,1),padding='VALID') 
        print('temp shape',temp.shape)
        return(tf.reshape(temp,(*temp.shape[:3],1,-1,*window)))
    print('tfa',tfa.eval(),tfa.eval().shape,'TFA')
    print('TFRS',tfrs.eval()[-1,-1],'TFRS')
    print('tfimp',(tfimp.eval())[-1,-1,:],'tfimp')
    """ if perm:
        combos=list(itertools.permutations([1,2,3,4,5,6]))
        for combo in combos:
            cc=[0,].append(combo)
            ttt=tf.transpose(tfrs, cc)
            try:
                if (ttt.eval()==baseline).all():
                    print(combo)
                    break
            except:
                if (ttt.eval()==baseline):
                    print(combo)
                    break"""
    vari=var(tran)
    cov1=conv1.eval()
    #deconv=tfconvt.eval()
    #print('reverse1',trev.eval()[-1,-1,-1],'reverse')
    #print('transpose',tran.eval()[0,3,2],'transpose')
    #print(deconv[-1],deconv.shape)
    print((tran.eval()[0]-baseline[0]).mean((-3,-2,-1)))
    print('conv',cov1[-1],cov1.shape,'conv')
    #print((tr3.eval()[0]-baseline[0]).mean((-3,-2,-1)))
    #print('reshapetf',tfrs.eval()[-1,-1])
    #print(tfss.eval())
    squa=1
    sqrtt=0
    print(convbase,convbase.shape)
    print('tran',tran.eval()[:,-1],'%%'*36,baseline[:,-1],'baseline')
    print('vari',vari.eval())
    print("TRAN W"*12)
    print('tran',var(tran,square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],'tran')#=imgtf2 HIT alg
    print("NHWC IMAGE"*12)
    print('imgtf2',var(wando(imgtf2),square=squa,sqrt=sqrtt,V=0,order='NCHW').eval()[-1,],'imgtf2')#=tfa
    print('tfa',var(wando(tfa),square=squa,V=0,sqrt=sqrtt,order='NHWC').eval()[-1,],'tfa')#=imgtf2 HIT alg
    print("TRANSPOSED IMAGE"*10)
    print('imgtf3',var(wando(imgtf3),square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],'imgtf3')#=tfar
    print('tfar',var(wando(tfar),square=squa,V=0,sqrt=sqrtt,order='NHWC').eval()[-1,],'tfar')#=imgtf3
    print('TRAN 1'*12)
    print('1tran',var(tran,W=ones,square=squa,V=1,sqrt=sqrtt,order='NCHW').eval()[-1,],'1tran')#=imgtf2 hit alg w11
    print("NHWC IMAGE"*12)
    print('1imgtf2',var(wando(imgtf2),W=ones,square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],'1imgtf2')#=tfa
    print('1tfa',var(wando(tfa),W=ones,square=squa,V=0,sqrt=sqrtt,order='NHWC').eval()[-1,],'1tfa')#=imgtf2 hit alg w11
    print("TRANSPOSED IMAGE"*10)
    print('1imgtf3',var(wando(imgtf3),W=ones,square=squa,sqrt=sqrtt,V=0,order='NCHW').eval()[-1,],'1imgtf3')#=tfar
    print('1tfar',var(wando(tfar),W=ones,square=squa,sqrt=sqrtt,V=0,order='NHWC').eval()[-1,],'1tfar')#=imgtf3 hitbase w11
    #print(var())
    print(tran.eval()[:,-1,-1])
#print('target input',baseline[0,-1,-1],'target')


