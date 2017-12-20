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
import timeit
import tensorflow as tf
squa=VT.squa
sqrtt=VT.sqart

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
baseline=np.expand_dims(baseline,len(baseline.shape)//2)
#numpy https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html
#broadcast, tile, split ??block?
#print(np.broadcast_to(inputest,baseline.shape))
print(np.split(inputest,3,1)[0].shape)
print(np.repeat(inputest, repeats=(0,3,3), axis=1).shape)#(1,2,3)))
MX=1
if MX:
    #mxnet https://mxnet.incubator.apache.org/tutorials/basic/ndarray.html
    
    mna=mnd.array(inputest,  dtype=np.float32)
    print(mna.shape)
    #MXNET_ENGINGE_TYPE
    __o=np.array([np.ones(window,dtype=np.float32)+i for i in [0,1,2]],dtype=np.float32)
    _W=mnd.array(np.reshape(__o,(1,*__o.shape)))
    #imag=mx.image.image.
    
    def mxwindow(mna,window):
        mnas=mna.shape
        mnout=(*mnas[:-2],*window,((mnas[-2]-window[-2])+1),((mnas[-1]-window[-1])+1))
        mne2=None
        for R in range(window[0]):
            j_lim = R + mnout[-2]
            for H in range(window[1]):
                tdata=mnd.slice(mna, begin=(None,None,R,H), end=(None,None,j_lim,(H +  mnout[-1])), step=(None,None,1,1))
                if mne2 is None:
                    mne2=tdata
                else:
                    mne2=mnd.concat(mne2,tdata,dim=1)
        return(mnd.expand_dims(mnd.transpose(mnd.reshape(mne2, shape=mnout),axes=(0,5,4,3,2,1)), 3))
    #print(mne2.shape,mne2)
    window=(3,3)
    #mne2=mxwindow(mna,window)
    def var(array,W=_W,B=None,square=0,sqrt=0,V=False,order='NCHW',sizz=0):
        arrs=array.shape
        ashp=W.shape
        xi=(-2,-1)
        x2=(-2,-1,-3)
        sb=(ashp[1],1,1)
        WV=ashp[-2:]
        print(sb)

        mnc=mnd.tile(mnd.reshape(mnd.array([WV[0]*WV[1]]), shape=(1,1,1)),ashp[1])
        print(mnc)

        if V:
            print(W.eval())
        print(arrs,ashp)
        mul=(mnd.broadcast_mul(array,W))
        if V:
            print('Wsamp',W[-1,-1])
            print('array*w',mul[0,-1])
        size=mnd.sum(W,axis=xi,keepdims=True)#shape=(outputs, channel)
        if V:
            print("sizesamp",size.shape,size)
        if B is None:
            B=mnd.zeros(W.shape[0:2],dtype=np.float32)#channel
        B=mnd.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        if sizz==1:
            mean=mnd.sum(mul,axis=xi,keepdims=True)/size
        else:
            mean=mnd.sum(mul,axis=xi,keepdims=True)/mnc
        if V:
            print("meansamp",mean[0,-1])
        if square:
            i=mnd.square(mnd.broadcast_add(mnd.broadcast_minus(mul,mean),B))
        else:
            i=mnd.broadcast_add(mnd.broadcast_minus(mul,mean),B)
        di=i/size
        if V==2:
            print("i",i,"i")
            print("di",di,"di")
        if V:
            print('isamp',i.shape,i[-1,-1,])
        out=mnd.sum(mnd.broadcast_add(i,B),axis=x2)
        #out=np.rollaxis(np.sum(i+B,axis=x2),-1,1)
        #print(out.shape)
        if sqrt:
            out=mnd.sqrt(out)
        out=mnd.swapaxes(out, 3, 1)
        #print(out.shape,(arrs[0],ashp[0],arrs[1],arrs[2]))
        assert out.shape==(arrs[0],ashp[0],arrs[1],arrs[2])
        return(out)
    mnd22=mnd.array(baseline, dtype=np.float32)
    print("MXNET",var(mxwindow(mna, window),square=squa,sqrt=sqrtt),"MXNET")
    print("MXNET2",var(mxwindow(mna, window),square=squa,sqrt=sqrtt),"MXNET2")
    '''
    def t1():#46sec
        mne2r=mnd.reshape(mne2, shape=(1,3,3,3,6,6))#mne.shape)#, reverse, target_shape, keep_highest, out, name)
        mne2rt=mnd.transpose(mne2r,axes=(0,5,4,3,2,1)) #HIT
        mne2rt=mnd.expand_dims(mne2rt, (len(mne2r.shape)//2))
        return(mne2rt)
    def t2():#faster, 41sec,36
        return(mnd.expand_dims(mnd.transpose(mnd.reshape(mne2, shape=(1,3,3,3,6,6)),axes=(0,5,4,3,2,1)), 3))'''
    #print(mne2)
    #print(mne2rt,mne2rt.shape)
    #print(mnd.reshape(mne2rt,shape=(1,6,6,1,3,3,3)))
    #print(timeit.timeit(stmt=t1, number=100000),timeit.timeit(stmt=t2, number=100000))
    #print(mne,mne.shape)
    #mng=mnd.gather_nd(mna,)
    #mns=mnd.scatter_nd()
    #mnb=mna.broadcast_to((numimg,outh,outw,1,shape[0],*window))
    #mnd.take(mna)
    #mnind=mna[:,:,0:6,0:6]
    #print(mnind,mnind.shape)
    
    #print(mnb.shape)
    #print(mnb.asnumpy()==baseline)
    
    #print(baseline[0,-1,-1])

cn=1
if cn:
    #cntk https://www.cntk.ai/pythondocs/cntk.ops.sequence.html
    #class cntktest(C.layers.layers):
    #    def __init__(self,cell,data,kern=(3,3),step=(1,1),pad=(0,0)):
    #        self.out=cell
    #        self.outh=get_conv_outsize(data[-2], kern[-2], step[-2], pad[-2])
    #        self.outw=get_conv_outsize(data[-1], kern[-1], step[-1], pad[-1])
    #C.input_variable(shape, dtype, needs_gradient, is_sparse, dynamic_axes, name)
    cna=CO.element_times(inputest, 1,)
    
    cnts=C.sequence.input_variable(inputest.shape, dtype=np.float32)#,sequence_axis=1)
    cnts2=C.sequence.input_variable(baseline.shape,dtype=np.float32)
    print(cna.shape)
    axs=1
    def cnwindow(mna,window):
        mnas=mna.shape
        mnout=(*mnas[:-2],*window,((mnas[-2]-window[-2])+1),((mnas[-1]-window[-1])+1))
        mne2=None
        for R in range(window[0]):
            j_lim = R + mnout[-2]
            for H in range(window[1]):
                tdata=C.slice(mna,[-2,-1], [R,H], [j_lim,(H +  mnout[-1])])
                if mne2 is None:
                    mne2=tdata
                else:
                    mne2=C.splice(mne2,tdata,axis=1)
        return(C.reshape(C.transpose(C.reshape(mne2, shape=mnout),(0,5,4,3,2,1)), (mnout[0],*mnout[5:3:-1],1,*mnout[3:0:-1])))
    cnwt=cnwindow(cnts, (3,3))
    _W=C.constant(1,(1,3,3,3),dtype=np.float32)
    #print(cnwt.eval({cnts:inputest}))
    #print(C.reduce_sum(_W,(-2,-1)))
    def var(array,W=_W,B=None,square=0,sqrt=0,V=False,sizz=0):
        #W=tf.transpose(W, [0,2,3,1])
        
        arrs=array.shape
        ashp=W.shape
        sb=(W.shape[1],1,1)
        WV=W.shape[-2:]
        xi=(-2,-1)
        x2=(-2,-1,-3)

        if V:
            print(W.eval())
            print(arrs,ashp)
        mul=(array*W)

        if V:
            print('Wsamp',W[-1,-1].eval())
            print('array*w',(mul.eval())[0,-1])

        size=C.reduce_sum(W,axis=xi)#shape=(outputs, channel)

        if V:
            print("sizesamp",size.shape,size.eval())
        if B is None:
            B=C.constant(0,shape=W.shape[0:2],dtype=np.float32)#channel
        B=C.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        if sizz==1:
            mean=C.reduce_sum(mul,axis=xi)/size
        else:
            mean=C.reduce_sum(mul,axis=xi)/C.constant(value=WV[0]*WV[1],shape=sb,dtype=np.float32)
        if V:
            print("meansamp",mean.eval()[0,-1])
        if square:
            i=(C.square(mul-mean)+B)
        else:
            i=(((mul)-mean)+B)
        di=i/size
        if V==2:
            print("i",i.eval(),"i")
            print("di",di.eval(),"di")
        if V:
            print('isamp',i.shape,i.eval()[-1,-1,])
        out=C.reduce_sum(i+B,axis=x2)
        #out=np.rollaxis(np.sum(i+B,axis=x2),-1,1)
        print(out.shape)
        if sqrt:
            out=C.sqrt(out)
        out=C.swapaxes(C.reshape(out,out.shape[:4]), 3, 1)
        print(out.shape)
        assert out.shape==(arrs[0],ashp[0],arrs[1],arrs[2])
        return(out)
    CWOne=C.constant(1,shape=(1,3,3,3),dtype=np.float32)
    cnvart=var(cnwt,W=CWOne,square=squa,sqrt=sqrtt,sizz=1)
    cnvart2=var(cnts2,W=CWOne,square=squa,sqrt=sqrtt,sizz=1)
    print("CNTK",cnvart.eval({cnts:inputest}),"CNTK")
    print("CNTK2",cnvart2.eval({cnts2:baseline}),"cntk2")
    #crsh=C.reshape(cna, (*cna.shape,1,1))
    #print(crsh.shape,crsh[0,2,2,2,0,0])
    #cnts2=C.sequence.input_variable(crsh.shape, dtype=np.float32)
    #cwd2=C.layers.layers._window(cnts2,axis=axs,begin=0,end=3,step=1,stride=1)
    #cwd=C.layers.layers._window(cna,axis=3,begin=0,end=8,step=1,stride=1)
                          #axis=-filter_rank, begin=-lpad, end=-lpad+filter_shape[-filter_rank], step=1, stride=strides[-filter_rank], initial_state=None)
    #cnb=CO.sequence.broadcast_as(operand, broadcast_as_operand, name)
    #cnbs=CO.sequence.slice(seq, begin_index, end_index, name)
    #cnbsc=CO.sequence.scatter(seq, condition, new_sequence_axis_typeinfo, name)
    #print(cwd2.shape,cwd2)
    #test=cwd2(crsh)
    #print(test.shape)
    #for I in range(test.shape[axs+1]):
    #    print(I)
    #    print(test[:,:,I])
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
TF=1
if TF:
    #print('original input',inputest[-1,-1])
    #inputest=np.reshape(inputest,(1,8,8,3))#np.rollaxis(inputest, 1, 4)
    #print('inputest',inputest)
    convbase=np.tensordot(baseline, np.ones((1,3,3,3), dtype=np.float32), axes=((-3,-2,-1),(-3,-2,-1)))
    window=(3,3)
    axes=[0,4]
    sess=tf.Session()
    
    with sess.as_default() as ff:
        perm=0
        BASETF=tf.convert_to_tensor(baseline,dtype=np.float32)
        imgtf=tf.convert_to_tensor(inputest, dtype=np.float32, name='input')
        imgtf2=tf.convert_to_tensor(np.rollaxis(inputest,1,4), dtype=np.float32, name='input')
        imgtf3=tf.convert_to_tensor(np.reshape(inputest,(1,8,8,3)), dtype=np.float32, name='input')
        __o=np.array([np.ones(window,dtype=np.float32)+i for i in [0,1,2]],dtype=np.float32)
        _W=tf.convert_to_tensor(np.reshape(__o,(1,*__o.shape)))
        ones=tf.ones((1,3,3,3), dtype=tf.float32)
        ones2=tf.ones((1,1,1,1), dtype=tf.float32)
        tfar=tf.reshape(imgtf,(1,8,8,3))
        tfa=tf.transpose(imgtf, [0,2,3,1])#NWHC hit
        conv1=tf.layers.conv2d(tfa, 1, (3,3),  kernel_initializer=tf.ones_initializer(tf.float32), bias_initializer=tf.zeros_initializer(tf.float32),trainable=False)#,data_format='channels_first')
        #tfconvt=tf.layers.conv2d_transpose(tfa, 1, (3,3), kernel_initializer=tf.ones_initializer(tf.float32),bias_initializer=tf.zeros_initializer(tf.float32),trainable=False)
        tf.initialize_all_variables().run()
        tfimp=tf.extract_image_patches(tfa, ksizes=(1,*window,1), strides=(1,1,1,1),rates=(1,1,1,1),padding='VALID')
        tfrs=tf.reshape(tfimp, (*tfimp.shape[:3],1,-1,*window))
        tran=tf.transpose(tfrs, perm=[0,1,2,3,6,4,5])#[0,3,1,2,6,4,5]) NCWH hit
        
        
        def var(array,W=_W,B=None,square=0,sqrt=0,V=False,order='NHWC',sizz=0):
            #W=tf.transpose(W, [0,2,3,1])
            arrs=array.shape
            ashp=W.shape
            if order=='NHWC':
                W=tf.transpose(W, [0,2,3,1])
                sb=(1,1,W.shape[-1])
                WV=W.get_shape().as_list()[1:3]
                xi=(-3,-2)
                x2=(-1,-3,-2)
            elif order=='NCHW':
                sb=(W.shape[1],1,1)
                WV=W.get_shape().as_list()[-2:]
                xi=(-2,-1)
                x2=(-2,-1,-3)

            if V:
                print(W.eval())
                print(arrs,ashp)
            mul=(array*W)
    
            if V:
                print('Wsamp',W[-1,-1].eval())
                print('array*w',(mul.eval())[0,-1])

            size=tf.reduce_sum(W,axis=xi,keep_dims=True)#shape=(outputs, channel)

            if V:
                print("sizesamp",size.shape,size.eval())
            if B is None:
                B=tf.zeros(W.shape[0:2],dtype=tf.float32)#channel
            B=tf.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
            if sizz==1:
                mean=tf.reduce_sum(mul,axis=xi,keep_dims=True)/size
            else:
                mean=tf.reduce_sum(mul,axis=xi,keep_dims=True)/tf.constant(WV[0]*WV[1],shape=sb,dtype=tf.float32)
            if V:
                print("meansamp",mean.eval()[0,-1])
            if square:
                i=(tf.square(mul-mean)+B)
            else:
                i=(((mul)-mean)+B)
            di=i/size
            if V==2:
                print("i",i.eval(),"i")
                print("di",di.eval(),"di")
            if V:
                print('isamp',i.shape,i.eval()[-1,-1,])
            out=tf.reduce_sum(i+B,axis=x2)
            #out=np.rollaxis(np.sum(i+B,axis=x2),-1,1)
            print(out.shape)
            if sqrt:
                out=tf.sqrt(out)
            assert out.shape==(arrs[0],arrs[1],arrs[2],ashp[0])
            return(out)
        
        
        #print('modified iimage tnsor',tfa.eval(),'modified iimage tnsor')
        def tfwindow(arr):
            temp=tf.extract_image_patches(arr, ksizes=(1,*window,1), strides=(1,1,1,1),rates=(1,1,1,1),padding='VALID') 
            return(tf.reshape(temp,(*temp.shape[:3],1,-1,*window)))
        #print('tfa',tfa.eval(),tfa.eval().shape,'TFA')
        #print('TFRS',tfrs.eval()[-1,-1],'TFRS')
        #print('tfimp',(tfimp.eval())[-1,-1,:],'tfimp')
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

        #print((tran.eval()[0]-baseline[0]).mean((-3,-2,-1)))
        #print('conv',cov1[-1],cov1.shape,'conv')
        #print((tr3.eval()[0]-baseline[0]).mean((-3,-2,-1)))
        #print(tfss.eval())

        #print(convbase,convbase.shape)
        #print('tran',tran.eval()[:,-1],'%%'*36,baseline[:,-1],'baseline')
        #print('vari',vari.eval())
        print("TRAN W"*12)
        print('tran',var(tran,square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],'tran')#=imgtf2 HIT alg
        print("NHWC IMAGE"*12)
        #print('imgtf2',var(tfwindow(imgtf2),square=squa,sqrt=sqrtt,V=0,order='NCHW').eval()[-1,],'imgtf2')#=tfa
        print('tfa',var(tfwindow(tfa),square=squa,V=0,sqrt=sqrtt,order='NHWC').eval()[-1,],'tfa')#=imgtf2 HIT alg
        #print("TRANSPOSED IMAGE"*10)
        #print('imgtf3',var(tfwindow(imgtf3),square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],'imgtf3')#=tfar
        #print('tfar',var(tfwindow(tfar),square=squa,V=1,sqrt=sqrtt,order='NHWC').eval()[-1,],'tfar')#=imgtf3
        print('TRAN 1'*12)
        print('1tran',var(tran,W=ones,square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],'1tran')#=imgtf2 hit alg w11
        print("NHWC IMAGE"*12)
        #print('1imgtf2',var(tfwindow(imgtf2),W=ones,square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],'1imgtf2')#=tfa
        print('1tfa',var(tfwindow(tfa),W=ones,square=squa,V=0,sqrt=sqrtt,order='NHWC').eval()[-1,],'1tfa')#=imgtf2 hit alg w11
        #print("TRANSPOSED IMAGE"*10)
        #print('1imgtf3',var(tfwindow(imgtf3),W=ones,square=squa,sqrt=sqrtt,V=0,order='NCHW').eval()[-1,],'1imgtf3')#=tfar
        #print('1tfar',var(tfwindow(tfar),W=ones,square=squa,sqrt=sqrtt,V=2,order='NHWC').eval()[-1,],'1tfar')#=imgtf3 
        #print(var())
        print("TFBtestNHWC",var(BASETF,W=ones,square=squa,V=0,sqrt=sqrtt,order='NHWC').eval()[-1,],"TFBtestNHWC")
        print("TFBtestNCHW",var(BASETF,W=ones,square=squa,V=0,sqrt=sqrtt,order='NCHW').eval()[-1,],"TFBtestNCHW")
        #t1=tfwindow(tfa).eval()
        #t2=tfwindow(tfar).eval()
        #print('tfa',t1[0,2],'tfa')
        #print('tfar',t2[0,2],'tfar')
        #for i in [(-2,-1),(-3,-2,-1)]:
        #    print(t1.min(i)-t1.max(i),'t1,min,max')
        #    print(t2.min(i)-t2.max(i),'t2 min max')
        #print('tran',tran.eval()[:,-1,-1])
    #print('target input',baseline[0,-1,-1],'target')
b0=b0=np.zeros((4,3),dtype=np.float32)
print("TARGET",VT.vecvari1(baseline, VT.w11[:1], B=b0[:1],square=squa,sqrt=sqrtt,verbose=0,sizz=1))
print("TARG2",VT.d2[-1,-1],"TARG2")
print("TArgetv3",VT.p2[-1,-1],"TArgetv3")