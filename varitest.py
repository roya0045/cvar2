
"""    if convol:#may be totally useless
        if len(ashp)==3:
            W=W[::-1,::-1,::-1]
            if not(B is None):
                B=B[::-1,::-1,::-1]
        elif len(ashp)==2:
            W=W[::-1,::-1]
            #W=np.fliplr(np.flipud(W))
            if not(B is None):
                B=B[::-1,::-1]
                #B=np.fliplr(np.flipud(B))# does B need to be flipped too?
        elif len(ashp)==4:
            W=W[::-1,::-1,::-1,::-1]
            if not(B is None):
                B=B[::-1,::-1,::-1,::-1]
"""
import chainercutils as cul
import numpy as np
from math import ceil
import itertools
squa=1
sqart=1
v=0
sizz=1
#https://docs.scipy.org/doc/numpy-1.10.0/reference/generated/numpy.apply_over_axes.html#numpy.apply_over_axes
#oginput=[num,chnl,(depth),height/row,width/cols]=4+nd
#altered input=[num,chnl,ouchnl?,(outd),outheight,outwidht,windowheight,windowwidht]=1+2*dims
#weight=[num_ouput,chnel,(depth),height/row,widht/col]=4+nd sum over channel to remove dims
#output=[num,numcel,(depth),H/R,W/C]=4+nd
#n.tensordot(a,b,axes=([axes for a],[axes for b])) the data are summed for the selected axes
#n.matmul== multiplication, no reduction
#n.einsum sum with rules
#dot  are simply dot product
#np.reshape(array,a.shape[:len(w.shape]) #makes an array with as many shape as w for mult
#np.reshape(array,(*W.shape[:2],*a.shape[:len(w.shape]) #don't forget to * if multiple tuple are passed

def baseline(array,w,sqrt=False,square=True):
    def inter(a):
        am=a.mean()
        asiz=len(a.flatten())
        if square:
            av=np.sum(np.square(a-am))/asiz
        else:
            av=np.sum(a-am)/asiz
        return (av)
    wsh=w.shape
    def ameanalg(a):
        return(a-a.mean())
    print(wsh)
    ash=array.shape
    hold=np.empty((ash[0],wsh[0],ash[-2]-2,ash[-1]-2),dtype=np.float32)
    mean=np.empty((ash[0],ash[1],ash[-2]-2,ash[-1]-2),dtype=np.float32)
    Input=np.empty((ash[0],ash[-2]-2,ash[-1]-2,ash[1],3,3),dtype=np.float32)
    for ixm,img in enumerate(array):
        for co in range(wsh[0]):
            temp=np.empty((wsh[1],ash[-2]-2,ash[-1]-2),dtype=np.float32)
            for chnl in range(wsh[1]):
                for col in range(ash[-2]-2):
                    for row in range(ash[-1]-2):
                        #intim=img[chnl,col:col+wsh[-2],row:row+wsh[-1]]
                        #print(intim.shape)
                        temp[chnl,col,row]=inter(img[chnl,col:col+wsh[-2],row:row+wsh[-1]])
                        if co==0 :
                            Input[ixm,col,row,chnl]=img[chnl,col:col+wsh[-2],row:row+wsh[-1]]
                            mean[ixm,chnl,col,row]=img[chnl,col:col+wsh[-2],row:row+wsh[-1]].mean()
            hold[ixm,co]=np.sum(temp,axis=0)
    if sqrt:
        hold=np.sqrt(hold)
    return((hold),mean,Input,temp)

def vecvari10(array,W,B=None,sqrt=False,BB=1,verbose=False,BS=False,sizz=0,KCD=False,mulb=False,**kwargs):#shape of array must be the same as B
    arrs=array.shape
    ashp=W.shape
    #array=np.expand_dims(array,len(array.shape)//2)
    if verbose:
        print('arrayshape',arrs)
        print('Wsample',W[0,0])
        print('wshape',ashp)
    xi=(-2,-1)
    x2=(-3,-2,-1)
    #if len(ashp)==5 :#not all data and all weights == 3d data
    #    xi=(-3,-2,-1)
    #    x2=(-4,-3,-2,-1)
    if mulb:
        B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        mul=(array*W)+B
    else:
        mul=array*W
    if not(B is None)and BB:
        size=(np.sum(W,axis=xi,keepdims=True)+np.sum(B,axis=xi[-len(B.shape):],keepdims=True))
    else:
        size=np.sum(W,axis=xi,keepdims=True)#shape=(outputs, channel)
        if B is None:
            B=np.zeros((ashp[1]))#channel
    if verbose:
        print('sizsamp',size[0,0,0])
        print('bbb',B.shape)
        print("size",size.shape)
        print('array',array.shape,'w',W.shape)
    ######################################
    if sizz==1:
        mean=np.sum(mul,xi,keepdims=1)/size
    else:
        mean=np.sum(mul,xi,keepdims=1)/np.broadcast_to([ashp[-2]*ashp[-1]],(3,1,1))
    if verbose:
        print("meansamp",mean[-1,:,:,-1,-1,-1])
        print("etst",mean.shape)
        print('size',size[0,0])
        print("amean",(mul-mean)[-1,-1,-1])
    B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
    if BB:
        i=(np.square((mul)-mean)+B)/size#B could be included
    else:
        i=(np.square((mul)-mean))/size
    if KCD:
        out=np.sum(i,axis=xi)
    else:
        out=np.rollaxis(np.sum(i,axis=x2),-1,1)
    if verbose:
        print('ishape',i.shape)
        print('isample',i[-1,-1,-1],i.dtype)
    if sqrt:
        out=np.sqrt(out)
    if verbose:
        print("out",out.shape,(arrs[0],ashp[0],arrs[1],arrs[2]))
    if KCD:
        out=np.reshape(out,(arrs[0],ashp[0]*arrs[-3],arrs[1],arrs[2]))
    else:
        assert out.shape==(arrs[0],ashp[0],arrs[1],arrs[2])
    
    if not(BB) and BS:
        B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        return(out+B[:,0])
    else:
        return(out)


###data

def vecvari1(array,W,B=None,sqrt=False,BB=False,BS=False,verbose=False,sizz=1,KCD=False,mulb=False,**kwargs):#shape of array must be the same as B
    arrs=array.shape
    #array=np.expand_dims(array,len(array.shape)//2)
    ashp=W.shape
    if verbose:
        print('arrayshape',arrs)
        print('Wsample',W[0,0])
        print('wshape',ashp)
    bt=len(B.shape)==2
    xi=(-2,-1)#xi=(-1,-2)
    x2=(-3,-2,-1)
    if len(ashp)==5 :#not all data and all weights == 3d data
        xi=(-3,-2,-1)
        x2=(-4,-3,-2,-1)
    if mulb:
        B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        mul=(array*W)+B
    else:
        mul=array*W
    size=np.sum(W,axis=xi,keepdims=True)#shape=(outputs, channel)
    if B is None:
        B=np.zeros(ashp[0:2],dtype=np.float32)#channel
    if BB :
        B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
    if verbose:
        print('sizsamp',size[0,0,0])
        print('bbb',B.shape)
        print("size",size.shape)
    if sizz==1:
        mean=np.sum((mul),axis=xi,keepdims=True)/size
    else:
        mean=np.sum((mul),axis=xi,keepdims=True)/np.broadcast_to([ashp[-2]*ashp[-1]],(ashp[1],1,1))
    if verbose:
        print("meansamp",mean[-1,:,:,-1,-1,-1,-1])
        print("etst",mean.shape)
        print("amean",(mul-mean)[-1,-1,-1])
    if BB:
        i=(np.square((mul)-mean)+B)/size
    else:
        i=(np.square((mul)-mean))/size
    if KCD:
        out=np.sum(i,axis=xi)
    else:
        out=np.rollaxis(np.sum(i,axis=x2),-1,1)
    if verbose:
        print(i.shape)
        print('isample',i[-1,-1,-1],i.dtype)
    if sqrt:
        out=np.sqrt(out)
    if verbose:
        print("out",out.shape,(arrs[0],ashp[0],arrs[-5],arrs[-4]))
    if KCD:
        out=np.reshape(out,(arrs[0],ashp[0]*arrs[-3],arrs[1],arrs[2]))
    else:
        assert out.shape==(arrs[0],ashp[0],arrs[1],arrs[2])
        
    if not(BB)and BS:
        B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        return(out+B[:,0])
    else:
        return(out)


###data


peakw=1
scale=1
shape=(3,8,8)
window=(3,3)
stride=(1,1)
pad=(0,0)
outputs=4
bias_depth=2
numimg=1
print('peakw',peakw,'scale',scale,'shape',shape,'window',window,'stride',stride,'pad',pad)

####tests

ival=shape[0]*shape[1]*shape[2]
vals=window[-2]*window[-1]
#sample input 8 by 8 with 3 channel
inputest=np.reshape(np.linspace(0, 255, num=ival,  dtype=np.float32,endpoint=True),shape)
inputest=np.array([inputest for _ in range(numimg)])
#sample weights
wint=np.linspace(0,peakw,num=ceil(vals/2),dtype=np.float32,endpoint=True)
wint=np.reshape(np.concatenate([wint,wint[:0:-1]]),window)
#array of ones
w11=np.ones((outputs,shape[0],*window), dtype=np.float32)
#array with increasing value for each channel
__o=np.array([np.ones(window,dtype=np.float32)+i for i in [0,1,2]],dtype=np.float32)
_W=np.reshape(__o,(1,*__o.shape))
#identity array
wiid=np.array([[np.identity(window[0], dtype=np.float32) for _ in range(shape[0])] for _ in range(outputs)])
#matrix with upscaling along the channel axis
t11=np.array([[wint+scale*1,wint+scale*2,wint+scale*3] for _ in range(outputs)])

w3=np.reshape(t11,(outputs,shape[0],*window))
w1=np.reshape(np.array([wint for _ in range(outputs)]),(outputs,1,*window))
#biases
b=np.array([1,2,3,4],dtype=np.float32)
b2=np.array([b for _ in range(shape[0])])
#reshaping
b2=np.rollaxis(b2, 1, 0)
inputcols=cul.im2col_cpuV2(inputest, window[0], window[1], stride[0], stride[1], pad[0], pad[1], pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None,og=0,channel1=0)
#inputcolsv1=cul.im2col_cpuV2(inputest, window[0], window[1], stride[0], stride[1], pad[0], pad[1], pval=0, cover_all=False, dy=1, dx=1,
#        out_h=None, out_w=None,og=1,reshape=True
#        ,channel1=0)#horrible, don't use
#print((inputcolsv1-inputcolsv2)[-1])
inputcolsog=cul.im2col_cpuV2(inputest, window[0], window[1], stride[0], stride[1], pad[0], pad[1], pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None,og=True,channel1=False)
#print((inputcolsog-inputcolsv2)[-1])
wid=_W#w11
b0=np.zeros(b2.shape,dtype=np.float32)
BASE,MEAN,INP,iBASE=baseline(inputest, w11,square=squa,sqrt=sqart)
#test with wid


if __name__=='__main__':
    p=vecvari1(inputcols, wid, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=sizz)
    d=vecvari10(inputcols, wid, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=sizz)
    #test with weights all set to 1
    p20=vecvari1(inputcols, w11, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=0)
    d20=vecvari10(inputcols, w11, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=0)
    p21=vecvari1(inputcols, w11, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=1)
    d21=vecvari10(inputcols, w11, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=1)
    compar1=0
    if compar1:
        print('#'*32)
        print('V10b',vecvari10(inputcols,wid,square=1,sqrt=0,B=b2[:-3],verbose=True)[-1,-1])
        print('#'*32)
        print('V10nob',vecvari10(inputcols,wid,square=1,sqrt=0,noB=1,B=b2[:-3],verbose=True)[-1,-1])
        print('#'*32)
    print('tensordot1',np.tensordot(inputcols, wid, axes=((-3,-2,-1),(-3,-2,-1)))[-1,-1])   
    print('tensordot2',np.tensordot(inputcolsog, wid, axes=((1,2,3),(1,2,3)))[-1,-1])
    print(inputcols[-1,-1,-1,:],'\n',inputcolsog[-1,-1,:,-1],'\n',w3[-1],'\n',wid[-1])
    
    print('#'*32)
    
    b20=b2*100
    
    holdd=[b,b2+1,b20,b0]
    compb=0
    if compb:
        for numb,bb in enumerate(holdd):
            print('ROUND ',numb)
            bb=bb[:1]
            b1110=(vecvari1(inputcols,wid,square=0,sqrt=0,B=bb,verbose=1)-vecvari10(inputcols,wid,square=1,sqrt=0,noB=0,B=bb,sizer='trone',verbose=0))
            nb1110=(vecvari1(inputcols,wid,square=0,sqrt=0,B=bb,verbose=0)-vecvari10(inputcols,wid,square=1,sqrt=0,noB=1,B=bb,sizer='trone',verbose=0))
    
            print('#'*32)
            print('rawsv1.1 vs 1.0bb','\n 11b',vecvari1(inputcols,wid,square=0,sqrt=0,B=bb,verbose=0)[-1,-1],'\n 10b',vecvari10(inputcols,wid,square=0,sqrt=0,noB=0,B=bb,sizer='10b',verbose=0)[-1,-1],'\n 10nob',vecvari10(inputcols,wid,square=0,sqrt=0,noB=1,B=bb,sizer='10nob',verbose=0)[-1,-1])
            print('v1.1 vs 1.0bb',b1110.mean(),b1110.max(),b1110.min(),b1110[-1,-1],'\n','v1.1 - 1nob',nb1110.mean(),nb1110.max(),nb1110.min(),nb1110[-1,-1])
            #u=vecvari10(inputcols, wid, B=b0)
    #v=vecvari10(inputcolsv2, wid, B=b0)
    #y=vecvari20(inputcolsog, wid, B=b0)
    print('#'*32)
    #print(v[-1,-1],y[-1,-1],u[-1,-1])
    #print(vecvari1(inputcolsv1, wid, B=b0,verbose=1)[-1,-1])
    

    print('MEANBASE',MEAN[-1],MEAN.shape)
    #mean2=np.rollaxis(MEAN, 1, 4)
    print(iBASE,'IBASE')
    print('BASE',BASE,BASE.shape,)
    print('p with wid',p[-1,-1])
    print('d with wid',d[-1,-1])
    print(np.transpose(p, [0,2,3,1]))
    print((vecvari1(inputcols, wid, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=0)-vecvari1(inputcols, wid, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=1))[-1,-1],'test sizz WID')
    print((vecvari1(inputcols, w11, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=0)-vecvari1(inputcols, w11, B=b0[:1],square=squa,sqrt=sqart,verbose=v,sizz=1))[-1,-1],'test sizz W11')
    #p1=p-BASE
    #d1=d-BASE
    print(BASE[-1,-1],'target?')
    print(inputcols.shape)
    t1=vecvari10(inputcols, wid, B=b0[:1],square=squa,sqrt=sqart,verbose=0,sizz=3,KCD=1)
    t2=vecvari10(inputcols, wid, B=b0[:1],square=squa,sqrt=sqart,verbose=0,sizz=2,KCD=1)
    t3=vecvari1(inputcols, wid, B=b0[:1], sqrt=sqart, square=squa, verbose=0, sizz=0, KCD=1)
    print(t1-t2)
    #print(p1.mean(),p1.max(),p1.min())
    #print(d1.mean(),d1.max(),d1.min())
    #print(INP.shape,inputcols.shape)
    pos=(0,1,3)
    #print('impucols',inputcols[pos],'baseline',INP[pos],INP.shape)
    #print(INP)#'im2col original',inputcolsog[pos]))
    #print('rreshaped og',np.reshape(inputcolsog,inputcols.shape)[-1,-1,-1])
    #print(inputcols==INP)
    #print(BASE,p)
    #print(np.tensordot(inputcolsv1,wid,axes=((-2,-1),(-2,-1))))
    ####cleaned code v1 ###
    print(np.array([-4.56674e-06])/9)


def VecVariB(array,W,B=None,sqrt=False,square=False):
    arrs=array.shape
    ashp=W.shape
    mul=array*W
    #array=np.expand_dims(array,len(array.shape)//2)
    xi=(-2,-1)
    x2=(-3,-2,-1)
    if len(ashp)==5 :#not all data and all weights == 3d data
        xi=(-3,-2,-1)
        x2=(-4,-3,-2,-1)
    if not(B is None):
        size=(np.sum(W,axis=xi,keepdims=True)+np.sum(B,axis=xi[-len(B.shape):],keepdims=True))
    else:
        size=np.sum(W,axis=xi,keepdims=True)#shape=(outputs, channel)
        if B is None:
            B=np.zeros((W.shape[1]))#channel
    try:
        mean=np.sum(((mul)+B),xi)/size
    except Exception as E:
        B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        mean=np.sum(((mul)+B),xi,keepdims=1)/size
    if len(B.shape)==1:
        B=np.reshape(B,(*B.shape,1))
    if square:
        i=(np.square((mul)-mean)+B)/size
    else:
        i=(mul-mean)/(size)
    out=np.reshape(np.sum(i+B,axis=x2),(arrs[0],ashp[0],arrs[1],arrs[2]))
    #out=np.rollaxis(np.sum(i,axis=x2),-1,1)
    if sqrt:
        out=np.sqrt(out)
    assert out.shape==(arrs[0],ashp[0],arrs[1],arrs[2])
    return(out)


def VecVari1(array,W,B=None,sqrt=False,square=False):
    arrs=array.shape
    ashp=W.shape
    mul=array*W
    xi=(-2,-1)
    x2=(-3,-2,-1)
    if len(ashp)==5 :#not all data and all weights == 3d data
        xi=(-3,-2,-1)
        x2=(-4,-3,-2,-1)
    size=np.sum(W,axis=xi,keepdims=True)#shape=(outputs, channel)
    if B is None:
        B=np.zeros(W.shape[0:2],dtype=np.float32)#channel
    B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
    mean=np.sum((mul),axis=xi,keepdims=True)/size
    if square:
        i=(np.square((mul)-mean)+B)/size
    else:
        i=((mul)-mean)/(size)
    out=np.reshape(np.sum(i+B,axis=x2),(arrs[0],ashp[0],arrs[1],arrs[2]))
    #out=np.rollaxis(np.sum(i+B,axis=x2),-1,1)
    if sqrt:
        out=np.sqrt(out)
    assert out.shape==(arrs[0],ashp[0],arrs[1],arrs[2])
    return(out)

def vecvari2(array,W,B=None,sizer='',sqrt=False,square=False,mean=None,noB=False,convol=False,verbose=False):#shape of array must be the same as B
    arrs=array.shape
    ashp=W.shape
    #array=np.expand_dims(array,1)
    if verbose:
        print(sizer)
        print('arrayshape',arrs)
        print('wshape',ashp)
  
    xi=(2,3)
    x2=(2,3,4)
    xib=(0,1)
    if len(ashp)==5 :#not all data and all weights == 3d data
        xi=(2,3,4)
        x2=(2,3,4,5)
        xib=(0,1,2)
    size=np.sum(W,axis=xi,keepdims=True)#shape=(outputs, channel)
    if B is None:
        B=np.zeros(W.shape[0:2],dtype=np.float32)#channel
    B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
    if verbose:
        print('bbb',B.shape)
        print("size",size.shape)
        print('sizesamp',size[0,0])
    if verbose:
        print('array',array.shape,'w',W.shape)
    mean=np.tensordot(array,W/size,axes=(xi,xi))#mean?

    if verbose:
        print("etst",mean.shape)
        print('meansamp',mean[-1,-1,-1,-1])
    ishp=(mean.shape[0],*mean.shape[2:-1],*mean.shape[1::4],1)

    if verbose:
        print('ishp2',ishp)
    #t=array-mean
    if square:
        #i=(np.square((array*W)-mean)+B)/size
        #i=(np.reshape(((np.square(np.tensordot(array,W/size,axes=(xi,xi))-mean))),ishp))/(size)#-1)#(np.sum(W,axis=xi,keepdims=True)-1)
        i=np.reshape((np.square(np.tensordot(array,W,axes=(xi,xi))-mean)),ishp)/(size)#-1)#(np.sum(W,axis=xi,keepdims=True)-1)
    else:
        #i=(((array*W)-mean)+B)/size
        i=(np.reshape((((np.tensordot(array,W,axes=(xi,xi))-mean))),ishp))/(size)
        #print(i[0,0,0])
    if not(sqrt):
        i=np.sqrt(i)

    if verbose:
        print('iii',i.shape)

    out=np.rollaxis(np.sum(i+B,axis=(-3,-2,-1)),-1,1)
    if verbose:
        print(out.shape)
        print('isample',i[-1,-1,-1],i.dtype)

    if verbose:
        print('outhspa???',arrs[0],ashp[0],arrs[-2],arrs[-1])
    assert out.shape==(arrs[0],ashp[0],arrs[-2],arrs[-1])
    return(out)
def vecvari20(array,W,B=None,sizer='',sqrt=False,square=False,mean=None,noB=False,convol=False,verbose=False):#shape of array must be the same as B

    arrs=array.shape
    ashp=W.shape
    #array=np.expand_dims(array,len(array.shape)//2)
    if verbose:
        print(sizer)
        print('arrayshape',arrs)
        print('wshape',ashp)
  
    xi=(2,3)
    x2=(2,3,4)
    xib=(0,1)
    if len(ashp)==5 :#not all data and all weights == 3d data
        xi=(2,3,4)
        x2=(2,3,4,5)
        xib=(0,1,2)
    if not(B is None) and not(noB):
        if verbose:
            print(B.shape)
            print('xib',xi[-len(B.shape):])
        size=(np.sum(W,axis=xi,keepdims=True)+np.sum(B,axis=xib[:len(B.shape)],keepdims=True))
    else:
        size=np.sum(W,axis=xi,keepdims=True)#shape=(outputs, channel)
        if B is None:
            B=np.zeros((W.shape[0]))#channel
    if verbose:
        print('bbb',B.shape)
        print("size",size.shape)
    #assert array.shape==W.shape
    if verbose:
        print('array',array.shape,'w',W.shape)

    try:
        if noB:
            mean=np.tensordot(array,W/size,axes=(xi,xi))#mean?
            #mean=(array*W)/size
        else:
            mean=np.tensordot(array,(W+B)/size,axes=(xi,xi))#mean?
            #mean=((array*W)+B)/size
        bimi=B.shape[-1]
    except Exception as E:
        B2=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        #######################################
        if verbose:
            print(E)
            print("B2")
            print(B2.shape)
        bimi=B2.shape[-3]
        if noB:
            mean=np.tensordot(array,(W/size),axes=(xi,xi))#mean?
        else:
            mean=np.tensordot(array,((W+B2)/size),axes=(xi,xi))#mean?
    if verbose:
        print("etst",mean.shape)
        print('size',size[0,0])
        print('meansamp',mean[-1,-1,-1,-1])
    if len(B.shape)==1:
        B=np.reshape(B,(*B.shape,1))
    
    ishp=(mean.shape[0],*mean.shape[2:-1],*mean.shape[1::4],1)

    #print(mean[0,0,0])
    if verbose:
        print('ishp2',ishp)
    #t=array-mean
    if square:
        i=np.reshape((np.square(np.tensordot(array,W,axes=(xi,xi))-mean)),ishp)/(size)#-1)#(np.sum(W,axis=xi,keepdims=True)-1)
        #i=(np.reshape(((np.square(np.tensordot(array,W/size,axes=(xi,xi))-mean)+B)),ishp))/(size)#-1)#(np.sum(W,axis=xi,keepdims=True)-1)
    else:
        i=(np.reshape((((np.tensordot(array,W,axes=(xi,xi))-mean)+B)),ishp))/(size)
        #print(i[0,0,0])
    if not(sqrt):
        i=np.sqrt(i)

    if verbose:
        print('iii',i.shape)
        print('isample',i[-1,-1,-1],i.dtype)
    if noB:
        B=np.reshape(B,(*B.shape,*[1 for _ in range(len(ashp)-len(B.shape))]))
        out=np.rollaxis(np.sum(i+B,axis=(-3,-2,-1)),-1,1)
    else:
        out=np.rollaxis(np.sum(i,axis=(-3,-2,-1)),-1,1)

    if verbose:
        print(out.shape)
    #print(out[0,0,0])
    #print("out",out.shape)
    #assert out.shape==(arrs[0],ashp[0],arrs[-5],arrs[-4])
    if verbose:
        print('outhspa???',arrs[0],ashp[0],arrs[-2],arrs[-1])
    assert out.shape==(arrs[0],ashp[0],arrs[-2],arrs[-1])
    return(out)