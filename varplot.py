
#check for implementing weight variance and the effect of variables on the output
from varitest import *
import matplotlib as mlp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cpy
bs=list()
bxx=[]
WAR=[]
numout=5
bseed=[0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,25,50,75,100,250,500,750,1000,5000]
for B in bseed:
    
    tmp=np.array([B,],dtype=np.float32)
    custW=np.broadcast_to(tmp, shape=(numout,3,3,3))
    WAR.append(custW)

for I in bseed:
    e=[I,I*2,I*3,I*4,I*5]
    for ii in e:
        bxx.append(ii)
    b=np.rollaxis(np.array([e for _ in range(shape[0])],dtype=np.float32),1,0)
    bs.append(b)
#print(bxx)

indict={'mean':[],'max':[],'min':[],'1mean':[],'1max':[],'1min':[],'2mean':[],'2max':[],'2min':[],}
outputs={1:cpy(indict),2:cpy(indict),3:cpy(indict)}
sizz=(0,0)
wid=np.broadcast_to(wid, (numout,*wid.shape[1:]))
sqrtv=0
mulbv=1
print(len(bs),bs[0])
def plotvars(BB):
    outputs={1:cpy(indict),2:cpy(indict),3:cpy(indict)}
    axs=(0,-2,-1)
    ding=0
    hol=[bs,WAR][BB]
    for bb in hol:
        print(bb[:,0])
        if BB==0:
            #print(wid.shape,bb.shape)
            d1=vecvari10(inputcols, wid,BB=0, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[0])
            d2=vecvari10(inputcols, wid,BB=1, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[0])
            d3=vecvari10(inputcols, wid,BB=0,BS=1, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[0])#-vecvari10(inputcols, wid,noB=0, B=bb,square=0,sqrt=0,verbose=0)
            p1=vecvari1(inputcols, wid, B=bb,BB=0,sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[1])
            p2=vecvari1(inputcols, wid, B=bb,BB=1,sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[1])
            p3=vecvari1(inputcols, wid, B=bb,BB=0,BS=1,sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[1])
        elif BB==1:
            print(bb.shape,bs[9].shape)
            d=vecvari10(inputcols, bb, B=bs[9],sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[0])
            p=vecvari1(inputcols, bb, B=bs[9],sqrt=sqrtv,verbose=0,mulb=mulbv,sizz=sizz[1])
        for iou,out in enumerate([(d1,d2,d3),(p1,p2,p3),(d1-p1,d2-p2,d3-p3)]):
            if iou==0:
                ding+=1
            iou+=1
            #print(iou)
            #print(out.mean(axs).shape)
            out1=out[0]
            out2=out[1]
            out3=out[2]#out1-out2
            [outputs[iou]['1mean'].append(oo) for oo in out1.mean(axs)]
            [outputs[iou]['1max'].append(oo) for oo in out1.max(axs)]
            [outputs[iou]['1min'].append(oo) for oo in out1.min(axs)]
            [outputs[iou]['2mean'].append(oo) for oo in out2.mean(axs)]
            [outputs[iou]['2max'].append(oo) for oo in out2.max(axs)]
            [outputs[iou]['2min'].append(oo) for oo in out2.min(axs)]
            [outputs[iou]['mean'].append(oo) for oo in out3.mean(axs)]
            [outputs[iou]['max'].append(oo) for oo in out3.max(axs)]
            [outputs[iou]['min'].append(oo) for oo in out3.min(axs)]
    bx=bxx#np.linspace(0, bxx[-1], num=outputs[1]['mean'].__len__(),  dtype=np.float32)
    print(ding,(np.array(outputs[1]['mean']),np.array(outputs[2]['mean'])))
    #print([outputs[d]['min'] for d in outputs ])
    tree=1
    ylog=0
    if tree: #tree plots
        fig, axs = plt.subplots(3, 1)
        #fig.set_xscale('log')
        plt.axis([0,max(bx),min(outputs[3]['min']),max([max(outputs[d]['max']) for d in outputs ])])
        f1=plt.subplot(311)
        plt.plot(bx,outputs[1]['1mean'],'g<--',
                 bx,outputs[1]['1max'],'m<--',
                 bx,outputs[1]['1min'],'b<--',
                 bx,outputs[1]['2mean'],'g>--',
                 bx,outputs[1]['2max'],'m>--',
                 bx,outputs[1]['2min'],'c>--',
                 bx,outputs[1]['mean'],'ys-',
                 bx,outputs[1]['max'],'rs-',
                 bx,outputs[1]['min'],'bs-',markersize=6,)
        plt.setp(f1.set_xscale('log'))
        plt.setp(f1.set_title('vecvari10,sizz{}, mulb{}'.format(sizz,mulbv)))
        if ylog:
            plt.setp(f1.set_yscale('log'))
        f2=plt.subplot(312,sharex=f1)#,sharey=f1)
        plt.plot(bx,outputs[2]['1mean'],'g<--',
                 bx,outputs[2]['1max'],'m<--',
                 bx,outputs[2]['1min'],'b<--',
                 bx,outputs[2]['2mean'],'y>--',
                 bx,outputs[2]['2max'],'m>--',
                 bx,outputs[2]['2min'],'c>--',
                 bx,outputs[2]['mean'],'yo-',
                 bx,outputs[2]['max'],'ro-',
                 bx,outputs[2]['min'],'bo-',markersize=6,)
        plt.setp(f2.set_xscale('log'))
        plt.setp(f2.set_title('vecvari1,sizz{}, mulb{}'.format(sizz,mulbv)))
        if ylog:
            plt.setp(f2.set_yscale('log'))
        f3=plt.subplot(313,sharex=f1)#,sharey=f1)
        plt.plot(bx,outputs[3]['1mean'],'g<--',
                 bx,outputs[3]['1max'],'m<--',
                 bx,outputs[3]['1min'],'b<--',
                 bx,outputs[3]['2mean'],'y>--',
                 bx,outputs[3]['2max'],'m>--',
                 bx,outputs[3]['2min'],'c>--',
                 bx,outputs[3]['mean'],'yH-',
                 bx,outputs[3]['max'],'rH-',
                 bx,outputs[3]['min'],'bH-',markersize=6,)
        plt.setp(f3.set_xscale('log'))
        plt.setp(f3.set_title('vecvari10-vecvari1,sizz{}, mulb{}'.format(sizz,mulbv)))
        if ylog:
            plt.setp(f3.set_yscale('log'))
    else: #all in one plot
        fig=plt.figure(1, figsize=(10,10))
        #plt.yscale('log')
        #plt.xscale('log')
        plt.axis([0,max(bx),min(outputs[3]['min']),max([max(outputs[d]['max']) for d in outputs ])])
        #plt.Axes.set_yscale(ax,'log')
        #plt.Axes.set_xscale(ax,'log')
        fg=plt.plot(bx,outputs[1]['mean'],'gs-',bx,outputs[1]['max'],'bs-',bx,outputs[1]['min'],'rs-',
           bx,outputs[2]['mean'],'go-',bx,outputs[2]['max'],'bo-',bx,outputs[2]['min'],'ro-',    
           bx,outputs[3]['mean'],'yH-',bx,outputs[3]['max'],'mH-',bx,outputs[3]['min'],'cH-',markersize=6)
        #print(fg)
        #plt.setp(fg,plt.yscale('log'))
        #plt.setp(fg,plt.xscale('log'))
    for I in range(3):
        I+=1
        print(outputs[I]['mean'][0],outputs[I]['mean'][9*4],outputs[I]['mean'][-1])
        print(outputs[I]['max'][0],outputs[I]['max'][9*4],outputs[I]['max'][-1])
        print(outputs[I]['min'][0],outputs[I]['min'][9*4],outputs[I]['min'][-1])
    #plt.ion() 
    plt.legend(['BB:0,BS:0, mean','BB:0,BS:0, max','BB:0,BS:0, min',
                'BB:1,BS:0, mean','BB:1,BS:0, max','BB:1,BS:0, min',
                'BB:0,BS:1, mean','BB:0,BS:1, max','BB:0,BS:1, min'],markerfirst=True) 
    plt.show()
    
    
for i in (0,1):
    sizz=(i,i)
    for l in (0,1):
        mulbv=l
        plotvars(0)