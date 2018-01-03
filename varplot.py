
#check for implementing weight variance and the effect of variables on the output
from varitest import *
import matplotlib as mlp
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as cpy
bs=list()
bxx=[]
WAR=[]
scaleW=0
numout=5
bseed=[0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,25,50,75,100,250,500,750,1000,5000]
for B in bseed:
    bxx=[]
    if scaleW:
        e=[1,2,3,4,5]
        custW=np.broadcast_to(np.array([B],dtype=np.float32), shape=(5,3,3,3))
        ii=custW*np.reshape(np.array(e),(5,1,1,1))
        WAR.append(ii)
        #b=np.rollaxis(np.array([e for _ in range(shape[0])],dtype=np.float32),1,0)
        #WAR.append(b)
    else:
        tmp=np.array(B,dtype=np.float32)
        custW=np.broadcast_to(tmp, shape=(5,3,3,3))
        WAR.append(custW)
bx=[]
for I in bseed:
    e=np.reshape(np.array([1,2,3,4,5]),(5,1,1,1))
    bax=np.broadcast_to(np.array([I]),(1,1,1,1))
    ba1=np.broadcast_to(np.array([I]),(1,3,1,1))
    bs.append(ba1*e)
    [bx.append(O) for O in np.reshape(bax*e,(5,)).tolist()]
#bxa=np.array(bx)
#print(WAR[0].shape,WAR[0])
#print(bs[0].shape,bs[0])
#print(bx)

#exit()
#print(bxx)
isodata=0#100
if isodata:
    inputcols=np.broadcast_to(np.array([isodata],dtype=np.float32),inputcols.shape)
indict={'mean':[],'max':[],'min':[],'1mean':[],'1max':[],'1min':[],'2mean':[],'2max':[],'2min':[],}
outputs={1:cpy(indict),2:cpy(indict),3:cpy(indict)}
sizz=(0,0)
wid=np.broadcast_to(wid, (numout,*wid.shape[1:]))
sqrtv=0
mulbv=0
mul2v=1#
v3v=0
bsvi=-1#9
print(len(bs),bs[0])
def plotvars(BB,short=False,Nob=False,bx=bx):
    outputs={1:cpy(indict),2:cpy(indict),3:cpy(indict)}
    axs=(0,-2,-1)
    ding=0
    BD="Weight -" if BB==1 else "Bias -"
    hol=[WAR,bs][BB]
    if short:#subsample data
        for bb in hol:#bb are upscaled 5 times
            #print(bb)
            #print('B',bs[bsvi],'B')
            if BB==1:
                print('W',wid,'W')
                print('B',bb,'B')
                #print(wid.shape,bb.shape)
                d1=vecvari10(inputcols, wid,BB=0, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                d2=vecvari10(inputcols, wid,BB=1, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                d3=vecvari10(inputcols, wid,BB=0,BS=1, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)#-vecvari10(inputcols, wid,noB=0, B=bb,square=0,sqrt=0,verbose=0)
                p1=vecvari1(inputcols, wid, B=bb,BB=0,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                p2=vecvari1(inputcols, wid, B=bb,BB=1,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                p3=vecvari1(inputcols, wid, B=bb,BB=0,BS=1,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
            elif BB==0:
                if Nob:
                    bval=None
                else:
                    bval=bs[bsvi]
                print('W',bb,'W')
                print('B',bval,'B')
                print(bb.shape,bs[9].shape)
                d1=vecvari10(inputcols, bb,BB=0, B=bval,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                p1=vecvari1(inputcols, bb,BB=1, B=bval,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                d2=vecvari10(inputcols, bb,BB=0,BS=1, B=bval,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                p2=vecvari1(inputcols, bb,BB=0, B=bval,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                d3=vecvari10(inputcols, bb,BB=1, B=bval,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                p3=vecvari1(inputcols, bb,BB=0,BS=1, B=bval,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
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
    else:#do each data point
        for kl in hol:
            for bb in kl:
                bb=np.broadcast_to(bb, kl.shape)
                #print(bb[:,0])
                #print('B',bs[bsvi],'B')
                if BB==1:
                    print('W',wid,'W')
                    print('B',bb,'B')
                    #print(wid.shape,bb.shape)
                    d1=vecvari10(inputcols, wid,BB=0, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                    d2=vecvari10(inputcols, wid,BB=1, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                    d3=vecvari10(inputcols, wid,BB=0,BS=1, B=bb,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)#-vecvari10(inputcols, wid,noB=0, B=bb,square=0,sqrt=0,verbose=0)
                    p1=vecvari1(inputcols, wid, B=bb,BB=0,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                    p2=vecvari1(inputcols, wid, B=bb,BB=1,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                    p3=vecvari1(inputcols, wid, B=bb,BB=0,BS=1,sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                elif BB==0:
                    if Nob:
                        bval=None
                    else:
                        bval=bs[bsvi]
                    print('W',bb,'W')
                    print('B',bval,'B')
                    print(bb.shape,bs[9].shape)
                    print(bb.shape,bs[9].shape)
                    d1=vecvari10(inputcols, bb,BB=0, B=bs[bsvi],sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                    p1=vecvari1(inputcols, bb,BB=1, B=bs[bsvi],sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                    d2=vecvari10(inputcols, bb,BB=0,BS=1, B=bs[bsvi],sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                    p2=vecvari1(inputcols, bb,BB=0, B=bs[bsvi],sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                    d3=vecvari10(inputcols, bb,BB=1, B=bs[bsvi],sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[0],v3=v3v)
                    p3=vecvari1(inputcols, bb,BB=0,BS=1, B=bs[bsvi],sqrt=sqrtv,verbose=0,mulb=mulbv,mul2=mul2v,sizz=sizz[1],v3=v3v)
                for iou,out in enumerate([(d1,d2,d3),(p1,p2,p3),(d1-p1,d2-p2,d3-p3)]):
                    if iou==0:
                        ding+=1
                    iou+=1
                    #print(iou)
                    #print(out.mean(axs).shape)
                    out1=out[0]
                    out2=out[1]
                    out3=out[2]#out1-out2
                    outputs[iou]['1mean'].append(out1.mean()) 
                    outputs[iou]['1max'].append(out1.max())  
                    outputs[iou]['1min'].append(out1.min())
                    outputs[iou]['2mean'].append(out2.mean())
                    outputs[iou]['2max'].append(out2.max()) 
                    outputs[iou]['2min'].append(out2.min()) 
                    outputs[iou]['mean'].append(out3.mean())
                    outputs[iou]['max'].append(out3.max())
                    outputs[iou]['min'].append(out3.min())
    #bx=bxx#np.linspace(0, bxx[-1], num=outputs[1]['mean'].__len__(),  dtype=np.float32)
    print(ding,(np.array(outputs[1]['mean']),np.array(outputs[2]['mean'])))
    #print([outputs[d]['min'] for d in outputs ])
    tree=1
    ylog=[1,1,0]
    coloros=["xkcd:racing green","xkcd:bright pink","xkcd:raw umber",
             "xkcd:bright orange", "xkcd:barney purple","xkcd:light green",
             "xkcd:piss yellow","xkcd:bright aqua","xkcd:fire engine red",]
    markers=['+--','*--','x-']
    mks=[8,6,6]
    ALPH=0.58
    if tree: #tree plots
        fig, axs = plt.subplots(3, 1)
        #fig.set_xscale('log')
        plt.axis([0,np.amax(bs[-1]),min(outputs[3]['min']),max([max(outputs[d]['max']) for d in outputs ])])
        f1=plt.subplot(311)
        f1y=outputs[1]
        plt.plot(bx,f1y['1mean'],markers[0],color=coloros[0],markersize=mks[0],alpha=ALPH,)
        plt.plot(bx,f1y['1max'],markers[0],color=coloros[1],markersize=mks[0],alpha=ALPH,)
        plt.plot(bx,f1y['1min'],markers[0],color=coloros[2],markersize=mks[0],alpha=ALPH,)
        plt.plot(bx,f1y['2mean'],markers[1],color=coloros[3],markersize=mks[1],alpha=ALPH,)
        plt.plot(bx,f1y['2max'],markers[1],color=coloros[4],markersize=mks[1],alpha=ALPH,)
        plt.plot(bx,f1y['2min'],markers[1],color=coloros[5],markersize=mks[1],alpha=ALPH,)
        plt.plot(bx,f1y['mean'],markers[2],color=coloros[6],markersize=mks[2],alpha=ALPH,)
        plt.plot(bx,f1y['max'],markers[2],color=coloros[7],markersize=mks[2],alpha=ALPH,)
        plt.plot(bx,f1y['min'],markers[2],color=coloros[8],markersize=mks[2],alpha=ALPH,)
        plt.setp(f1.set_xscale('log'))
        plt.setp(f1.set_title('{},vecvari10,V3:{},sizz:{},mulb:{},mul2v:{}'.format(BD,v3v,sizz,mulbv,mul2v)))
        if ylog[0]:
            plt.setp(f1.set_yscale('log'))
        f2=plt.subplot(312,sharex=f1)#,sharey=f1)
        f2y=outputs[2]
        plt.plot(bx,f2y['1mean'],markers[0],color=coloros[0],markersize=mks[0],alpha=ALPH,)
        plt.plot( bx,f2y['1max'],markers[0],color=coloros[1],markersize=mks[0],alpha=ALPH,)
        plt.plot( bx,f2y['1min'],markers[0],color=coloros[2],markersize=mks[0],alpha=ALPH,)
        plt.plot(bx,f2y['2mean'],markers[1],color=coloros[3],markersize=mks[1],alpha=ALPH,)
        plt.plot(bx,f2y['2max'],markers[1],color=coloros[4],markersize=mks[1],alpha=ALPH,)
        plt.plot( bx,f2y['2min'],markers[1],color=coloros[5],markersize=mks[1],alpha=ALPH,)
        plt.plot(bx,f2y['mean'],markers[2],color=coloros[6],markersize=mks[2],alpha=ALPH,)
        plt.plot(bx,f2y['max'],markers[2],color=coloros[7],markersize=mks[2],alpha=ALPH,)
        plt.plot(bx,f2y['min'],markers[2],color=coloros[8],markersize=mks[2],alpha=ALPH,)
        plt.setp(f2.set_xscale('log'))
        plt.setp(f2.set_title('{},vecvari1,V3:{},sizz:{},mulb:{},mul2v:{}'.format(BD,v3v,sizz,mulbv,mul2v)))
        if ylog[1]:
            plt.setp(f2.set_yscale('log'))
        f3=plt.subplot(313,sharex=f1)#,sharey=f1)
        f3y=outputs[3]
        plt.plot(bx,f3y['1mean'],markers[0],color=coloros[0],markersize=mks[0],alpha=ALPH,)
        plt.plot( bx,f3y['1max'],markers[0],color=coloros[1],markersize=mks[0],alpha=ALPH,)
        plt.plot( bx,f3y['1min'],markers[0],color=coloros[2],markersize=mks[0],alpha=ALPH,)
        plt.plot(bx,f3y['2mean'],markers[1],color=coloros[3],markersize=mks[1],alpha=ALPH,)
        plt.plot(bx,f3y['2max'],markers[1],color=coloros[4],markersize=mks[1],alpha=ALPH,)
        plt.plot( bx,f3y['2min'],markers[1],color=coloros[5],markersize=mks[1],alpha=ALPH,)
        plt.plot(bx,f3y['mean'],markers[2],color=coloros[6],markersize=mks[2],alpha=ALPH,)
        plt.plot(bx,f3y['max'],markers[2],color=coloros[7],markersize=mks[2],alpha=ALPH,)
        plt.plot(bx,f3y['min'],markers[2],color=coloros[8],markersize=mks[2],alpha=ALPH,)
        plt.setp(f3.set_xscale('log'))
        plt.setp(f3.set_title('{},vecvari10-vecvari1,V3:{},sizz:{},mulb:{},mul2v:{}'.format(BD,sv3v,sizz,mulbv,mul2v)))
        if ylog[2]:
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
    plt.tight_layout(pad=1, h_pad=0.35, w_pad=0.35,rect=(0,0.07,1,1))
    plt.legend(['BB:0,BS:0, mean','BB:0,BS:0, max','BB:0,BS:0, min',
                'BB:1,BS:0, mean','BB:1,BS:0, max','BB:1,BS:0, min',
                'BB:0,BS:1, mean','BB:0,BS:1, max','BB:0,BS:1, min'],markerfirst=True,
               bbox_to_anchor=(0,0,1,0), loc=2,
               ncol=3, mode="expand", borderaxespad=0.5) 
    #https://matplotlib.org/users/legend_guide.html
    plt.show()
plots=1
V3=0
No_B=0
shortrun=1
BBB=1 # 1 = test B, 0 = test W
if plots:
    for mu2 in (0,1):
        mul2v=mu2
        for i in (0,1):
            sizz=(i,i)
            for l in (0,1):
                mulbv=l
                if V3:
                    for V in [0,1,2,3]:
                        v3v=V
                        plotvars(BBB,short=shortrun,Nob=No_B)
                else:    
                    plotvars(BBB,short=shortrun,Nob=No_B)
bdx=-1
wdx=0
mbb=1
alg=[vecvari10,vecvari1]
multprint=1
sidechec=True
"""tocheck:
W:
sizz0 mulb0 W[0]/W[-1] bb0,bs0
sizz1 mulb0,w[0]/w[-1] bb0bs1/bb1bs0
sizz1 mulb1,w[0]/w[-1]bb1bs0

B;
mulb soften sizz0
sizz1,mulb0,b[0],b[-1]
vec10 est + uniform sauf pout bb1,
vec1 est similaire et + disperse"""

if No_B:
    bval=None
else:
    bval=bs[bdx]
    print(bs[bdx])
if multprint:
    for STPG in [(0,0),(1,0),(1,1)]:#sizz,mulb value pair
        for STP in [(0,0),(1,0),(1,0)]:#bb,bs value pair
            for X in [1,-2]:#W/b index
                if sidechec:
                    for stp in [-1,0]:#side step
                        x2=X+X%2+stp
                        if BBB==0:#test weight
                            alg[0](inputcols, WAR[x2],BB=STP[0],BS=STP[1], B=bval,sqrt=sqrtv,verbose=2,mulb=STPG[1],sizz=STPG[0],mul2=mul2v,)
                        elif BBB==1:
                            alg[0](inputcols, wid,BB=STP[0],BS=STP[1], B=bs[x2],sqrt=sqrtv,mul2=mul2v,verbose=2,mulb=STPG[1],sizz=STPG[0])
                if BBB==0:
                    alg[0](inputcols, WAR[X],BB=STP[0],BS=STP[1], B=bval,sqrt=sqrtv,verbose=2,mulb=STPG[1],sizz=STPG[0],mul2=mul2v,)
                elif BBB==1:
                    alg[0](inputcols, wid,BB=STP[0],BS=STP[1], B=bs[X],sqrt=sqrtv,mul2=mul2v,verbose=2,mulb=STPG[1],sizz=STPG[0])
else:              
    if BBB==0:
        print("b",bs[bdx],"b")
        vecvari10(inputcols, WAR[wdx],BB=1,BS=0, B=bval,sqrt=sqrtv,mul2=mul2v,verbose=2,mulb=mbb,sizz=1)
        vecvari1(inputcols, WAR[wdx],BB=1,BS=0, B=bval,sqrt=sqrtv,mul2=mul2v,verbose=2,mulb=mbb,sizz=1)
    elif BBB==1:
        vecvari10(inputcols, wid,BB=1,BS=0, B=bs[bdx],sqrt=sqrtv,mul2=mul2v,verbose=2,mulb=mbb,sizz=1)
        vecvari1(inputcols, wid,BB=1,BS=0, B=bs[bdx],sqrt=sqrtv,mul2=mul2v,verbose=2,mulb=mbb,sizz=1)