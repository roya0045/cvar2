'''
Created on Dec 12, 2017

@author: ARL
'''
from varitest import *
import matplotlib as mlp
import matplotlib.pyplot as plt
from copy import deepcopy as cpy
bs=list()
bxx=[]

bseed=[0,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1,0.5,1,5,10,25,50,75,100,250,500,750,1000,5000]
for I in bseed:
    e=[I,I*2,I*3,I*4]
    for ii in e:
        bxx.append(ii)
    b=np.rollaxis(np.array([e for _ in range(shape[0])],dtype=np.float32),1,0)
    #b2=np.reshape(np.array([b for _ in range(shape[0])]),(outputs,shape[0]))
    bs.append(b)
#print(bxx)
axs=(0,-2,-1)
indict={'mean':[],'max':[],'min':[]}
outputs={1:cpy(indict),2:cpy(indict),3:cpy(indict)}
ding=0
print(len(bs),bs[0])
for bb in bs:
    p=vecvari1(inputcols, wid, B=bb,square=0,sqrt=0,sizer='v1',verbose=0)
    d=vecvari10(inputcols, wid, B=bb,square=0,sqrt=0,sizer='v10',verbose=0)
    for iou,out in enumerate([p,d,p-d]):
        #print([oo for oo in out.mean(axs)])
        if iou==0:
            ding+=1
        iou+=1
        #print(iou)
        #print(out.mean(axs).shape)
        [outputs[iou]['mean'].append(oo) for oo in out.mean(axs)]
        [outputs[iou]['max'].append(oo) for oo in out.max(axs)]
        [outputs[iou]['min'].append(oo) for oo in out.min(axs)]
bx=bxx#np.linspace(0, bxx[-1], num=outputs[1]['mean'].__len__(),  dtype=np.float32)
print(ding,(np.array(outputs[1]['mean']).shape,np.array(outputs[2]['mean'])))
#print([outputs[d]['min'] for d in outputs ])
tree=1
if tree: #tree plots
    fig, axs = plt.subplots(3, 1)
    #fig.set_yscale('log')
    plt.axis([0,max(bx),min(outputs[3]['min']),max([max(outputs[d]['max']) for d in outputs ])])
    f1=plt.subplot(311)
    plt.plot(bx,outputs[1]['mean'],'gs-',bx,outputs[1]['max'],'bs-',bx,outputs[1]['min'],'rs-',markersize=6)
    #plt.setp(f1.set_yscale('log'))
    f2=plt.subplot(312,sharex=f1)#,sharey=f1)
    plt.plot(bx,outputs[2]['mean'],'go-',bx,outputs[2]['max'],'bo-',bx,outputs[2]['min'],'ro-',markersize=6)
    #plt.setp(f2.set_yscale('log'))
    f3=plt.subplot(313,sharex=f1)#,sharey=f1)
    plt.plot(bx,outputs[3]['mean'],'yH-',bx,outputs[3]['max'],'mH-',bx,outputs[3]['min'],'cH-',markersize=6)
    #plt.setp(f3.set_yscale('log'))
else: #all in one plot
    fig=plt.figure(1, figsize=(10,10))
    #plt.yscale('log')
    plt.axis([0,max(bx),min(outputs[3]['min']),max([max(outputs[d]['max']) for d in outputs ])])
    #plt.Axes.set_yscale(ax,'log')
    fg=plt.plot(bx,outputs[1]['mean'],'gs-',bx,outputs[1]['max'],'bs-',bx,outputs[1]['min'],'rs-',
       bx,outputs[2]['mean'],'go-',bx,outputs[2]['max'],'bo-',bx,outputs[2]['min'],'ro-',    
       bx,outputs[3]['mean'],'yH-',bx,outputs[3]['max'],'mH-',bx,outputs[3]['min'],'cH-',markersize=6)
    #print(fg)
    #plt.setp(fg,plt.yscale('log'))
for I in range(3):
    I+=1
    print(outputs[I]['mean'][0],outputs[I]['mean'][9*4],outputs[I]['mean'][-1])
    print(outputs[I]['max'][0],outputs[I]['max'][9*4],outputs[I]['max'][-1])
    print(outputs[I]['min'][0],outputs[I]['min'][9*4],outputs[I]['min'][-1])
#plt.ion()  
plt.show()