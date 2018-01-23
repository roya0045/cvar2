'''
Created on Oct 24, 2017

@author: ARL
'''
import numpy as np
import pandas as pd
from math import ceil,floor
from keras.utils import to_categorical

def caster(array,typ=None,*args,**kwargs):
    return(array.astype(typ))
def kerasdatasets(name:int,cat2vec=False,float32=False):#a way to get keras dataset
    import keras.datasets as KD
    dss=[KD.boston_housing,KD.cifar10,KD.cifar100,KD.mnist,KD.fashion_mnist,KD.boston_housing,KD.reuters,KD.imdb,KD.cifar]
    (x_train, y_train), (x_test, y_test) = dss[name].load_data()
    if name in [1,2,3,4]:
        if name in [1,3,4]:
            num=10
        elif name ==2:
            num=100
        elif name==6:
            num=46
        if cat2vec:
            y_train=to_categorical(y_train, num)
            y_test=to_categorical(y_test, num)
        if float32:
            x_test=np.array((x_test/225.0),dtype=np.float32)
            x_train=np.array((x_train/225.0),dtype=np.float32)
    return((x_train, y_train), (x_test, y_test))

def convshape(x,k,s=1,p=0,d=1):
    #taken from chaine to get the outputdim given thee params of that dim
    """
    x=original size
    k=kernel size
    s= stride lenght
    p = pad size
    d= dilation
    """
    if len(x)==1:
        size=(x+2*p-d*(k-1)-1)//s+1
    else:
        size=((x[0]+2*p-d*(k[0]-1)-1)//s+1,(x[1]+2*p-d*(k[1]-1)-1)//s+1)
    return(size)

def DATASHAP1(dataa):#see extend dimension
    dataa=np.reshape(np.array(dataa),dataa.shape+(1,))
    return(dataa)

def arger(function,kwargs:dict):#filters out useless keyword argument
    "check if a keyword is in the function and return those that are present/remove useless keywords"
    import inspect
    fargspec=inspect.signature(function)
    FKEY=fargspec.parameters.keys()
    out={ ky: va  for(ky , va) in kwargs.items() if ky in FKEY}
    return(out)

def categorical_emb(inp):
    for cat in inp:
        if isinstance(inp[cat][0],str):
            uniq=list(set(inp[cat]))
            x=0
            #print(uniq)
            for word in uniq:
                inp[cat].replace(to_replace=word,value=uniq.index(word),inplace=True)
    return inp

def layer_interp(nlayers:int=10,inpu:list=[], method:int=0):#light method
    """
    nlayer=number of point
    inpu: list of tuples to establish the shape, in the form of [(1x,1y)...(nx,ny)]
    method: interpolation method
    """
    import scipy.interpolate as si
    methos=[si.interp1d, 
        si.pchip_interpolate,
        si.Akima1DInterpolator,
        si.KroghInterpolator,
        si.CubicSpline, 
        si.make_interp_spline,
        si.UnivariateSpline,
        si.InterpolatedUnivariateSpline]
    smeth = [si.make_interp_spline,
        si.UnivariateSpline,
        si.InterpolatedUnivariateSpline] #spline interpolation (3)
    x=[]
    y=[]
    methodd=methos[method % methos.__len__()]
    try:
        if inpu[0].__len__()==2:
            for xy in inpu:
                x.append(xy[0])
                y.append(xy[1])
            x2=np.linspace(0,x[-1],100)#max x
    except:#set up to 100
        itr=inpu.__len__()-2
        x.append(0)
        [x.append(100/(itr+1)*(i+1)) for i in range(itr)]
        x.append(100) 
        y=inpu
        x2=np.linspace(0,100,100)#layers
    if method==0:
        try:
            interpo=methodd(x,y,kind='quadratic')
        except:
            interpo=methodd(x,y,kind='slinear')
        y2=interpo(x2)
    elif method==1 or method==2 or method==3:
        y2=methodd(x,y,x2) 
    elif method in smeth:
        interp=methodd(x,y,k=(ceil(len(inpu)*1.5)%4)+1)
        y2=interp(x2)
    else:
        interpo=methodd(x,y)
        y2=interpo(x2)
    if nlayers==100:
        y3=y2
    else:
        x3=[]
        [x3.append(100/(nlayers-1)*(i+1)) for i in range(nlayers-2)]
        y3=[]
        y3.append(y[0])
        for xx in x3:
            y3.append(y2[ceil(xx)])
        y3.append(y[-1])
    return(y3)



def doubler(data1,window=1,step=1):#windows should be even
    if window%2>0:
        print("this will probably duplicate existing data, better use an even number for the window")
        exit()
    if isinstance(data1, (float,str,int)):
        data=pd.read_csv(data1)#tochange/adapt
    elif isinstance(data1, pd.DataFrame):
        print("DATAFRAM")
        data=data1
    else:
        print("bad data type")
        exit()
    LL=len(data)
    x=int(window)
    NL=list()
    while x<(LL-window):
        NL1=(data.iloc[(x-window):(x+window),:].mean(axis=0))
        NL1['day']=data['day'].iloc[(x-window)]+((data['day'].iloc[1:3].mean()-data['day'].iloc[1]))
        into=NL1.to_frame().T
        NL.append(into)
        x+=step
    NL2=pd.concat(NL,axis=0,join='outer',ignore_index=True)
    list0=[data,NL2]
    data=pd.concat(list0, axis=0, join='outer', verify_integrity=True,ignore_index=True)
    data.set_index(data['date'],drop=False,inplace=True)
    data.sort_values(['date'],axis=0,inplace=True)
    data.fillna(0)
    data.replace('NaN', value=0, inplace=True)
    data.fillna(0)
    data.replace('NaN', value=0, inplace=True)
    return(data)
    
def normalize(data,cols=None,EXP=False):#exp is another (safer?) way to handle data
    from natsort import natsorted
    if cols is None:
        data_norm=(data-data.mean())/(data.max()-data.min())
    elif EXP:
        if not(isinstance(cols, (list,tuple))):
            tmp=cols
            cols=list()
            cols.append(tmp)
        if all(isinstance(v, (float,int)) for v in cols):
            tmp=data.columns
            col2=natsorted(cols,reverse=True)
            [tmp.pop(X) for X in col2]
            cols=tmp
            data[data.columns[list(cols)]] = data[data.columns[list(cols)]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            data_norm=data
        elif all(isinstance(v, str) for v in cols):
            input("ensure that no columns have the same name otherwise the wrong column may be deleted, press enter to continu")
            tmp=data.columns
            [tmp.remove(X) for X in cols]#will remove the first element with the given name, horrible, may delete the wrong columns if no surname are present
            cols=tmp
            data[cols] = data[list(cols)].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            data_norm=data
    else:
        if not(isinstance(cols, (list,tuple))):
            tmp=cols
            cols=list()
            cols.append(tmp)
        if all(isinstance(v, (float,int)) for v in cols):
            data[data.columns[list(cols)]] = data[data.columns[list(cols)]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            data_norm=data
        elif all(isinstance(v, str) for v in cols):
            data[cols] = data[list(cols)].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
            data_norm=data
        else:
            data_norm=(data-data.mean())/(data.max()-data.min())
    data_norm.dropna(axis=1,thresh=(len(data_norm)*0.75))
    data_norm.replace('NaN', value=0, inplace=True)
    data_norm.fillna(0)
    return(data_norm)
try:
    import imgaug as IAG #https://imgaug.readthedocs.io/en/latest/source/augmenters.html
    import imgaug.augmenters as IAGA
    class augmenter():#https://wiki.python.org/moin/Generators
        #use imgaug lib as backend
        #https://imgaug.readthedocs.io/en/latest/source/augmenters.html
        #https://imgaug.readthedocs.io/en/latest/source/alpha.html
        def __init__(self,inputdata,inputlabels,augs="basic",#["all","basic","form","valalt","pxlalt","imgalt"]
                     num_outs=5,og_out=True, mode='G',em=0,intensity=1.0,rescaledata=None,formatd='NCHW',
                     min_augs=0,max_augs=5):
            if self.mode.lower()=='g':
                self.NM=self.rung
            elif self.mode.lower()=='i':
                self.NM=self.runi()
            elif self.mode.lower()=='i2':
                self.NM=self.runi2()
            else:
                print("invalid mode, use 'g' for generator or 'i' for iterator or 'i2'")
                exit()
            self.minaug=min_augs
            self.maxaug=max_augs
            #self.affineopt=["scale","translate_percent","translate_px","rotate","shear"]
            #self.chnlopt=[{"per_channel":True},{"per_channel":False}]
            if len(inputdata.shape)==4:
                self.D=4
            elif len(inputdata.shape)==3:
                self.D=3
            elif len(inputdata.shape)==2:
                self.D=2
            if formatd=="NCHW":
                if self.D==4:
                    self.inputd=np.transpose(inputdata, [0,2,3,1])
                elif self.D==3:
                    self.inputd=np.transpose(inputdata,[1,2,0])
            else:
                self.inputd=inputdata
            self.Y=inputlabels
            leninten=8
            if isinstance(intensity, (float,int)):
                itensity=[intensity for _ in range(leninten)]
            else:
                assert len(intensity)==leninten
            self.datashape=np.array(inputdata.shape)#inputdata[0].shape
            if self.datashape.min()==self.datashape[-1]:
                self.pixls=self.datashape[:-1] 
            elif self.datashape.min()==self.datashape[1]:
                self.pixls=np.delete(self.datashape,1)
            elif self.datashape.shape==(3,):
                self.pixls=self.datashape[1:]
            else:
                print("error cannot fin the shape of images")
                exit()
            # can use "keep-aspect-ratio" for an arg to have a relative and absolute scale
            #or can also use list for randomization between options
            self.scalevals=(0.5/(2*intensity),1.0)#use % of image
            self.augs=augs
            self.Pchances=0.44*itensity[0]
            self.intrange=((ceil(10*intensity[1]),ceil(10+140*itensity[1])))
            self.windowrange=(ceil(2*intensity[2]),ceil((min(self.pixls)/5)-8)*intensity[2])#mean/median things
            self.relatrange=(0.1*intensity[3],0.95*intensity[3])#normalisation,invert
            self.bigfloat=(0.085*intensity[4],1.75*intensity[4])#some scale values,multiply,contrastnorm,elasti trans,(sigman&alpha)
            self.smallfloat=(0.001*intensity[5],0.45*intensity[5])#coarse dropout/droput(p)
            self.addrange=(ceil(-140*intensity[6]),ceil(140*intensity[6]))
            self.multrange=(-2.0*intensity[7],2.0*intensity[7])
            self.perchannelsplit=0.75*intensity[8] #used for per_channel on the mult
            self.allaugs={"add":IAGA.Add(value=self.addrange,per_channel=0.75*intensity),
                        "scale":IAGA.Scale(size=self.scalevals),
                        "adde":IAGA.AddElementwise(value=self.addrange,per_channel=0.75*intensity),
                        "addg":IAGA.AdditiveGaussianNoise(scale=(0,self.smallfloat[1]*255),per_channel=0.75*intensity),
                        "addh":IAGA.AddToHueAndSaturation(value=self.addrange,per_channel=0.75*intensity),
                        "mult":IAGA.Multiply(mul=self.bigfloat,per_channel=0.75*intensity),
                        "mule":IAGA.MultiplyElementwise(mul=self.bigfloat,per_channel=0.75*intensity),
                        "drop":IAGA.Dropout(p=self.smallfloat,per_channel=0.75*intensity),
                        "cdrop":IAGA.CoarseDropout(p=self.smallfloat, size_px=None, size_percent=self.smallfloat,
                                        per_channel=True, min_size=3),
                        "inv":IAGA.Invert(p=self.Pchances,per_channel=0.75*intensity,min_value=-255,max_value=255),
                        "cont":IAGA.ContrastNormalization(alpha=self.bigfloat,per_channel=0.75*intensity),
                        "aff":IAGA.Affine(scale=self.bigfloat,
                                    translate_percent={'x':(-40*intensity,40*intensity),'y':(-40*intensity,40*intensity)}, translate_px=None,#moving functions
                                    rotate=(-360*intensity,360*intensity), shear=(-360*intensity,360*intensity),
                                    order=[0,1]#2,3,4,5 may be too much
                                    , cval=0,#for filling
                                    mode=["constant","edge","reflect","symmetric","wrap"][em],#filling method
                                    deterministic=False,
                                    random_state=None),
                        "paff":IAGA.PiecewiseAffine(scale=(-0.075*intensity,0.075*intensity), nb_rows=(ceil(2*intensity),ceil(7*intensity)), nb_cols=(ceil(2*intensity),ceil(7*intensity)),
                                              order=[0,1], cval=0, mode=["constant","edge","reflect","symmetric","wrap"][em],
                                              deterministic=False, random_state=None),
                        "elas":IAGA.ElasticTransformation(alpha=self.bigfloat,sigma=self.relatrange),
                        "noop":IAGA.Noop(name="nope"),
                        #IAGA.Lambda:{},
                        "cropad":IAGA.CropAndPad(px=None, percent=(-0.65*intensity[7],0.65*intensity[7]),pad_mode=["constant","edge","reflect","symmetric","wrap"][em], pad_cval=0, keep_size=True, sample_independently=True,),
                        "fliplr":IAGA.Fliplr(p=self.Pchances),
                        "flipud":IAGA.Flipud(p=self.Pchances),
                        "spixel":IAGA.Superpixels(p_replace=self.Pchances, n_segments=self.intrange),
                        #IAGA.ChangeColorspace:,
                        "gray":IAGA.Grayscale(alpha=self.relatrange),
                        "gblur":IAGA.GaussianBlur(sigma=self.bigfloat),
                        "ablur":IAGA.AverageBlur(k=self.windowrange),
                        "mblur":IAGA.MedianBlur(k=self.windowrange),
                        #IAGA.BilateralBlur,
                        #IAGA.Convolve:,
                        "sharp":IAGA.Sharpen(alpha=self.relatrange,lightness=self.bigfloat),
                        "embo":IAGA.Emboss(alpha=self.relatrange,strenght=self.bigfloat),
                        "edge":IAGA.EdgeDetect(alpha=self.relatrange),
                        "dedge":IAGA.DirectedEdgeDetect(alpha=self.bigfloat,direction=(-1.0*intensity,1.0*intensity)),
                        "pert":IAGA.PerspectiveTransform(scale=self.smallfloat),
                        "salt":IAGA.Salt(p=self.Pchances,per_channel=0.75*intensity),
                        #IAGA.CoarseSalt(p=, size_px=None, size_percent=None,per_channel=False, min_size=4),
                        #IAGA.CoarsePepper(p=, size_px=None, size_percent=None,"per_channel=False, min_size=4),
                        #IAGA.CoarseSaltAndPepper(p=, size_px=None, size_percent=None,per_channel=False, min_size=4),
                        "pep":IAGA.Pepper(p=self.Pchances,per_channel=0.75*intensity),
                        "salpep":IAGA.SaltAndPepper(p=self.Pchances,per_channel=0.75*intensity),
                        #"alph":IAGA.Alpha(factor=,first=,second=,per_channel=0.75*intensity,),
                        #"aplhe":IAGA.AlphaElementwise(factor=,first=,second=,per_channel=0.75*intensity,),
                        #IAGA.FrequencyNoiseAlpha(exponent=(-4, 4),first=None, second=None, per_channel=False,size_px_max=(4, 16), upscale_method=None,iterations=(1, 3), aggregation_method=["avg", "max"],sigmoid=0.5, sigmoid_thresh=None,),
                        #IAGA.SimplexNoiseAlpha(first=None, second=None, per_channel=False,size_px_max=(2, 16), upscale_method=None,iterations=(1, 3), aggregation_method="max",sigmoid=True, sigmoid_thresh=None,),
                        }
            ["all","basic","form","valalt","pxlalt","imgalt"]
            self.augs=[]
            if (augs=="all")or ("all" in augs):
                self.augs=["add","scale","adde","addg","addh","mult", "mule","drop",
                        "cdrop","inv","cont","aff","paff","elas","noop","cropad","fliplr",
                        "flipud","spixel","gray","gblur","ablur","mblur","sharp","embo",
                        "edge","dedge","pert","salt","pep","salpep",]#"alph", "aplhe",]
            else:
                if (augs=="basic") or ("basic"in augs):
                    self.augs.append(["add","scale","addh","mult","drop","cont","noop"])
                if (augs=="form") or ("form"in augs):
                    self.augs+["scale","aff","paff","elas","noop","pert"]
                if (augs=="valalt") or ("valalt"in augs):
                    self.augs+["mult","mule","inv","fliplr","flipud","cropad","noop"]
                if (augs=="pxlalt") or ("pxlalt"in augs):
                    self.augs+["addg","drop","salt","pep","salpep","noop"]
                if (augs=="imgalt") or ("imgalt"in augs):
                    self.augs+["elas","noop","spixel","gblur","ablur","mblur","sharp","embo",
                            "edge","dedge",]
                if len(augs)==0:
                    self.augs+["add","scale","addh","drop","cont","aff","elas",
                               "noop","cropad","gray","ablur","sharp","salpep",]
            self.AUG=IAGA.SomeOf((self.minaug,self.maxaug),self.augs,
                            random_order=True)
            """self.affineopts={"scale":self.biglfoat,
                              "translate_percent":{'x':(-40*intensity,40*intensity),'y':(-40*intensity,40*intensity)}, "translate_px":None,#moving functions
                     "rotate":(-360*intensity,360*intensity), "shear":(0*intensity,360*intensity),
                      "order":[0,1]#2,3,4,5 may be too much
                     , "cval":0,#for filling
                      "mode":"constant",#filling method
                      "deterministic":False,
                       "random_state":None}
            self.pieceaffinev={"scale"=(-0.075*intensity,0.075*intensity), "nb_rows"=(ceil(2*intensity),ceil(7*intensity)), "nb_cols"=(ceil(2*intensity),ceil(7*intensity)),
                                "order"=[0,1], "cval"=0, "mode"="constant",
                      "deterministic"=False, "random_state"=None}"""
            self.num_outs=num_outs-og_out
            self.og_out=og_out
            self.mode=mode
            self.iimg=-1
            self.iout=0
            try:
                self.len=inputdata.shape[0]
            except:
                self.len=len(inputdata)
    
            def __iter__(self):
                return self
            def __next__(self):
                return(self.NM())
            def next(self):
                return(self.NM())
            
            def runi(self):
                if self.iimg==self.len:
                    raise StopIteration
                self.iimg+=1
                img=self.inputd[self.iimg]
                y=self.Y[self.iimg]
                out=np.broadcast_to(img, (self.num_out,*img.shape[-3:]))
                out=self.AUG.augment_images(out[self.og_out:])
                if self.og_out:
                    if len(img.shape)==3:
                        out=np.concatenate(out,np.expand_dims(img, 0))
                    else:
                        out=np.concatenate(out,img)
                if self.format=="NCHW":
                    out=np.transpose(out, [0,3,1,2])
                return([(outi,y) for outi in out])
            
            def runi2(self):
                if self.iimg==self.len:
                    raise StopIteration
                if (self.iout==self.num_outs) or (self.iimg==-1):
                    self.iimg+=1
                    self.iout=0
                    img=self.inputd[self.iimg]
                    y=self.Y[self.iimg]
                    out=np.broadcast_to(img, (self.num_out,*img.shape[-3:]))
                    self.out=self.AUG.augment_images(out[self.og_out:])
                    if self.og_out:
                        if len(img.shape)==3:
                            self.out=np.concatenate(out,np.expand_dims(img, 0))
                        else:
                            self.out=np.concatenate(out,img)
                    if self.format=="NCHW":
                        self.out=np.transpose(out, [0,3,1,2])
                    outp=(self.out[self.iout],y)
                else:
                    self.iout+=1
                    outp=(self.out[self.iout],self.Y[self.iimg])
                return(outp)
            
            def rung(self):
                for ix,img in enumerate(self.inputd):
                    out=np.broadcast_to(img, (self.num_out,img.shape[-3:]))
                    out=self.AUG.augment_images(out[self.og_out:])
                    y=self.Y[ix]
                    if self.og_out:
                        if len(img.shape)==3:
                            out=np.concatenate(out,np.expand_dims(img, 0))
                        else:
                            out=np.concatenate(out,img)
                    if self.format=="NCHW":
                                out=( np.transpose(out, [0,3,1,2]))
                    for sout in out:
                        yield (sout,y)
except:
    pass
            

def data_splitah(inputt,Y_c,inp_type='file',header=True,ynam=False,split=True,fract_test=0.15,shuffle=True,seed=21654,sepa=',',delim=None,NA=None,paandas=False):
    if inp_type=='file':
        with open(r'C:\Users\utilisateur\Desktop\tuto_py\data\{0}'.format(inputt),'r')as filee:
            if not(paandas):
                if header:
                    data1=np.genfromtxt(filee,delimiters=delim,dtype=None,names=True)
                else:
                    data1=np.genfromtxt(filee,delimiters=delim,dtype=None)
            else: 
                if header:
                    x1=pd.read_csv(filee,sep=sepa,delimiter=delim,na_values=NA,header='infer')
                else:   
                    x1=pd.read_csv(filee,sep=sepa,delimiter=delim,na_values=NA,header=None)#(header='infer', names=None, index_col=None, usecols=None, squeeze=False, prefix=None, mangle_dupe_cols=True, dtype=None, engine=None, converters=None, true_values=None, false_values=None, skipinitialspace=False, skiprows=None, nrows=None, na_values=None, keep_default_na=True, na_filter=True, verbose=False, skip_blank_lines=True, parse_dates=False, infer_datetime_format=False, keep_date_col=False, date_parser=None, dayfirst=False, iterator=False, chunksize=None, compression='infer', thousands=None, decimal=b'.', lineterminator=None, quotechar='"', quoting=0, escapechar=None, comment=None, encoding=None, dialect=None, tupleize_cols=False, error_bad_lines=True, warn_bad_lines=True, skipfooter=0, skip_footer=0, doublequote=True, delim_whitespace=False, as_recarray=False, compact_ints=False, use_unsigned=False, low_memory=True, buffer_lines=None, memory_map=False, float_precision=None)
    else:
        try:#check if panda data
            x1=inputt.copy()
        except:#catches all
            x1=np.copy(inputt)
    if paandas:
        if shuffle:
            np.random.seed(seed)
            x1.reindex(np.random.permutation(x1.index))
        if ynam:
            y=x1.pop(Y_c)
        else:
            y=x1.pop(x1.columns[[Y_c]])
        if split:
            x_train = (x1[:int(len(x1) * (1 - fract_test))]).as_matrix(columns=x1.columns[:])
            y_train = (y[:int(len(x1) * (1 - fract_test))]).as_matrix()
            x_test = (x1[int(len(x1) * (1 - fract_test)):]).as_matrix(columns=x1.columns[:])
            y_test = (y[int(len(x1) * (1 - fract_test)):]).as_matrix()
            return (x_train, y_train), (x_test, y_test)
        else:
            return (x1.as_matrix(columns=x1.columns[:]),y.as_matrix())
    else:
        if shuffle:
            np.random.seed(seed)
            np.random.shuffle(data1)
        arra=np.core.numeric.moveaxis(data1,Y_c,-1)
        y=arra[:,-1]
        x=np.delete(arra,(len(arra[0]-1)),axis=1)
        if split:
            x_train = np.array(x[:int(len(x) * (1 - fract_test))])
            y_train = np.array(y[:int(len(x) * (1 - fract_test))])
            x_test = np.array(x[int(len(x) * (1 - fract_test)):])
            y_test = np.array(y[int(len(x) * (1 - fract_test)):])
            return (x_train, y_train), (x_test, y_test)
        else:
            return (x,y)
    

"""
import keras.backend as K
import keras.callbacks as KC
class _setter:#attemp to make a loger, not much use, would be better to do 
    #a predict run on all the data to see what the final state would output
    
    def __init__(self,method,log=False,delay=10,filling_with=True,FOLD=".\\",name='test',restartwaitonepoch=False):
        self.name="LOG{}.csv".format(name)
        self.delay=delay
        self.FW=filling_with
        self.FOLD=FOLD
        self.log=log
        self.EPR=restartwaitonepoch
        self._resetx()
        if not(filling_with):
            
            self.file=open((FOLD+name),'a')
        if method=="MSE":#mean square error
            def metr(y_true,y_pred):
                error=K.square(y_pred - y_true)
                return error
        elif method=="MAE":#mean absolute error
            def metr(y_true,y_pred):
                error=K.abs(y_pred - y_true)
                return error
        elif method=="MAPE":#mean absolute percentage error
            def metr(y_true,y_pred):
                error=K.abs((y_true - y_pred) / K.clip(K.abs(y_true),
                                     K.epsilon(),None))
                return error
        elif method=="MSLE":# mean squared log error
            def metr(y_true,y_pred):
                first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
                second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
                error=K.square(first_log - second_log)
                return error
        elif method=="SH":#squared hinge
            def metr(y_true,y_pred):
                error=K.square(K.maximum(1. - y_true * y_pred, 0.))
                return error
        elif method=="H":#hinge
            def metr(y_true,y_pred):
                error=K.maximum(1. - y_true * y_pred, 0.)
                return error
        elif method=="LC":#logcosh
            def metr(y_true,y_pred):
                def cosh(x):
                    return (K.exp(x) + K.exp(-x)) / 2
                error=K.log(cosh(y_pred - y_true))
                return error
        elif method=="BC":#binary cross entropy
            def metr(y_true,y_pred):
                error=K.binary_crossentropy(y_true, y_pred)
                return error
        elif method=="P":#poisson
            def metr(y_true,y_pred):
                error=y_pred - y_true * K.log(y_pred + K.epsilon())
                return error
        elif method=="CP":#cosine proximity
            def metr(y_true,y_pred):
                y_true = K.l2_normalize(y_true, axis=-1)
                y_pred = K.l2_normalize(y_pred, axis=-1)
                error=y_true * y_pred
                return error
        elif method=="KLD":#kullback leiber divergence
            def metr(y_true,y_pred):
                y_true = K.clip(y_true, K.epsilon(), 1)
                y_pred = K.clip(y_pred, K.epsilon(), 1)
                error=(y_true * K.log(y_true / y_pred))
                return error
        elif method=="CACC":#categorical arrucary, may be horrible
            def metr(y_true,y_pred):
                error=K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1))
                return error
        elif method=="CSAC":#categorical sparse accuracy, maybe be horrible
            def metr(y_true,y_pred):
                error=K.equal(K.argmax(y_true, axis=-1),K.argmax(y_pred, axis=-1))
                return error
        elif method=="TOK":#top k
            def metr(y_true,y_pred,k=5):
                error=K.in_top_k(y_pred, K.argmax(y_true, axis=-1), k)
                return error
        elif method=="STK":#sparse top k
            def metr(y_true,y_pred,k=5):
                error=K.in_top_k(y_pred, K.cast(K.max(y_true, axis=-1), 'int32'), k)
                return error
        elif method=="CBIN":#binary accuracy
            def metr(yt,yp):
                print('hit')
                error=K.equal(yt, K.round(yp))
                K.print_tensor(error, message=None)
                return error
        else:
            def metr(y_true,y_pred):
                return y_pred
        self.metr=metr
        self.x=0
        
    def _incx(self):
        global IDEXEDI
        IDEXEDI+=1
        print(IDEXEDI)
    def _getx(self):
        self.x=IDEXEDI
    def _resetx(self):
        global IDEXEDI
        IDEXEDI = 0
    def _IDX(self):
        global IDEXEDI
        print(IDEXEDI)
    def M(self):
    
        if self.log:
            
            #methods https://github.com/fchollet/keras/blob/master/keras/losses.py
            if self.FW:
                def Collector(y_true,y_pred):
                    error=self.metr(y_true,y_pred)
                    self._IDX()
                    with open((self.FOLD+self.name),'a') as file:
                        file.write(np.array2string(np.array(error)))
                        if self.delay < IDEXEDI: #alt delay < tracker:#might not work
                            out=K.concatenate([y_pred,error],axis=0)
                            file.write(out)
                            
                        else:
                            self._incx()
                            
                    return(error)
                return Collector
            else:
                def Collector(y_true,y_pred):
                    x=0
                    error=self.metr(y_true,y_pred)
                    print(x)
                    self.file.write(np.array2string(np.array(error)))
                    if self.delay < IDEXEDI: #alt delay < tracker:#might not work      
                        out=K.concatenate([y_pred,error],axis=0)
                        print(out)
                        self.file.write(out) #https://github.com/fchollet/keras/blob/master/keras/engine/training.py
                        
                    else:
                        self._incx()
                        
                    return(error)
                file=self.file
                EPR=self.EPR
                class closer(KC.Callback):#tweaking closer may be the right way to write the output to a file via trigger
                    def __init__(self,file=file,epoch_restart=EPR):
                        #super(closer, self).__init__()
                        self.file=file
                        self.EPR=EPR   
                    def set_model(*args):
                        pass
                        #self.model="model"
                    def _resetx():
                        global IDEXEDI
                        IDEXEDI = 0
                    def set_params(*args):
                        pass
                    def on_epoch_start(*args):
                        if self.EPR:
                            self._resetx()
                    def on_train_end(*args):
                        print(self.file)
                        self.file.close()
                    def on_batch_begin(*args):
                        pass
                    def on_batch_end(*args):
                        pass
                return(Collector,closer)
            
        else:
            def passor(self,y_true,y_pred):
                pass
                #return( K.mean(metr(y_true,y_pred))
            return passor"""