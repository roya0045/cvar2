#make edge similar for autoencoding
#1 get a shape that is not in the dict of the proper size
#the shape must be composed of 2 or up to 4 values
#2 scale the values according to a distribution

#import tensorflow as tf
import numpy as np
from functools import reduce,partial
import operator
import numpy.random as nr



def edgemaker(shape,points,filtered=True,maxpoint=None,arrange=[0,2]):
    """
    shape:shape of the kernels
    points:minimal number of points in the kernel
    filtered: filter arrays or not, filtering =( removing arrays with values out of the cross and too little or too many values)
    maxpoint: maximum number of points above 0
    arrange:(list of ints) range of values to input data, default [0,2) (2 not included)
    """
    arrange.sort()
    if isinstance(shape, int):
        shape=[shape,shape]
    shapesum=reduce(operator.mul,shape)
    if maxpoint is None:
        maxpoint=shapesum-1
    #print(maxpoint)
    emptyar=np.zeros(shape)
    fullar=np.array([ 1 for _ in range(shapesum)]).reshape(shape)
    arrayss=[emptyar.reshape(shape),fullar]
    for _ in range((((shapesum)**2)-2)):
        INN=0
        stx=len(arrayss)
        x=np.copy(emptyar).reshape(shape)
        while not(INN):
            for row in range(shape[0]):
                for col in range(shape[1]):
                    x[row,col]=np.random.randint(arrange[0],high=arrange[-1]+1,size=1)
                    if x[row,col]<0:
                        x[row,col]=0
            if not(any([np.array_equiv(x, I) for I in arrayss])):
                arrayss.append(x)
                INN=1
    if not(filtered):
        return(arrayss)
    tt=[]
    cross=np.array([[ 1 if ((0<c<(shape[1]-1)) or (0<R<(shape[0]-1))) else 0 for c in range(shape[1]) ]for R in range(shape[0])])
    nm=reduce(operator.mul,arrayss[0].shape)
    for i in arrayss:
        ib=i.astype(bool)
        #print(i)
        ic=reduce(operator.add,ib*cross)#number of values in the cross
        while not(isinstance(ic, (float,int))):
            ic=reduce(operator.add,ic.tolist())
        iv=reduce(operator.add,ib)#number of values in the cross
        while not(isinstance(iv, (float,int))):
            iv=reduce(operator.add,iv.tolist())
        #print("values in cross",ic,"values total",iv)
        if (iv>=points and iv<=maxpoint and iv-ic>=(points//2)):
            try:
                if not(i in tt):
                    tt.append(i)
            except:
                tt.append(i)
    #print(cross)
    #print(len(tt))
    return(tt)


def scaler(data,ranges,scale=1,dist=None,seed=None,oper=0,center=None):
    """
    data=data
    ranges=list of [min, (center (optionnal)), max] value
    scale 0: add single value to each kernel,
    scale 1: add the same random kernel to each kernel
    scale2: a different random kernel to each kernel
    dist:int for distribution
    seed: seed for random gen
    oper: 0= add
    oper 1= mult
    oper 2= (data+0.1)*random elements
    center: central distribution value of the distribution to skew the data
    alternative to the 3 element list
    """
    scale=scale%3
    ld=len(data)
    assert all( isinstance(k,(float,int)) for k in ranges)
    ranges.sort()
    if len(ranges)==3 and center is None:
        center=ranges[1]
    elif center is None:
        center=(ranges[0]+ranges[-1])/2.0
    dsh=data[0].shape
    totar=reduce(operator.mul,dsh)
    def multef(in1,in2):
        return((in1+0.1)*in2)
    operat=[np.add,np.multiply,multef][oper]
    if seed is None:
        seed=nr.random(1)*nr.random_integers(-4,high=4)
    nr.seed(seed)
    distribs=[partial(nr.triangular,*(ranges[0],center,ranges[-1])),#left, mode,rightpartial(nr.beta,(2,3))#a,b
              partial(nr.uniform,*(None,),**{"low":ranges[0],'high':ranges[-1]}),
              partial(nr.chisquare,*(5,))#df
              ,partial(nr.dirichlet,*(dsh,))#alpha, alpha can be the shape
              ,nr.exponential,
              partial(nr.f,*(4,3))#dfnum,dfden
              ,partial(nr.gamma,*(3.0,)),#shape
              nr.gumbel,
              nr.laplace,
              nr.logistic,
              partial(nr.logseries,*(0.6,)),#p
              nr.lognormal,
              #partial(nr.multivariate_normal,)#mean,cov
              partial(nr.noncentral_chisquare,*(3,1))#df nonc
              ,partial(nr.noncentral_f,*(4,3,3.0)),#dfnum,dfen nonc
              nr.normal,
              partial(nr.pareto,*(dsh,)),#a
              nr.poisson,
              nr.power,
              nr.rayleigh,
              nr.standard_cauchy,
              nr.standard_exponential,
              partial(nr.standard_gamma,*(0.7,)),#shape
              nr.standard_normal,
              partial(nr.standard_t,*(4,)),#df
              partial(nr.vonmises,*(center,1.0))#mu,kappa
              ,partial(nr.wald,*(center,1.0))#mean,scale
              ,partial(nr.weibull,*(1.5,))#a
              ,partial(nr.zipf,*(1.5,))]#a
    if dist is None:
        dist=nr.randint(0,high=len(distribs))
    auger=distribs[dist]
    if scale==1:#1 randomly generated kernel and add
        #print('2d')
        aug=np.reshape(auger(size=dsh),dsh)
        for I,D in enumerate(data):
            data[I]=operat(D,aug)
    elif scale==2:#randomly generated kernels
        #print('many 2d')
        aug=[]
        for _ in range(ld):
            taug=np.reshape(auger(size=dsh),dsh)
            print(taug)
            aug.append(taug)
    else:#range of var and add single var to whole array
        #print("1d")
        aug=auger(size=ld)
    if scale!=1:
        for I,D in enumerate(data):
            data[I]=operat(D,aug[I])
    if not(dist==1 or dist==2) and ranges[0]!=0 and ranges[-1]!=1:
        for I in range(len(data)):
            data[I]=(data[I]-center)*(ranges[-1]-ranges[0])
    return(data)

def edger(shape,minpoints,num_edges=None,batch_edges=None,ranges=[0,2],arrange=[0,1],dist=0,mode=1,maxpoint=None,operator=1,seed=235464,round=None):
    """
    shape: shape of kernels
    minpoints: minimal number of non-zeros to have ina kernel
    num_edges: number of kernels, None= all available
    batch_edges: will scale up num_edges(num_edges*batch_edges)
    ranges: range of value to use in scaler
    arrange: range of values to use in edgemaker
    dist: distribution to use
    maxpoints: max number of points
    mode: 0: add single value to each kernel,
          1: add the same random kernel to each kernel
          2: a different random kernel to each kernel
    operator: 0: add, 1: mul, 2: (x+0.1)*random
    seed=rng seed
    """
    output=[]
    if maxpoint is None:
        maxpoint=reduce(operator.mul,shape)//2
    shapes=edgemaker(shape, minpoints,maxpoint=maxpoint,arrange=arrange)
    if num_edges is None:
        num_edges=len(shapes)
    if isinstance(batch_edges, int):
        edges=batch_edges*num_edges
    else:
        edges=num_edges
    for _ in range(edges//len(shapes)):
        [output.append(ir) for ir in shapes]
    for I in range(edges%len(shapes)):
            output.append(shapes[I])
    output=scaler(output,ranges,oper=operator,scale=mode,dist=dist,seed=seed)
    if round:
        output=np.round(output, decimals=round)
    return(output)
"""
def edgmaker(shape,vari):
    
    return(kern)
    
def scaler(data,range,dist):
    
    return(kern)"""
if __name__=='__main__':
    print(edger([3,3],2,maxpoint=5,mode=2,operator=2,seed=10102))
    
    
