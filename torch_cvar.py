import torch
from torch import nn
from torch.nn import functional as fnn
from PIL import Image
from os.path import exists
from torchvision import transforms as imt


import numpy as np



def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    "kernel, step, stride,pad,cover, dilation"
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1

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

# imgtf=tf.convert_to_tensor()
# tfa=tf.transpose(imgtf, [0,2,3,1])#NWHC hit

class TvarLayer(nn.Module):  # K.layers.convolutional._Conv):#layers.Layer):
    """
    prototype of a variance layer/weighter variance,
    the concept is similar to convolution but uses weighted variance instead of a simple multiplication

    :KCD keep channel data, this is a way to not compress all the channel data, uses more mem but might be of use
    :param num_c: number of cell in this layer
    :filter size of the filter/kernel (int or container)
    :stride size of the stride/step (int or container)
    :padding type of padding to use, todo
    """

    def __init__(self, num_c, filtr, stride=1,  # num_routing=3,
                 sqrt=0, V=False, format='NCHW', sizz=0,independant_channels=True,squash_ch=False,
                 KCD=False,dilation=1, bigB=False, pad=0, activation="relu",
                 **kwargs):
        super(TvarLayer,self).__init__()
        self.noutputs = num_c
        self.window = filtr
        self.dilat=dilation
        self.stride = stride
        self.SCH=squash_ch
        self.sqrt = sqrt
        self.KCD = KCD
        assert (format == "NCHW") #or (format == "NHWC")
        self.format = format
        self.IDC=independant_channels
        self.sizz = sizz
        self.pad = pad
        self.shape = None
        self.arrs = None
        self.ashp = None
        self.convshape=None
        self.built=False
        self.L=False
        # print('modified iimage tnsor',tfa.eval(),'modified iimage tnsor')

    def tfwindow(self, arr, nc_to_nh=False, pad="VALID", stride=1):
        """
        this functions split the data into array of the proper shape for computation, this is based on the chainer function im2col
        https://docs.chainer.org/en/stable/reference/generated/chainer.functions.im2col.html

        pad can be "SAME",will add 0s to get the same output shape as input
        nc_to_nh: change from "NHWC" to "NCHW"
        stride = stride for the 2 internal axis
        """
        assert pad == "VALID" or pad == "SAME"
        print(arr.shape, "inwind")
        if self.format == "NCHW":
            dsize=(1, 1, *self.window)
        elif self.format == "NHWC":
            dsize=(1, *self.window, 1)
        temp=fnn.unfold(arr,self.window,dilation=self.dilat,padding=self.pad,stride=self.stride)
        #input(temp.shape)
        #if nc_to_nh and self.format == 'NCHW':
        #return (tf.transpose(tf.reshape(temp, (-1, *self.convshape, 1, self.ch, *self.window)),perm=[0, 1, 2, 3, 6, 4, 5]))
        if self.IDC:
            temp=torch.reshape(temp,[-1,self.ch,self.window[0]*self.window[1],temp.shape[-1]])
        temp=torch.unsqueeze(temp,1)
        if not self.L:
            self.L=temp.shape[-1]
        return (temp)

    def build(self, input_shape,dtype=torch.double):
        """
        create the weights and bias for the layer according to the keras docs
        """
        self.batchsize=input_shape[0]
        # self.arrs=input_shape
        if self.format == 'NHWC':
            self.ch=input_shape[-1]
            if self.IDC:
                if self.SCH:
                    self.W = nn.Parameter(
                        torch.randn((self.noutputs, self.window[0] * self.window[1]* self.ch,1), dtype=dtype))
                else:
                    self.W = nn.Parameter(
                    torch.randn((self.noutputs, self.window[0] * self.window[1], self.ch,1), dtype=dtype))
            else:
                self.W = nn.Parameter(torch.randn((self.noutputs, self.window[0]* self.window[1],1),dtype=dtype))
            self.ashp = (self.noutputs, self.window[0]* self.window[1], input_shape[-1])
        elif self.format == 'NCHW':
            self.ch=input_shape[1]
            if self.IDC:
                if self.SCH:
                    self.W = nn.Parameter(
                    torch.randn((self.noutputs, self.ch * self.window[0] * self.window[1],1), dtype=dtype))
                else:
                    self.W = nn.Parameter(
                        torch.randn((self.noutputs, self.ch, self.window[0] * self.window[1],1), dtype=dtype))
            else:
                self.W = nn.Parameter(torch.randn((self.noutputs, self.window[0] * self.window[1],1),dtype=dtype))
            self.ashp = (self.noutputs, input_shape[1], self.window[0]* self.window[1])
        if self.SCH:
            self.B = nn.Parameter(torch.randn((self.noutputs * self.ch,1,1),dtype=dtype))
        elif self.IDC:
            self.B = nn.Parameter(torch.randn((self.noutputs, self.ch,1,1),dtype=dtype))
        else:
            self.B = nn.Parameter(torch.randn((self.noutputs,1,1),dtype=dtype))
        self.convshape = convshape(input_shape[-2:], self.window, s=self.stride, p=self.pad, d=self.dilat)
        self.prodconv=np.product(self.convshape)
        twv =list(self.W.shape)
        if self.format == 'NHWC':
            # W=tf.transpose(W, [0,2,3,1])
            self.sb = (1, 1, self.W.shape[-1])
            self.WV = twv[1:3]
            self.ouch = twv[-1]
            self.xi = (-3, -2)
            self.x2 = (-1, -3, -2)
            self.x3=(-1,-3)
        elif self.format == 'NCHW':
            self.sb = (self.W.shape[1], 1, 1)
            self.WV = twv[-2:]
            self.ouch = twv[1]
            self.xi =  -2
            self.x2 = (-3, -2,)
            self.x3=(-2,-1)

        self.built = True

    def forward(self, array, training=None):
        """
        this is where the magic happens
        """
        if not(self.built):
            self.build(array.shape,dtype=array.dtype)
        print('convshape',self.convshape)
        reshaped = self.tfwindow(array)
        if (self.arrs is None):
            self.arrs = list(reshaped.shape)
            print('arss', self.arrs)
        print(self.xi,self.W.shape)
        mul = (reshaped * self.W)
        print(mul.shape,'mul')
        size = torch.sum(self.W, self.xi,keepdim=True)# keepdims=True)  # shape=(outputs, channel)
        print(size.shape,'size')
        mean=torch.mean(mul,self.xi,keepdim=True)
        print(mean.shape,'mean')
        i = (torch.pow((mul - mean),2)) / size
        print(i.shape,'premsum i',self.B.shape)
        if not (self.B is None):
            i = i + self.B
        #out = torch.reshape(i, (*i.shape[:2], -1))
        out = torch.reshape(i, (self.batchsize, -1,self.prodconv,))
        #i=torch.sum(i,2)

        print(i.shape,'i')

        #out = torch.sum(out, self.xi ir self.KCD else self.x2 ,keepdim=True)

        print(out.shape,'summed')
        if self.sqrt:
            out = torch.sqrt(out)

        print('convshape',self.convshape,out.shape)
        folder = fnn.fold(out,output_size=self.convshape, kernel_size=self.window,padding=[1,1])
        print(folder.shape,'folded')
        #TODO



        # print(out.shape,self.format,(self.arrs[0],self.arrs[1],self.arrs[2],self.ashp[0]))
        print(out.shape, self.format, (self.arrs[1], self.arrs[2], self.ashp[0]))
        return(out)
        '''
        if self.format == "NCHW":
            if self.KCD:
                return (tf.transpose(
                    tf.reshape(out, (self.arrs[0], self.arrs[1], self.arrs[2], self.ashp[0] * self.arrs[-3])),
                    (0, 3, 1, 2)))
            else:
                assert out.shape[1:] == (self.arrs[1], self.arrs[2], self.ashp[0])
                print(tf.transpose(out, (0, 3, 1, 2)).shape, "outshapenchw")
                return (tf.transpose(out, (
                0, 3, 1, 2)))  # tf.reshape(out,(self.arrs[0],self.ashp[0],self.arrs[1],self.arrs[2]))
        else:
            if self.KCD:
                return (tf.reshape(out, (self.arrs[0], self.arrs[1], self.arrs[2], self.ashp[0] * self.arrs[-3])))
            else:
                assert out.shape[1:] == (self.arrs[1], self.arrs[2], self.ashp[0])
                return (out)'''

        # return tuple([None, self.num_c,self.chnl, self.dim_vector,self.num_c])

if __name__=='__main__':
    display=False
    use_img=False
    img_path=r"C:\Users\ROYA2\Desktop\Capture.PNG"

    if exists(img_path) and use_img:
        image = Image.open(img_path)
        if display:
            image.save('input_img.png')
            image.show()

        img=imt.ToTensor()(image)
        input(img.shape)
        samp_data=torch.unsqueeze(img,0)
    else:
        img=np.random.rand(3,5,45,45)
        samp_data = torch.from_numpy(img)
    if display:
        image=imt.ToPILImage()(img)
        image.save('input_img.png')
        image.show()

    layer=TvarLayer(14,[3,3])
    out=layer.forward(samp_data)
    #input(out.shape)
    if display:
        image=imt.ToPILImage()(out[0])
        image.save('output_img.png')
        image.show()
    print(out,out.shape)
