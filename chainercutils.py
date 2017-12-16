
##taken from chainer
###conv utils

import numpy
import six

from chainer import cuda


def get_conv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of convolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_deconv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of convolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the convolution operation.

    """
    dk = k + (k - 1) * (d - 1)
    if cover_all:
        return (size + p * 2 - dk + s - 1) // s + 1
    else:
        return (size + p * 2 - dk) // s + 1


def get_deconv_outsize(size, k, s, p, cover_all=False, d=1):
    """Calculates output size of deconvolution.

    This function takes the size of input feature map, kernel, stride, and
    pooling of one particular dimension, then calculates the output feature
    map size of that dimension.

    .. seealso:: :func:`~chainer.utils.get_conv_outsize`

    Args:
        size (int): The size of input feature map. It usually is the length of
            a side of feature map.
        k (int): The size of deconvolution kernel.
        s (int): The size of stride.
        p (int): The size of padding.
        cover_all (bool): Use ``cover_all`` option or not.
        d (int): The size of dilation.

    Returns:
        int: The expected output size of the deconvolution operation.

    """
    dk = (k - 1) * d + 1
    if cover_all:
        return s * (size - 1) + dk - s + 1 - 2 * p
    else:
        return s * (size - 1) + dk - 2 * p


def im2col_cpuV2(
        img, kh, kw, sy, sx, ph, pw, pval=0, cover_all=False, dy=1, dx=1,
        out_h=None, out_w=None,og=True,channel1=False):
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

    img = numpy.pad(img,
                    ((0, 0), (0, 0), (ph, ph + sy - 1), (pw, pw + sx - 1)),
                    mode='constant', constant_values=(pval,))
    if og==1:#og shape is [num,chanel,window,window,output,output]
        col = numpy.ndarray((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
        for j in six.moves.range(kh):
            jdy = j * dy #window index * dilation
            j_lim = jdy + sy * out_h
            for i in six.moves.range(kw):
                idx = i * dx #window index * dilation
                i_lim = idx + sx * out_w
                col[:, :, j, i, :, :] = img[:, :, jdy:j_lim:sy, idx:i_lim:sx]

    elif og==2:
        pass
    else:#now shape is [number ,output,output, channel, window, window
        #if channel1:
        colshape=(n,c, out_h, out_w, kh, kw)
        colshape2=(n, out_h, out_w, c, kh, kw)
        #col = numpy.ndarray((n, out_h, out_w,1, c, kh, kw), dtype=img.dtype)
        col = numpy.ndarray(colshape, dtype=img.dtype)  
        #col = numpy.ndarray((n, out_h, out_w, c, kh, kw), dtype=img.dtype)
        
        for R in six.moves.range(kh):
            jdy = R * dy #window index * dilation
            j_lim = jdy + sy * out_h
            for H in six.moves.range(kw):
                idx = H * dx #window index * dilation
                i_lim = idx + sx * out_w
                col[:,:,:,:,R,H]=img[:, :, jdy:j_lim:sy, idx:i_lim:sx]#pour chaque point
        if not(channel1):
            col=numpy.expand_dims(numpy.rollaxis(col,1,-2),len(col.shape)//2)
            #col=numpy.expand_dims(numpy.reshape(col,colshape2),len(col.shape)//2)
        print(col.shape)
        #col=numpy.transpose(col, (0,2,1,3,4,5,6))
    
    return col     

def im2col_gpu(img, kh, kw, sy, sx, ph, pw, cover_all=False, dy=1, dx=1,
               out_h=None, out_w=None):
    n, c, h, w = img.shape
    if out_h is None:
        out_h = get_conv_outsize(h, kh, sy, ph, cover_all, dy)
    assert out_h > 0, 'Height in the output should be positive.'
    if out_w is None:
        out_w = get_conv_outsize(w, kw, sx, pw, cover_all, dx)
    assert out_w > 0, 'Width in the output should be positive.'

    col = cuda.cupy.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)
    cuda.elementwise(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)
    return col


def col2im_cpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = numpy.zeros((n, c, h + 2 * ph + sy - 1, w + 2 * pw + sx - 1),
                      dtype=col.dtype)
    for j in six.moves.range(kh):
        jdy = j * dy
        j_lim = jdy + sy * out_h
        for i in six.moves.range(kw):
            idx = i * dx
            i_lim = idx + sx * out_w
            img[:, :, jdy:j_lim:sy, idx:i_lim:sx] += col[:, :, j, i]
    return img[:, :, ph:h + ph, pw:w + pw]


def col2im_gpu(col, sy, sx, ph, pw, h, w, dy=1, dx=1):
    n, c, kh, kw, out_h, out_w = col.shape
    img = cuda.cupy.empty((n, c, h, w), dtype=col.dtype)
    cuda.elementwise(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img





####conv nd
import itertools
import numpy
import six

from chainer import cuda
from chainer.utils.conv import get_conv_outsize
from chainer.utils import conv_nd_kernel


def as_tuple(x, n):
    if hasattr(x, '__getitem__'):
        assert len(x) == n
        return tuple(x)
    return (x,) * n


def im2col_nd_cpu(img, ksize, stride, pad, pval=0, cover_all=False):
    n, c = img.shape[0:2]       # (n, c, d_1, d_2, ..., d_N)
    dims = img.shape[2:]
    ndim = len(dims)
    assert ndim == len(ksize) == len(stride) == len(pad)
    outs = tuple(get_conv_outsize(d, k, s, p, cover_all)
                 for (d, k, s, p) in zip(dims, ksize, stride, pad))
    assert all(out > 0 for out in outs), 'Output sizes should be positive.'

    # Pad around image.
    pad_width = ((0, 0), (0, 0)) + tuple(
        (p, p + s - 1) for (s, p) in zip(stride, pad))
    img = numpy.pad(img, pad_width, mode='constant', constant_values=(pval,))

    # Make patch array with which we will compute correlation with filter.
    # shape: (n, c, k_1, k_2, ..., k_N, out_1, out_2, ..., out_N)
    shape = (n, c) + ksize + outs
    col = numpy.ndarray(shape, dtype=img.dtype)

    # Fill the patch array.
    colon = slice(None)
    for kxs in itertools.product(*[six.moves.range(k) for k in ksize]):
        # col[:, :, kx_1, kx_2, ..., kx_N, :, :, ..., :]
        col_index = (colon, colon) + kxs + (colon,) * ndim
        # img[:, :, kx_1:kx_lim_1:s_1, ..., kx_N:kx_lim_N:s_N]
        kx_lims = tuple(kx + s * out
                        for (kx, s, out) in zip(kxs, stride, outs))
        img_index = (colon, colon) + tuple(
            slice(kx, kx_lim, s)
            for (kx, kx_lim, s) in zip(kxs, kx_lims, stride))
        col[col_index] = img[img_index]

    return col


def im2col_nd_gpu(img, ksize, stride, pad, cover_all=False):
    n, c = img.shape[0:2]       # (n, c, d_1, d_2, ..., d_N)
    dims = img.shape[2:]
    ndim = len(dims)
    assert ndim == len(ksize) == len(stride) == len(pad)
    outs = tuple(get_conv_outsize(d, k, s, p, cover_all)
                 for (d, k, s, p) in zip(dims, ksize, stride, pad))
    assert all(out > 0 for out in outs), 'Output sizes should be positive.'

    # col_shape: (n, c, k_1, k_2, ..., k_N, out_1, out_2, ..., out_N)
    shape = (n, c) + ksize + outs
    col = cuda.cupy.empty(shape, dtype=img.dtype)

    in_params, out_params, operation, name = \
        conv_nd_kernel.Im2colNDKernel.generate(ndim)

    cuda.elementwise(in_params, out_params, operation, name)(
        img.reduced_view(), *(dims + outs + ksize + stride + pad + (col,)))

    return col


def col2im_nd_cpu(col, stride, pad, dims):
    n, c = col.shape[:2]  # (n, c, kx_1, ..., kx_N, out_1, ..., out_N)
    mid = (len(col.shape) - 2) // 2 + 2
    ksize = col.shape[2:mid]
    outs = col.shape[mid:]
    colon = slice(None)
    assert len(outs) == len(ksize) == len(stride) == len(pad) == len(dims)

    # Image with padded size.
    img_shape = (n, c) + tuple(d + 2 * p + s - 1
                               for (d, p, s) in zip(dims, pad, stride))
    img = numpy.zeros(img_shape, dtype=col.dtype)
    for kxs in itertools.product(*[six.moves.range(k) for k in ksize]):
        # (:, :, kx_1:kx_lim_1:s_1, ..., kx_N:kx_lim_N:s_N)
        kx_lims = tuple(kx + s * out
                        for (kx, s, out) in zip(kxs, stride, outs))
        img_index = (colon, colon) + tuple(
            slice(kx, kx_lim, s)
            for (kx, kx_lim, s) in zip(kxs, kx_lims, stride))
        # (:, :, kx_1, kx_2, ..., kx_N, :, :, ..., :)
        col_index = (colon, colon) + kxs + (colon,) * len(outs)
        img[img_index] += col[col_index]

    # (:, :, p_1:d_1 + p_1, p_2:d_2 + p_2, ..., p_N:d_N + p_N]
    img_index = (colon, colon) + tuple(
        slice(p, d + p) for (p, d) in zip(pad, dims))
    return img[img_index]


def col2im_nd_gpu(col, stride, pad, dims):
    n, c = col.shape[:2]        # (n, c, k_1, ..., k_N, out_1, ..., out_N)
    mid = (len(col.shape) - 2) // 2 + 2
    ksize = col.shape[2:mid]
    outs = col.shape[mid:]
    ndim = len(dims)
    assert len(outs) == len(ksize) == len(stride) == len(pad) == ndim

    img_shape = (n, c) + dims   # (n, c, d_1, d_2, ..., d_N)
    img = cuda.cupy.empty(img_shape, dtype=col.dtype)

    in_params, out_params, operation, name = \
        conv_nd_kernel.Col2imNDKernel.generate(ndim)

    cuda.elementwise(in_params, out_params, operation, name)(
        col.reduced_view(), *(dims + outs + ksize + stride + pad + (img,)))

    return img






########ndkernel

import functools
import six

from chainer import cuda


def mulexp(xs, init=None):
    if init is not None:
        return functools.reduce('{} * {}'.format, xs, init)
    else:
        return functools.reduce('{} * {}'.format, xs)


def andexp(xs, init=None):
    if init is not None:
        return functools.reduce('{} && {}'.format, xs, init)
    else:
        return functools.reduce('{} && {}'.format, xs)


def muladdexp(xs, ys, init=None):
    def aux(exp, arg):
        x, y = arg
        return '({} + {} * {})'.format(y, x, exp)
    if init is not None:
        return functools.reduce(aux, six.moves.zip(xs, ys), init)
    else:
        return functools.reduce(aux, six.moves.zip(xs, ys))


def map_(fn, *lst):
    # For py2/py3 compatibility.
    return list(map(fn, *lst))


def succ_sublists(xs):
    # Returns successive sublists of xs.
    return [xs[i:] for i in six.moves.range(len(xs))]


def vars(prefix, n):
    return ['{}_{}'.format(prefix, i) for i in six.moves.range(n)]


class Writer(object):

    def __init__(self):
        self._indent = 0
        self._lines = []

    def write(self, line, indent=None):
        if indent == 'dec' or indent == 'decinc':
            self._indent -= 1
        self._lines.append('  ' * self._indent + line)
        if indent == 'inc' or indent == 'decinc':
            self._indent += 1

    def get(self):
        return '\n'.join(self._lines)


#
# im2col

class Im2colNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps):
        # 2D: raw T img, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(['raw T img'] + map_(aux, ds + outs + ks + ss + ps))

    def _out_params(self):
        return 'T col'

    def _compile_c0(self, outs, ks):
        # 2D: int c0 = i / (k_0 * k_1 * out_0 * out_1)
        return ['int c0 = i / ({});'.format(mulexp(ks + outs))]

    def _compile_kx(self, ndim, outs, ks):
        # 2D: int kx_0 = i / (k_1 * out_0 * out_1) % k_0;
        #     int kx_1 = i / (out_0 * out_1) % k_1;
        def aux(kx, xs):
            head = xs[0]
            tail = xs[1:] + outs
            if tail:
                return 'int {} = i / ({}) % {};'.format(kx, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(kx, head)
        kxs = vars('kx', ndim)
        kx_decls = map_(aux, kxs, succ_sublists(ks))
        return kx_decls, kxs

    def _compile_out_x(self, ndim, outs):
        # 2D: int out_x0 = i / (out_1) % out_0;
        #     int out_x1 = i % out_1;
        def aux(out_x, xs):
            head = xs[0]
            tail = xs[1:]
            if tail:
                return 'int {} = i / ({}) % {};'.format(
                    out_x, mulexp(tail), head)
            else:
                return 'int {} = i % {};'.format(out_x, head)
        out_xs = vars('out_x', ndim)
        out_x_decls = map_(aux, out_xs, succ_sublists(outs))
        return out_x_decls, out_xs

    def _compile_main(self, ndim, ds, ks, ss, ps, kxs, out_xs):
        # 2D: int in_0 = kx_0 + out_x_0 * s_0 - p_0;
        #     int in_1 = kx_1 + out_x_1 * s_1 - p_1;
        #     if (0 <= in_0 && in_0 < d_0 && 0 <= in_1 && in_1 < d_1) {
        #       int idx_0 = in_0 + d_0 * c0;
        #       int idx_1 = in_1 + d_1 * idx_0;
        #       col = img[idx_1];
        #     } else {
        #       col = (T)0;
        #     }
        w = Writer()

        ins = vars('in', ndim)
        for _in, kx, out_x, s, p in six.moves.zip(ins, kxs, out_xs, ss, ps):
            w.write('int {} = {} + {} * {} - {};'.format(_in, kx, out_x, s, p))

        def rel_aux(_in, d):
            return '0 <= {} && {} < {}'.format(_in, _in, d)
        w.write(
            'if ({}) {{'.format(andexp(map_(rel_aux, ins, ds))), indent='inc')

        idxs = vars('idx', ndim)
        idx0s = ['c0'] + idxs[:-1]
        for idx, _in, d, idx0 in six.moves.zip(idxs, ins, ds, idx0s):
            w.write('int {} = {} + {} * {};'.format(idx, _in, d, idx0))

        w.write('col = img[{}];'.format(idxs[-1]))
        w.write('} else {', indent='decinc')
        w.write('col = (T)0;')
        w.write('}', indent='dec')

        return [w.get()]

    def _operation(self, ndim, ds, outs, ks, ss, ps):
        c0 = self._compile_c0(outs, ks)
        kx, kxs = self._compile_kx(ndim, outs, ks)
        out_x, out_xs = self._compile_out_x(ndim, outs)
        main = self._compile_main(ndim, ds, ks, ss, ps, kxs, out_xs)
        return '\n'.join(c0 + kx + out_x + main)

    def _generate(self, ndim):
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)

        in_params = self._in_params(ds, outs, ks, ss, ps)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps)
        name = name = 'im2col_{}d'.format(ndim)
        return in_params, out_params, operation, name

    @staticmethod
    @cuda.memoize()
    def generate(ndim):
        return _im2col_nd_kernel._generate(ndim)


_im2col_nd_kernel = Im2colNDKernel()


#
# col2im

class Col2imNDKernel(object):

    def _in_params(self, ds, outs, ks, ss, ps):
        # 2D: raw T col, int32 d_0, int32 d_1, int32 out_0, int32 out_1,
        #     int32 k_0, int32 k_1, int32 s_0, int32 s_1, int32 p_0, int32 p_1
        def aux(x):
            return 'int32 {}'.format(x)
        return ', '.join(['raw T col'] + map_(aux, ds + outs + ks + ss + ps))

    def _out_params(self):
        return 'T img'

    def _compile_c0(self, ds):
        # 2D: int c0 = i / (d_0 * d_1);
        return ['int c0 = i / ({});'.format(mulexp(ds))]

    def _compile_x(self, ndim, ds, ps):
        # 2D: int x_0 = i / (d_1) % d_0 + p_0;
        #     int x_1 = i % d_1 + p_1;
        def aux(x, ds, p):
            head = ds[0]
            tail = ds[1:]
            if tail:
                return 'int {} = i / ({}) % {} + {};'.format(
                    x, mulexp(tail), head, p)
            else:
                return 'int {} = i % {} + {};'.format(x, head, p)
        xs = vars('x', ndim)
        x_decls = map_(aux, xs, succ_sublists(ds), ps)
        return x_decls, xs

    def _compile_loop(self, ndim, outs, ks, ss, xs):
        # 2D: int out_x0_0 = max(0,     (x_0 - k_0 + s_0) / s_0);
        #     int out_x1_0 = min(out_0, (x_0       + s_0) / s_0);
        #     int out_x0_1 = max(0,     (x_1 - k_1 + s_1) / s_1);
        #     int out_x1_1 = min(out_1, (x_1       + s_1) / s_1);
        #     ... Before-part here ...
        #     for (int out_x_0 = out_x0_0; out_x_0 < out_x1_0; ++out_x_0) {
        #       int kx_0 = x_0 - out_x_0 * s_0 + k_0 * c0;
        #       for (int out_x_1 = out_x0_1; out_x_1 < out_x1_1; ++out_x_1) {
        #         int kx_1 = x_1 - out_x_1 * s_1 + k_1 * kx_0;
        #         ... Main-part here ...
        #       }
        #     }
        #     ... After-part here ...
        def aux(out_x0, out_x1, out, x, k, s):
            return [
                'int {} = max(0, ({} - {} + {}) / {});'.format(
                    out_x0, x, k, s, s),
                'int {} = min({}, ({} + {}) / {});'.format(
                    out_x1, out, x, s, s)]
        out_x0s = vars('out_x0', ndim)
        out_x1s = vars('out_x1', ndim)
        bounds = sum(map_(aux, out_x0s, out_x1s, outs, xs, ks, ss), [])

        def _loop_main(main, ndim, ks, ss):
            w = Writer()

            # Loop openings.
            out_xs = vars('out_x', ndim)
            kxs = vars('kx', ndim)
            kxs1 = ['c0'] + kxs[:-1]
            for out_x, out_x0, out_x1, kx, s, x, k, kx1 in six.moves.zip(
                    out_xs, out_x0s, out_x1s, kxs, ss, xs, ks, kxs1):
                w.write('for (int {} = {}; {} < {}; ++{}) {{'.format(
                    out_x, out_x0, out_x, out_x1, out_x), indent='inc')
                w.write('int {} = {} - {} * {} + {} * {};'.format(
                    kx, x, out_x, s, k, kx1))

            # Main-part.
            kx = kxs[-1]
            for l in main(kx, out_xs).split('\n'):
                w.write(l)

            # Loop closings.
            for _ in out_xs:
                w.write('}', indent='dec')

            return [w.get()]

        return bounds, _loop_main

    def _compile_procedure(self, outs, xs):
        # 2D: val = val + col[(out_x_1 + out_1 * (out_x_0 + out_0 * kx_1))];
        def _main(kx, out_xs):
            index = muladdexp(outs, out_xs, kx)
            return 'val = val + col[{}];'.format(index)
        before = ['T val = 0;']
        after = ['img = val;']
        return before, _main, after

    def _operation(self, ndim, ds, outs, ks, ss, ps):
        c0 = self._compile_c0(ds)
        x, xs = self._compile_x(ndim, ds, ps)
        loop_bounds, loop_main = self._compile_loop(ndim, outs, ks, ss, xs)
        before, main, after = self._compile_procedure(outs, xs)
        return '\n'.join(
            c0 + x + loop_bounds + before + loop_main(
                main, ndim, ks, ss) + after)

    def _generate(self, ndim):
        ds = vars('d', ndim)
        outs = vars('out', ndim)
        ks = vars('k', ndim)
        ss = vars('s', ndim)
        ps = vars('p', ndim)

        in_params = self._in_params(ds, outs, ks, ss, ps)
        out_params = self._out_params()
        operation = self._operation(ndim, ds, outs, ks, ss, ps)
        name = 'col2im_{}d'.format(ndim)
        return in_params, out_params, operation, name

    @staticmethod
    @cuda.memoize()
    def generate(ndim):
        return _col2im_nd_kernel._generate(ndim)


_col2im_nd_kernel = Col2imNDKernel()
