# cvar2
Work in progress

As of now the package needs tensorflow and numpy to work, matplotlib is useful for the plotting scrip.


File explanations:

  chainerutils: code from chainer with some modified functions to split the data 
  reshaper: testing ground for the TF implementation ( and potentially MXNET and CNTK)
  varitest: implementation of the baseline functions and potential functions for chainer
  varplot: currently use to plot the two functions of varitest in order to see how the bias affects the output


reshaper has the prototypes for TF,cntk and MXNET the layer for Chainer is in another file, the current versions of the alg need some tweaking though, this should be fixed soon


The results of varplot documenting the effect of bias on the algorithm are available, they are the 2 png.
