# cvar2
Work in progress

As of now the package needs tensorflow and numpy to work, matplotlib is useful for the plotting scrip.


File explanations:

  chainerutils: code from chainer with some modified functions to split the data 
  reshaper: testing ground for the TF implementation ( and potentially MXNET and CNTK)
  varitest: implementation of the baseline functions and potential functions for chainer
  varplot: currently use to plot the two functions of varitest in order to see how the bias affects the output
# the plots seems to have an issue, The plot for Vecvari1 on the weight plots has (BB:0,BS:0) and (BB:1,BS:0) inverted, this may have repercussions on the last plot too

reshaper has the prototypes for TF,cntk and MXNET the layer for Chainer is in another file, the current versions of the alg need some tweaking though, this should be fixed soon


The results of varplot documenting the effect of bias on the algorithm are available, they are the 2 png.

Current setup under considerations:
* sizz:0,mul2:0,v3:0 
* sizz:0,mul2:1,v3:not(0)* 
* sizz:1,mul2:1,v3:0 

*v3[1,2,3] doesn't matter if mul2 is True, v3 only need to be also true
