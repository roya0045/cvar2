# cvar2
Work in progress Currently this function seems to break backpropagation in TF, it seemed to have worked once or twice but I can't get it to work properly, backpropagation may need to be formulated explicitly ( this is currently out of my expertise)

As of now the package needs tensorflow and numpy to work, matplotlib is useful for the plotting scrip.


File explanations:

  chainerutils: code from chainer with some modified functions to split the data 
  reshaper: testing ground for the TF implementation ( and potentially MXNET and CNTK)
  varitest: implementation of the baseline functions and potential functions for chainer
  varplot: currently use to plot the two functions of varitest in order to see how the bias affects the output
  edger: makes kernels for testing
  

