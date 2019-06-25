#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 11:15:51 2018

@author: chenchen
"""

# As usual, a bit of setup
import os
os.chdir("/Users/chenchen/Documents/MSNE/B_Semester_2/DL/Exercise/i2dl/exercise_2")
import sys
sys.path.append("/Users/chenchen/Documents/MSNE/B_Semester_2/DL/Exercise/i2dl/exercise_2")
# A bit of setup
import time
import numpy as np
import matplotlib.pyplot as plt
from exercise_code.classifiers.fc_net import *
from exercise_code.data_utils import get_CIFAR10_data
from exercise_code.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from exercise_code.solver import Solver

# =============================================================================
# %matplotlib inline
# =============================================================================
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
# =============================================================================
    # %load_ext autoreload
    # %autoreload 2
# =============================================================================

# supress cluttering warnings in solutions
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# def rel_error(x, y):
#   """ returns relative error """
#   return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
# # Load the (preprocessed) CIFAR10 data.
# 
# data = get_CIFAR10_data()
# for k, v in data.items():
#     print('%s: ' % k, v.shape)
#     
# # Test the affine_forward function
# 
# num_inputs = 2
# input_shape = (4, 5, 6)
# output_dim = 3
# 
# input_size = num_inputs * np.prod(input_shape)
# weight_size = output_dim * np.prod(input_shape)
# 
# x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
# w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
# b = np.linspace(-0.3, 0.1, num=output_dim)
# 
# out, _ = affine_forward(x, w, b)
# correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
#                         [ 3.25553199,  3.5141327,   3.77273342]])
# 
# # Compare your output with ours. The error should be around 1e-9.
# print('Testing affine_forward function:')
# print('difference: ', rel_error(out, correct_out))
# 
# x = np.random.randn(10, 2, 3)
# w = np.random.randn(6, 5)
# b = np.random.randn(5)
# dout = np.random.randn(10, 5)
# 
# dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
# dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
# db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
# 
# _, cache = affine_forward(x, w, b)
# dx, dw, db = affine_backward(dout, cache)
# 
# # The error should be around 1e-10
# print('Testing affine_backward function:')
# print('dx error: ', rel_error(dx_num, dx))
# print('dw error: ', rel_error(dw_num, dw))
# print('db error: ', rel_error(db_num, db))
# =============================================================================
N, D, H1, H2, C = 2, 15, 20, 30, 10
X = np.random.randn(N, D)
y = np.random.randint(C, size=(N,))

for reg in [0, 3.14]:
  print('Running check with reg = ', reg)
  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,
                            reg=reg, weight_scale=5e-2, dtype=np.float64)

  loss, grads = model.loss(X, y)
  print('Initial loss: ', loss)

  for name in sorted(grads):
    f = lambda _: model.loss(X, y)[0]
    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)
    print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))