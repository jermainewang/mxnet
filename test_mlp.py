# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging

# symbol net
batch_size = 128
hidden_size = 32
input_size = hidden_size
out_size = hidden_size
net = mx.symbol.Variable('data')
net = mx.symbol.FullyConnected(net, name='fc1', num_hidden=hidden_size)
#net = mx.symbol.Activation(net, name='relu1', act_type="relu")
net = mx.symbol.FullyConnected(net, name = 'fc2', num_hidden=hidden_size)
#net = mx.symbol.Activation(net, name='relu2', act_type="relu")
net = mx.symbol.FullyConnected(net, name='fc3', num_hidden=hidden_size)
#net = mx.symbol.Activation(net, name='relu3', act_type="relu")
#net = mx.symbol.FullyConnected(net, name='fc4', num_hidden=hidden_size)
#net = mx.symbol.Activation(net, name='relu4', act_type="relu")
#net = mx.symbol.FullyConnected(net, name='fc5', num_hidden=hidden_size)
#net = mx.symbol.Activation(net, name='relu5', act_type="relu")
#net = mx.symbol.FullyConnected(net, name='fc6', num_hidden=hidden_size)

def test_mlp():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=(batch_size, input_size))
    arg_types, out_types, aux_types = net.infer_type(data=mx.base.mx_real_t)

    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    print('Num arguments: ', len(arg_arrays))
    # create gradient ndarray for all parameters.
    grad_dict = {name : mx.nd.zeros(shape, mx.cpu(0), dtype=dtype)
                 for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 if name != 'data'}
    print('Argument grads: ', grad_dict.keys())

    executor = net.bind(ctx=mx.cpu(0),
                        args=arg_arrays,
                        args_grad=grad_dict,
                        grad_req='write')

    for i in range(100):
        if i == 10:
            t0 = time.clock()
        outputs = executor.forward()
        executor.backward([outputs[0]])
        for name, grad in grad_dict.items():
            grad.wait_to_read()
    t1 = time.clock()

    duration = t1 - t0
    print('duration %f, speed %f' % (duration, float(duration) / 90))


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_mlp()
