# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import math
import argparse

num_loops = 20
cold_skip = 10

# symbol net
def ConvModule(net, num_filter, kernel, pad=(0, 0), stride=(1, 1), fix_gamma=False):
    net = mx.sym.Convolution(data=net, kernel=kernel,
            stride=stride, pad=pad, num_filter=num_filter, no_bias=True)
    #net = mx.sym.BatchNorm(data=net, fix_gamma=fix_gamma)
    net = mx.sym.Activation(data=net, act_type="relu") # same memory to our act, less than CuDNN one
    return net

def ResModule(sym, base_filter, stage, layer, fix_gamma=False):
    num_f = base_filter * int(math.pow(2, stage))
    s = 1
    if stage != 0 and layer == 0:
        s = 2
    conv1 = ConvModule(sym, num_f, kernel=(1, 1), pad=(0, 0), stride=(1, 1))
    conv2 = ConvModule(conv1, num_f, kernel=(3, 3), pad=(1, 1), stride=(s, s))
    conv3 = ConvModule(conv2, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(1, 1))

    if layer == 0:
        sym = ConvModule(sym, num_f * 4, kernel=(1, 1), pad=(0, 0), stride=(s, s))

    sum_sym = sym + conv3
    force = layer % 2 == 1
    return sum_sym

# [3, 4, 6, 3]
def get_symbol(args, layers=[3, 4, 6, 3]):
    """Get a 4-stage residual net, with configurations specified as layers.

    Parameters
    ----------
    layers : list of stage configuratrion
    """
    assert(len(layers) == 4)
    base_filter = 64 * 2
    net = mx.sym.Variable(name='data')
    net = ConvModule(net, base_filter, kernel=(7, 7), pad=(3, 3), stride=(2, 2))
    net = mx.sym.Pooling(data=net, pool_type="max", kernel=(3, 3), stride=(2, 2))
    for j in range(len(layers)):
        for i in range(layers[j]):
            net = ResModule(net, base_filter, j, i)

    net = mx.symbol.Pooling(data=net, kernel=(7, 7), stride=(1, 1),
            name="globalpool", pool_type='avg')
    net = mx.symbol.Flatten(data=net, name='flatten')
    net = mx.symbol.FullyConnected(data=net, num_hidden=1000, \
            name='fc1', no_bias=True, attr={'num_gpus' : str(args.num_gpus)})
    # TODO(minjie): SoftmaxOutput
    #net = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    #return net, [('data', (64, 3, 224, 224))], [('softmax_label', (64,))]
    return net, [('data', (args.batch_size, 3, 224, 224))]

def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of gpus')
    args = parser.parse_args()
    net, data_shapes = get_symbol(args)

    data_shapes = dict(data_shapes)
    data_types = {name: mx.base.mx_real_t for name, shp in data_shapes.items()}

    #print(net.list_arguments())
    #print(net.list_outputs())

    # infer shapes
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**data_shapes)
    arg_types, out_types, aux_types = net.infer_type(**data_types)

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

    for i in range(num_loops):
        print('=> loop %d' % i);
        if i == cold_skip:
            t0 = time.time()
        outputs = executor.forward()
        executor.backward([outputs[0]])
        for name, grad in grad_dict.items():
            grad.wait_to_read()
    t1 = time.time()

    duration = t1 - t0
    print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_net()
