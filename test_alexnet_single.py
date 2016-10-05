# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 50
cold_skip = 10

# symbol net
def conv_factory(data, num_filter, kernel, stride=(1, 1), pad=(1, 1), with_bn=False):
    net = mx.sym.Convolution(data,
                             num_filter=num_filter,
                             kernel=kernel,
                             stride=stride,
                             pad=pad)
    if with_bn:
        net = mx.sym.BatchNorm(net, fix_gamma=False)
    net = mx.sym.Activation(net, act_type="relu")
    return net

def get_symbol(args):
    net = mx.sym.Variable("data")
    # group 0
    net = conv_factory(net, num_filter=64, kernel=(11, 11), stride=(4, 4), pad=(2, 2))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # group 1
    net = conv_factory(net, num_filter=192, kernel=(5, 5), stride=(1, 1), pad=(2, 2))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # group 2
    net = conv_factory(net, num_filter=384, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = conv_factory(net, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = conv_factory(net, num_filter=256, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    net = mx.sym.Pooling(net, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # group 3
    net = mx.sym.Flatten(net)
    net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096)
    net = mx.sym.Activation(net, act_type="relu")
    # group 4
    net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096)
    net = mx.sym.Activation(net, act_type="relu")
    # group 5
    net = mx.sym.FullyConnected(net, num_hidden=1000, attr={'num_gpus' : '1'})
    # TODO(minjie): SoftmaxOutput
    return net, [('data', (args.batch_size, 3, 224, 224))]

def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()
    net, data_shapes = get_symbol(args)

    data_shapes = dict(data_shapes)
    data_types = {name: mx.base.mx_real_t for name, shp in data_shapes.items()}

    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**data_shapes)
    arg_types, out_types, aux_types = net.infer_type(**data_types)

    # create ndarrays for all arguments.
    arg_arrays = [mx.nd.zeros(shape, mx.gpu(0), dtype=dtype)
                  for shape, dtype in zip(arg_shapes, arg_types)]
    print('Num arguments: ', len(arg_arrays))
    # create gradient ndarray for all parameters.
    grad_dict = {name : mx.nd.zeros(shape, mx.gpu(0), dtype=dtype)
                 for name, shape, dtype in zip(net.list_arguments(), arg_shapes, arg_types)
                 if name != 'data'}
    print('Argument grads: ', grad_dict.keys())

    executor = net.bind(ctx=mx.gpu(0),
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
