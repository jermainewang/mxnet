# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 25
cold_skip = 5

vgg_type = \
{
    'A' : [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

def conv_factory(data, num_filter, kernel, stride=(1, 1), pad=(1, 1), with_bn=False):
    net = mx.sym.Convolution(data,
                             num_filter=num_filter,
                             kernel=kernel,
                             stride=stride,
                             pad=pad,
                             no_bias=True)
    if with_bn:
        net = mx.sym.BatchNorm(net, fix_gamma=False)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    return net


def vgg_body_factory(structure_list):
    net = mx.sym.Variable("data")
    for item in structure_list:
        if type(item) == str:
            net = mx.sym.Pooling(net, kernel=(2, 2), stride=(2, 2), pool_type="max")
        else:
            net = conv_factory(net, num_filter=item, kernel=(3, 3), stride=(1, 1), pad=(1, 1))
    return net


def get_symbol(args, net_type='D'):
    net = vgg_body_factory(vgg_type[net_type])
    # group 3
    net = mx.sym.Flatten(net)
    net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    # group 4
    net = mx.sym.Dropout(net, p=0.5)
    net = mx.sym.FullyConnected(net, num_hidden=4096)
    net = mx.sym.Activation(net, act_type="relu")
    net._set_attr(mirror_stage='True')
    # group 5
    net = mx.sym.FullyConnected(net, num_hidden=1000, attr={'num_gpus' : str(args.num_gpus)})
    # TODO(xxx): Test speed difference between SoftmaxActivation and SoftmaxOutput
    #net = mx.sym.SoftmaxOutput(net, name="softmax")
    return net, [('data', (args.batch_size, 3, 224, 224))]


def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_gpus', type=int, default=2, help='Number of gpus')
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
