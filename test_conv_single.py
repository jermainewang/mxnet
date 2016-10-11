# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 15
cold_skip = 5

# symbol net
def get_symbol(args):
  batch_size = args.batch_size
  num_filters = args.num_filters
  print('Batch size: %d, #Filter: %d' % (batch_size, num_filters))
  net = mx.symbol.Variable('data')
  for i in range(args.num_layers):
    net = mx.symbol.Convolution(data=net, kernel=(3, 3), pad=(1, 1), stride=(1, 1),
                                num_filter=args.num_filters, no_bias=True,
                                attr={'num_gpus': '1'})
  return net, [('data', (args.batch_size, args.num_filters, 6, 6))]


def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_filters', type=int, help='Number of filters')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of hidden layers')
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
