# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

# symbol net
def get_symbol(args):
  batch_size = args.batch_size
  hidden_size = args.hidden_size
  print('Batch size: %d, Hidden size: %d' % (batch_size, hidden_size))
  net = mx.symbol.Variable('data')
  for i in range(args.num_layers):
    net = mx.symbol.FullyConnected(net, name='fc%d' % i, num_hidden=hidden_size, no_bias=True, attr={'num_gpus': '1'})
  return net

def test_mlp():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--hidden_size', type=int, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=10, help='Number of hidden layers')
    args = parser.parse_args()
    net = get_symbol(args)

    print(net.list_arguments())
    print(net.list_outputs())

    # infer shapes
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(data=(args.batch_size, args.hidden_size))
    arg_types, out_types, aux_types = net.infer_type(data=mx.base.mx_real_t)

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

    for i in range(60):
        if i == 10:
            t0 = time.time()
        outputs = executor.forward()
        executor.backward([outputs[0]])
        for name, grad in grad_dict.items():
            grad.wait_to_read()
    t1 = time.time()

    duration = t1 - t0
    print('duration %f, average %f' % (duration, float(duration) / 50))


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_mlp()
