# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import math
import argparse

num_loops = 30
cold_loops = 5
class Timer:
  def __init__(self):
    self.t0 = None
    self.t1 = None
  def start(self):
    self.t0 = time.time()
  def dur(self):
    return time.time() - self.t0

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
    layers[0] *= args.res1
    layers[1] *= args.res2
    layers[2] *= args.res3
    layers[3] *= args.res4
    base_filter = 64 * args.fat
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
            name='fc1', no_bias=True, attr={'num_gpus' : '1'})
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    return net, [('data', (args.batch_size, 3, 224, 224))], [('softmax_label', (args.batch_size,))]

def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of gpus used in data parallelism')
    parser.add_argument('--fat', type=int, default=1, help='Multiplier on channel size')
    parser.add_argument('--res1', type=int, default=1, help='Multiplier on the number of 1st ResModule')
    parser.add_argument('--res2', type=int, default=1, help='Multiplier on the number of 2nd ResModule')
    parser.add_argument('--res3', type=int, default=1, help='Multiplier on the number of 3rd ResModule')
    parser.add_argument('--res4', type=int, default=1, help='Multiplier on the number of 4th ResModule')
    args = parser.parse_args()
    net, data_shapes, label_shapes = get_symbol(args)

    train_data_shape = data_shapes[0][1]
    train_label_shape = label_shapes[0][1]
    train_iter = mx.io.NDArrayIter(
        data=mx.nd.zeros(train_data_shape, mx.gpu(0)),
        label=mx.nd.zeros(train_label_shape, mx.gpu(0)),
        batch_size=args.batch_size)
    kv = mx.kvstore.create('device')
    train_ctx = [mx.gpu(i) for i in range(args.num_gpus)]
    
    model = mx.model.FeedForward(ctx=train_ctx,
                                 symbol=net,
                                 num_epoch=num_loops,
                                 learning_rate=0.0,
                                 optimizer='sgd')

    timer = Timer()
    def _callback(epoch):
      if epoch.epoch == cold_loops:
        timer.start()
    model.fit(X=train_iter,
              kvstore=kv,
              batch_end_callback=_callback)

    duration = timer.dur()
    print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_loops)))


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_net()
