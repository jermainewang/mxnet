# pylint: skip-file
from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
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
def get_symbol(args):
  batch_size = args.batch_size
  hidden_size = args.hidden_size
  print('Batch size: %d, Hidden size: %d' % (batch_size, hidden_size))
  net = mx.symbol.Variable('data')
  net = mx.symbol.FullyConnected(net, name="infc", num_hidden=hidden_size, no_bias=True)
  for i in range(args.num_layers):
    net = mx.symbol.FullyConnected(net, name='fc%d' % i, num_hidden=hidden_size, no_bias=True)
  net = mx.symbol.FullyConnected(net, name='presoft', num_hidden=1, no_bias=True)
  net = mx.symbol.SoftmaxOutput(net, name='softmax')
  return net, [('data', (args.batch_size, 1))], [('softmax_label', (args.batch_size,))]

def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--hidden_size', type=int, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of gpus used in data parallelism')
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
