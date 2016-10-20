from __future__ import print_function

import mxnet as mx
import numpy as np
import os, sys,time
import pickle as pickle
import logging
import argparse

num_loops = 30
cold_loops = 10

class Timer:
  def __init__(self):
    self.t0 = None
    self.t1 = None
  def start(self):
    self.t0 = time.time()
  def dur(self):
    return time.time() - self.t0

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
    net = mx.sym.SoftmaxOutput(net, name="softmax")
    return net, [('data', (args.batch_size, 3, 224, 224))], [('softmax_label', (args.batch_size,))]

def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
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
