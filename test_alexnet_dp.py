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
    net = mx.sym.SoftmaxOutput(net, name='softmax')
    return net, [('data', (args.batch_size, 3, 224, 224))], [('softmax_label', (args.batch_size,))]

def test_net():
    # print logging by default
    logging.basicConfig(level=logging.DEBUG)

    print(sys.argv)
    parser = argparse.ArgumentParser("MLP single card code")
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of gpus used in data parallelism')
    args = parser.parse_args()
    net, data_shapes, label_shapes = get_symbol(args)

    train_iter = mx.io.NDArrayIter(
        data=mx.nd.zeros(data_shapes[0][1], mx.cpu()),
        label=mx.nd.zeros(label_shapes[0][1], mx.cpu()),
        batch_size=args.batch_size)
    kv = mx.kvstore.create('device')
    train_ctx = [mx.gpu(i) for i in range(args.num_gpus)]
    
    model = mx.model.FeedForward(ctx=train_ctx,
                                 symbol=net,
                                 num_epoch=10,
                                 learning_rate=0.0,
                                 optimizer='sgd')

    t0 = time.time()
    model.fit(X=train_iter,
              kvstore=kv)
    t1 = time.time()

    duration = t1 - t0
    print('duration %f, average %f' % (duration, float(duration) / (num_loops - cold_skip)))


if __name__ == "__main__":
    print('================ Test Begin ====================')
    test_net()
