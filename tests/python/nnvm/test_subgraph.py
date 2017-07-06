# pylint: skip-file
import mxnet as mx
import mxnet.graph as graph
import mxnet.symbol as sym

def _conv_block():
    net = sym.Variable('data')
    net = sym.Convolution(net, name='conv', num_filter=32, kernel=(3,3), stride=(2,2))
    net = sym.BatchNorm(net, name='bn')
    net = sym.Activation(net, name='relu', act_type='relu')
    net = sym.Pooling(net, name='pool', kernel=(2,2), stride=(2,2), pool_type='max')
    g = graph.create(net)
    return g

def test_conv():
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    net = ConvBlock(net, name='conv1')
    net = ConvBlock(net, name='conv2')

if __name__ == '__main__':
    test_conv()
