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
    g = graph.Graph(net)
    return g

'''
def _conv_block_transform_grad():
    g = _conv_block()
    g = g.transform(passes="GradientV2")
    return g

def _conv_block_specialize_shape():
    g = _conv_block()
    g.specialize(passes="InferShapeV2", input_shapes=[(32, 100)])
    return g

def _conv_block_grad_and_shape():
    g = _conv_block()
    # ATTENTION: the following order cannot be changed.
    g = g.transform(passes="GradientV2")
    g.specialize(passes="InferShapeV2", input_shapes=[(32, 100)])
    return g
'''

def test_conv_compose_no_share():
    """
    Graph symbol compose rules:
    1. Allow missing input symbols. If missing, new variable
       will be created.
    2. How to deal with multiple outputs?
    """
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    net = ConvBlock(net, name='conv1')
    net = ConvBlock(net, name='conv2')
    assert net.list_arguments() == [
            'data',
            'conv1_conv_weight', 'conv1_conv_bias', 'conv1_bn_gamma', 'conv1_bn_beta', 'conv1_bn_moving_mean', 'conv1_bn_moving_var',
            'conv2_conv_weight', 'conv2_conv_bias', 'conv2_bn_gamma', 'conv2_bn_beta', 'conv2_bn_moving_mean', 'conv2_bn_moving_var'
            ]

def test_conv_compose_share():
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    conv_weight = sym.Variable('conv_weight')
    net = ConvBlock(data=net, conv_weight=conv_weight, name='conv1')
    net = ConvBlock(data=net, conv_weight=conv_weight, name='conv2')
    assert net.list_arguments() == [
            'data',
            'conv_weight',
            'conv1_conv_bias', 'conv1_bn_gamma', 'conv1_bn_beta', 'conv1_bn_moving_mean', 'conv1_bn_moving_var',
            'conv2_conv_bias', 'conv2_bn_gamma', 'conv2_bn_beta', 'conv2_bn_moving_mean', 'conv2_bn_moving_var'
            ]

def test_specialize_coloring():
    g = _conv_block()
    g.specialize(color=1)
    print(g.get_node_attr("node_colors"))
    ConvBlock = graph.symbolize(g)
    net = sym.Variable('data')
    net = ConvBlock(net, name='conv1')
    net = ConvBlock(net, name='conv2')
    net_graph = graph.Graph(net)
    net_graph.specialize(color=2)
    print(net_graph.get_node_attr("node_colors"))

if __name__ == '__main__':
    test_conv_compose_no_share()
    test_conv_compose_share()
    test_specialize_coloring()
