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

def _conv_block_with_grad():
    g = _conv_block()
    '''
    xs = [{'node': 0, 'index': 0},  # data
          {'node': 1, 'index': 0},  # conv_weight
          {'node': 2, 'index': 0},  # conv_bias
          {'node': 4, 'index': 0},  # bn_gamma
          {'node': 5, 'index': 0}]  # bn_beta
    ys = [{'node': 10, 'index': 0}] # pool
    args = {'xs' : xs, 'ys' : ys, 'full_graph' : 0}
    '''
    # Compute gradients for all inputs using all outputs.
    # The gradient graph is also generated separately.
    xs_blacklist = [{'node': 6},  # bn_moving_mean
                    {'node': 7}]  # bn_moving_var
    args = {'xs_blacklist' : xs_blacklist}
    g = g.transform(["MXGradient"], mx_gradient_args=args)
    return g

def test_conv_compose_no_share():
    """
    Graph symbol compose rules:
    1. Allow missing input symbols. If missing, new variable
       will be created.
    2. How to deal with multiple outputs?
    """
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
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

def test_specialize_dot():
    g = _conv_block()
    g.specialize(save_dot=True)
    g_dot = g.get_global_attr("dot")[1]
    print(g_dot)
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    net_graph.specialize(save_dot=True)
    graph_dot = net_graph.get_global_attr("dot")[1]
    print(graph_dot)


def test_specialize_json():
    g = _conv_block()
    g.specialize(save_json=True)
    g_json = g.get_global_attr("json")[1]
    #print(g_json)
    '''
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    net = ConvBlock(net, name='conv1')
    net = ConvBlock(net, name='conv2')
    net_graph = graph.create(net)
    net_graph.specialize(save_json=True)
    graph_json = net_graph.get_global_attr("json")[1]
    print(graph_json)
    '''

def test_specialize_coloring():
    g = _conv_block()
    g.specialize(color=1)
    print(g.get_node_attr("node_colors"))
    ConvBlock = graph.symbolize(g)
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    net_graph.specialize(color=2)
    print(net_graph.get_node_attr("node_colors"))

def test_transform_grad():
    blk_graph = _conv_block_with_grad()
    blk_graph.specialize(save_dot=True)
    blk_dot = blk_graph.get_global_attr("dot")[1]
    print(blk_dot)
    ConvBlock = graph.symbolize(blk_graph)
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    xs_blacklist = [{'node': 0},  # data
                    {'node': 5},  # conv1_bn_moving_mean
                    {'node': 6},  # conv1_bn_moving_var
                    {'node': 12}, # conv2_bn_moving_mean
                    {'node': 13}] # conv2_bn_moving_var
    args = {'xs_blacklist' : xs_blacklist, 'full_graph' : 1}
    net_graph = net_graph.transform(["MXGradient"], mx_gradient_args=args)
    net_graph.specialize(save_dot=True)
    net_dot = net_graph.get_global_attr("dot")[1]
    print(net_dot)


if __name__ == '__main__':
    test_conv_compose_no_share()
    test_conv_compose_share()
    test_specialize_dot()
    #test_specialize_json()
    #test_specialize_coloring()
    test_transform_grad()
