# pylint: skip-file
import mxnet as mx
import mxnet.graph as graph
import mxnet.symbol as sym

kOnlyForward = 0
kOnlyBackward = 1
kFullGraph = 2

def _Viz(dot_str, fname='tmp.dot'):
    print(dot_str)
    with open(fname, 'w') as f:
        f.write(dot_str)
        f.flush()

def _conv_block():
    net = sym.Variable('data')
    net = sym.Convolution(net, name='conv', num_filter=32, kernel=(3,3), stride=(1,1), pad=(1,1))
    net = sym.BatchNorm(net, name='bn')
    net = sym.Activation(net, name='relu', act_type='relu')
    g = graph.create(net)
    return g

def _conv_block_with_grad(mode=kOnlyForward):
    g = _conv_block()
    # Compute gradients for all inputs using all outputs.
    # The gradient graph is also generated separately.
    xs_blacklist = [{'node': 6},  # bn_moving_mean
                    {'node': 7}]  # bn_moving_var
    args = {'xs_blacklist' : xs_blacklist}
    if mode == kFullGraph:
        g = g.transform(["MXGradientFullJSON"], mx_gradient_args_json=args)
    elif mode == kOnlyForward:
        g = g.transform(["MXGradientJSON"], mx_gradient_args_json=args)
    else:
        g = g.transform(["MXGradientOnlyBackwardJSON"], mx_gradient_args_json=args)
    return g

def _conv_net():
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    return graph.create(net)

def _conv_net_with_grad(mode=kOnlyForward):
    ConvBlock = graph.symbolize(_conv_block_with_grad())
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    xs_blacklist = [{'node': 0},  # data
                    {'node': 5},  # conv1_bn_moving_mean
                    {'node': 6},  # conv1_bn_moving_var
                    {'node': 12}, # conv2_bn_moving_mean
                    {'node': 13}] # conv2_bn_moving_var
    args = {'xs_blacklist' : xs_blacklist}
    if mode == kFullGraph:
        net_graph = net_graph.transform(["MXGradientFullJSON"], mx_gradient_args_json=args)
    elif mode == kOnlyForward:
        net_graph = net_graph.transform(["MXGradientJSON"], mx_gradient_args_json=args)
    else:
        net_graph = net_graph.transform(["MXGradientOnlyBackwardJSON"], mx_gradient_args_json=args)
    return net_graph

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
            'conv1_conv_weight', 'conv1_conv_bias', 'conv1_bn_gamma', 'conv1_bn_beta',
            'conv2_conv_weight', 'conv2_conv_bias', 'conv2_bn_gamma', 'conv2_bn_beta']
    assert net.list_auxiliary_states() == [
            'conv1_bn_moving_mean', 'conv1_bn_moving_var',
            'conv2_bn_moving_mean', 'conv2_bn_moving_var']
    assert set(net.list_inputs()) == set(net.list_arguments()).union(set(net.list_auxiliary_states()))
    assert net.list_outputs() == ['conv2_output']

def test_conv_compose_share():
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    conv_weight = sym.Variable('conv_weight')
    net = ConvBlock(data=net, conv_weight=conv_weight, name='conv1')
    net = ConvBlock(data=net, conv_weight=conv_weight, name='conv2')
    assert net.list_arguments() == [
            'data',
            'conv_weight',
            'conv1_conv_bias', 'conv1_bn_gamma', 'conv1_bn_beta',
            'conv2_conv_bias', 'conv2_bn_gamma', 'conv2_bn_beta',]
    assert net.list_auxiliary_states() == [
            'conv1_bn_moving_mean', 'conv1_bn_moving_var',
            'conv2_bn_moving_mean', 'conv2_bn_moving_var']
    assert set(net.list_inputs()) == set(net.list_arguments()).union(set(net.list_auxiliary_states()))
    assert net.list_outputs() == ['conv2_output']

def test_specialize_dot():
    g = _conv_block()
    g.specialize(save_dot=True)
    g_dot = g.get_global_attr("dot")[1]
    _Viz(g_dot, 'blk.dot')
    ConvBlock = graph.symbolize(_conv_block())
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    net_graph.specialize(save_dot=True)
    graph_dot = net_graph.get_global_attr("dot")[1]
    _Viz(graph_dot, 'net.dot')


def test_specialize_json():
    g = _conv_block()
    g.specialize(save_json=True)
    g_json = g.get_global_attr("json")[1]
    print(g_json)
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

def test_transform_grad_no_subgraph():
    g = _conv_block_with_grad(mode=kFullGraph)
    g.specialize(save_dot=True)
    _Viz(g.get_global_attr("dot")[1], 'blk_grad.dot')
    #print(g.get_global_attr("json")[1])

def test_transform_grad_subgraph():
    net_graph = _conv_net_with_grad(mode=kFullGraph)
    net_graph.specialize(save_dot=True)
    _Viz(net_graph.get_global_attr("dot")[1], 'net_grad.dot')
    #print(net_graph.get_global_attr("json")[1])

def test_transform_grad_only_backward():
    g = _conv_block_with_grad(mode=kOnlyBackward)
    g.specialize(save_dot=True)
    _Viz(g.get_global_attr("dot")[1], 'blk_only_grad.dot')
    net_graph = _conv_net_with_grad(mode=kOnlyBackward)
    net_graph.specialize(save_dot=True)
    _Viz(net_graph.get_global_attr("dot")[1], 'net_only_grad.dot')

def test_high_order_grad():
    x = sym.Variable('x')
    y = sym.exp(x)
    g = graph.create(y)
    g = g.transform(["MXGradientFullJSON"], mx_gradient_args_json={})
    g = g.transform(["MXGradientFullJSON"], mx_gradient_args_json={})
    # Following will raise error since _backward_mul did not register gradient function.
    # g = g.transform(["MXGradientFullJSON"], mx_gradient_args_json={})
    g.specialize(save_dot=True)
    print(g.get_global_attr("dot")[1])


def test_infer_shape_no_subgraph():
    # Test no grad
    g = _conv_block()
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    g.specialize(mx_infer_shape_args_json=args)
    print(g.get_node_entry_attr("shape"))
    # Test with grad
    g_grad = _conv_block_with_grad(mode=kFullGraph)
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    g_grad.specialize(save_dot=True)
    _Viz(g_grad.get_global_attr("dot")[1])
    g_grad.specialize(mx_infer_shape_args_json=args)
    print(g_grad.get_node_entry_attr("shape"))

def test_infer_shape_subgraph1():
    # Not pre-specialized.
    g = _conv_net()
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    g.specialize(mx_infer_shape_args_json=args)
    print(g.get_node_entry_attr("shape"))

def test_infer_shape_subgraph2():
    # Specialized subgraph.
    g = _conv_block()
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    g.specialize(mx_infer_shape_args_json=args)
    ConvBlock = graph.symbolize(g)
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    net_graph.specialize(mx_infer_shape_args_json=args)
    print(net_graph.get_node_entry_attr("shape"))

def test_infer_shape_subgraph_grad1():
    g = _conv_net_with_grad(mode=kFullGraph)
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    g.specialize(mx_infer_shape_args_json=args)
    print(g.get_node_entry_attr("shape"))

def test_infer_shape_subgraph_grad2():
    g = _conv_block_with_grad()
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    g.specialize(mx_infer_shape_args_json=args)
    ConvBlock = graph.symbolize(g)
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    xs_blacklist = [{'node': 0},  # data
                    {'node': 5},  # conv1_bn_moving_mean
                    {'node': 6},  # conv1_bn_moving_var
                    {'node': 12}, # conv2_bn_moving_mean
                    {'node': 13}] # conv2_bn_moving_var
    args = {'xs_blacklist' : xs_blacklist}
    net_graph = net_graph.transform(["MXGradientFullJSON"], mx_gradient_args_json=args)
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    net_graph.specialize(mx_infer_shape_args_json=args)
    print(net_graph.get_node_entry_attr("shape"))

def test_infer_shape_only_backward():
    g = _conv_block_with_grad()
    args = {'shape_inputs' : [[256, 32, 100, 100]]}
    g.specialize(mx_infer_shape_args_json=args)
    fwd_shapes = g.get_node_entry_attr("shape")[1]
    print(fwd_shapes)
    g_grad = _conv_block_with_grad(mode=kOnlyBackward)
    args = {'shape_inputs' : [[256, 32, 100, 100]],
            'forward_shapes' : fwd_shapes}
    g_grad.specialize(mx_infer_shape_args_json=args)
    print(g_grad.get_node_entry_attr("shape"))

def test_plan_memory_no_subgraph():
    # Test no grad
    g = _conv_block()
    shape_args={'shape_inputs' : [[256, 32, 100, 100]]}
    dtype_args={'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args_json=shape_args,
                 mx_infer_dtype_args_json=dtype_args,
                 graph_frozen=1)
    # Test with grad
    g_grad = _conv_block_with_grad(mode=kFullGraph)
    shape_args = {'shape_inputs' : [[256, 32, 100, 100]]}
    dtype_args = {'dtype_inputs' : [0]}
    g_grad.specialize(mx_infer_shape_args_json=shape_args,
                      mx_infer_dtype_args_json=dtype_args,
                      graph_frozen=1)

def test_plan_memory_subgraph():
    # Specialized subgraph.
    g = _conv_block()
    shape_args = {'shape_inputs' : [[256, 32, 100, 100]]}
    dtype_args = {'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args_json=shape_args,
                 mx_infer_dtype_args_json=dtype_args)
    ConvBlock = graph.symbolize(g)
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    net_graph.specialize(mx_infer_shape_args_json=shape_args,
                         mx_infer_dtype_args_json=dtype_args,
                         graph_frozen=1)

def test_plan_memory_subgraph_grad():
    g = _conv_block_with_grad()
    shape_args = {'shape_inputs' : [[256, 32, 100, 100]]}
    dtype_args = {'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args_json=shape_args,
                 mx_infer_dtype_args_json=dtype_args)

    ConvBlock = graph.symbolize(g)
    net = sym.Variable('data')
    net = ConvBlock(data=net, name='conv1')
    net = ConvBlock(data=net, name='conv2')
    net_graph = graph.create(net)
    xs_blacklist = [{'node': 0},  # data
                    {'node': 5},  # conv1_bn_moving_mean
                    {'node': 6},  # conv1_bn_moving_var
                    {'node': 12}, # conv2_bn_moving_mean
                    {'node': 13}] # conv2_bn_moving_var
    args = {'xs_blacklist' : xs_blacklist}
    net_graph = net_graph.transform(["MXGradientFullJSON"], mx_gradient_args_json=args)
    shape_args = {'shape_inputs' : [[256, 32, 100, 100]]}
    dtype_args = {'dtype_inputs' : [0]}
    net_graph.specialize(mx_infer_shape_args_json=shape_args,
                         mx_infer_dtype_args_json=dtype_args,
                         graph_frozen=1)

if __name__ == '__main__':
    test_conv_compose_no_share()
    test_conv_compose_share()
    test_specialize_dot()
    test_specialize_json()
    test_transform_grad_no_subgraph()
    test_transform_grad_subgraph()
    test_transform_grad_only_backward()
    test_high_order_grad()
    test_infer_shape_no_subgraph()
    test_infer_shape_subgraph1()
    test_infer_shape_subgraph2()
    test_infer_shape_subgraph_grad1()
    test_infer_shape_subgraph_grad2()
    #test_infer_shape_only_backward()
    test_plan_memory_no_subgraph()
    test_plan_memory_subgraph()
    test_plan_memory_subgraph_grad()
