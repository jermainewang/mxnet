# pylint: skip-file
import mxnet as mx
import mxnet.graph as graph
import mxnet.symbol as sym

kOnlyForward = 0
kOnlyBackward = 1
kFullGraph = 2

def _conv_block():
    net = sym.Variable('data')
    net = sym.Convolution(net, name='conv', num_filter=8, kernel=(3,3), stride=(1,1), pad=(1,1))
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
        g = g.transform(["MXGradientFull"], mx_gradient_args=args)
    elif mode == kOnlyForward:
        g = g.transform(["MXGradient"], mx_gradient_args=args)
    else:
        g = g.transform(["MXGradientOnlyBackward"], mx_gradient_args=args)
    return g

def test_simple_eval():
    g = _conv_block()
    shape_args = {'shape_inputs' : [[16, 8, 10, 10]]}
    dtype_args = {'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args=shape_args,
                 mx_infer_dtype_args=dtype_args,
                 graph_frozen=1)
    print(g.get_node_entry_attr("shape"))
    print(g.get_node_entry_attr("dtype"))
    data = mx.nd.zeros([16, 8, 10, 10], ctx=mx.cpu())
    conv_w = mx.nd.zeros([8, 8, 3, 3], ctx=mx.cpu())
    conv_b = mx.nd.zeros([8,], ctx=mx.cpu())
    bn_gamma = mx.nd.zeros([8,], ctx=mx.cpu())
    bn_beta = mx.nd.zeros([8,], ctx=mx.cpu())
    bn_mean = mx.nd.zeros([8,], ctx=mx.cpu())
    bn_var = mx.nd.zeros([8,], ctx=mx.cpu())
    results = g.eval([data, conv_w, conv_b, bn_gamma, bn_beta, bn_mean, bn_var])
    print(results, results.shape)

def test_grad_eval():
    g = _conv_block_with_grad(mode=kFullGraph)
    shape_args = {'shape_inputs' : [[16, 8, 10, 10]]}
    dtype_args = {'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args=shape_args,
                 mx_infer_dtype_args=dtype_args,
                 graph_frozen=1)
    print(g.get_node_entry_attr("shape"))
    print(g.get_node_entry_attr("dtype"))

if __name__ == '__main__':
    test_simple_eval()
    test_grad_eval()
