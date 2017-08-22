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
    #net = sym.Pooling(net, name='pool', kernel=(2,2), stride=(2,2), pool_type='max')
    g = graph.create(net)
    return g

def test_simple_eval():
    g = _conv_block()
    shape_args = {'shape_inputs' : [[16, 8, 10, 10]]}
    dtype_args = {'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args=shape_args,
                 mx_infer_dtype_args=dtype_args,
                 graph_frozen=1)
    a = mx.nd.zeros(shape_args['shape_inputs'][0], ctx=mx.cpu())
    g.eval([a])

if __name__ == '__main__':
    test_simple_eval()
