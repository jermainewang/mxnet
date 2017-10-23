# pylint: skip-file
import mxnet as mx
import mxnet.graph as graph
import mxnet.symbol as sym

import numpy as np

kOnlyForward = 0
kOnlyBackward = 1
kFullGraph = 2
kFullGraphWithOutput = 3

rng = np.random.RandomState(seed=32)
ctx = mx.gpu(0)

def _conv_block_sym():
    net = sym.Variable('data')
    net = sym.Convolution(net, name='conv', num_filter=8, kernel=(3,3), stride=(2,2), pad=(1,1))
    net = sym.BatchNorm(net, name='bn')
    net = sym.Activation(net, name='relu', act_type='relu')
    return net

def _conv_block():
    g = graph.create(_conv_block_sym())
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
    elif mode == kFullGraphWithOutput:
        g = g.transform(["MXGradientFullWithOutput"], mx_gradient_args=args)
    elif mode == kOnlyForward:
        g = g.transform(["MXGradient"], mx_gradient_args=args)
    else:
        g = g.transform(["MXGradientOnlyBackward"], mx_gradient_args=args)
    #s = graph.symbolize(g)
    #n = s(data=mx.sym.Variable('data'))
    #print(g.name)
    #print(n.list_inputs())
    #print(n.list_outputs())
    return g

def _pack_in_arrays(arg_arrays, arg_names, aux_arrays, aux_names, input_names):
    arg_name_dict = {}
    aux_name_dict = {}
    for i, name in enumerate(arg_names):
        arg_name_dict[name] = i
    for i, name in enumerate(aux_names):
        aux_name_dict[name] = i
    in_arrays = []
    for name in input_names:
        if name in arg_name_dict:
            in_arrays.append(arg_arrays[arg_name_dict[name]])
        else:
            in_arrays.append(aux_arrays[aux_name_dict[name]])
    return in_arrays

def _pack_grad_in_arrays(arg_arrays, arg_names, aux_arrays, aux_names,
                         head_grad_arrays, output_names, input_names):
    arg_name_dict = {}
    aux_name_dict = {}
    for i, name in enumerate(arg_names):
        arg_name_dict[name] = i
    for i, name in enumerate(aux_names):
        aux_name_dict[name] = i
    in_arrays = []
    for name in input_names:
        if name in arg_name_dict:
            in_arrays.append(arg_arrays[arg_name_dict[name]])
        else:
            in_arrays.append(aux_arrays[aux_name_dict[name]])
    # TODO(minjie): better solution here.
    in_arrays += head_grad_arrays
    return in_arrays


def _test_eval_helper(net, g, data_shape):
    assert 'data' in net.list_inputs()
    # Create legacy executor.
    in_shapes = {'data': tuple(data_shape)}
    in_types = {'data': mx.base.mx_real_t}
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**in_shapes)
    arg_dtypes, out_dtypes, aux_dtypes = net.infer_type(**in_types)
    np_arg_arrays = [rng.uniform(-0.1, 0.1, shape).astype(np.float32) for shape in arg_shapes]
    np_aux_states = [np.zeros(shape).astype(np.float32) for shape in aux_shapes]
    arg_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_arg_arrays]
    aux_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_aux_states]
    executor = net.bind(ctx=ctx,
                        args=arg_arrays,
                        args_grad=None,
                        grad_req='write',
                        aux_states=aux_arrays)
    legacy_results = executor.forward(is_train=False)
    # Use new eval API.
    shape_args = {'shape_inputs' : [data_shape]}
    dtype_args = {'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args=shape_args,
                 mx_infer_dtype_args=dtype_args,
                 graph_frozen=1)
    # Reset all inputs.
    arg_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_arg_arrays]
    aux_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_aux_states]
    in_arrays = _pack_in_arrays(arg_arrays, net.list_arguments(),
                                aux_arrays, net.list_auxiliary_states(),
                                net.list_inputs())
    new_results = g.eval(in_arrays, is_training=False)
    if not isinstance(new_results, list):
        new_results = [new_results]
    assert len(legacy_results) == len(new_results)
    assert all([np.allclose(r1.asnumpy(), r2.asnumpy(), rtol=1e-5, atol=1e-6)
                for r1, r2 in zip(legacy_results, new_results)])

def _test_grad_eval_helper(net, g, data_shape):
    assert 'data' in net.list_inputs()
    # Create legacy executor.
    in_shapes = {'data': tuple(data_shape)}
    in_types = {'data': mx.base.mx_real_t}
    arg_shapes, out_shapes, aux_shapes = net.infer_shape(**in_shapes)
    arg_dtypes, out_dtypes, aux_dtypes = net.infer_type(**in_types)
    np_arg_arrays = [rng.uniform(-0.1, 0.1, shape).astype(np.float32) for shape in arg_shapes]
    np_aux_states = [np.zeros(shape).astype(np.float32) for shape in aux_shapes]
    arg_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_arg_arrays]
    aux_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_aux_states]
    legacy_grad_arrays = [mx.nd.zeros(arr.shape, ctx=ctx) for arr in arg_arrays]
    executor = net.bind(ctx=ctx,
                        args=arg_arrays,
                        args_grad=legacy_grad_arrays,
                        grad_req='write',
                        aux_states=aux_arrays)
    results = executor.forward(is_train=True)
    np_head_grad_arrays = [rng.uniform(-0.1, 0.1, shape).astype(np.float32)
                           for shape in out_shapes]
    head_grad_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_head_grad_arrays]
    executor.backward(head_grad_arrays)
    # Use new eval API.
    shape_args = {'shape_inputs' : [data_shape]}
    dtype_args = {'dtype_inputs' : [0]}
    g.specialize(mx_infer_shape_args=shape_args,
                 mx_infer_dtype_args=dtype_args,
                 graph_frozen=1)
    # Reset all inputs.
    arg_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_arg_arrays]
    aux_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_aux_states]
    in_arrays = _pack_grad_in_arrays(arg_arrays, net.list_arguments(),
                                     aux_arrays, net.list_auxiliary_states(),
                                     head_grad_arrays, net.list_outputs(),
                                     net.list_inputs())
    new_grad_arrays = g.eval(in_arrays, is_training=True)
    if not isinstance(new_grad_arrays, list):
        new_grad_arrays = [new_grad_arrays]
    assert len(legacy_grad_arrays) == len(new_grad_arrays)
    assert all([np.allclose(r1.asnumpy(), r2.asnumpy(), rtol=1e-5, atol=1e-6)
                for r1, r2 in zip(legacy_grad_arrays, new_grad_arrays)])

def test_simple_eval():
    _test_eval_helper(_conv_block_sym(), _conv_block(), [16, 8, 10, 10])

def test_simple_grad_eval():
    _test_grad_eval_helper(_conv_block_sym(), _conv_block_with_grad(mode=kFullGraph), [16, 8, 10, 10])
    
if __name__ == '__main__':
    test_simple_eval()
    test_simple_grad_eval()
