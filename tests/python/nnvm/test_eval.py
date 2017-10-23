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

################################################################################################
def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax')

def _resnet_symbol(num_classes, num_layers, image_shape, conv_workspace=256, **kwargs):
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    (nchannel, height, width) = image_shape
    if height <= 28:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))

    return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)

################################################################################################
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

def _test_grad_eval_helper(net, g, data_shape, need_head_grad=False):
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
    if need_head_grad:
        np_head_grad_arrays = [rng.uniform(-0.1, 0.1, shape).astype(np.float32)
                               for shape in out_shapes]
        head_grad_arrays = [mx.nd.array(nparr, ctx=ctx) for nparr in np_head_grad_arrays]
        executor.backward(head_grad_arrays)
    else:
        head_grad_arrays = []
        executor.backward()
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

################################################################################################
def test_simple_eval():
    _test_eval_helper(_conv_block_sym(),
                      _conv_block(),
                      [16, 8, 10, 10],
                      need_head_grad=True)

def test_simple_grad_eval():
    _test_grad_eval_helper(_conv_block_sym(),
                           _conv_block_with_grad(mode=kFullGraph),
                           [16, 8, 10, 10],
                           need_head_grad=True)

def test_resnet():
    data_shape = (3, 224, 224)
    num_classes = 1000
    resnet_sym = _resnet_symbol(num_classes, 152, data_shape)
    resnet_grf = graph.create(resnet_sym)
    _test_eval_helper(resnet_sym,
                      resnet_grf,
                      [16, 3, 224, 224])
    
if __name__ == '__main__':
    #test_simple_eval()
    #test_simple_grad_eval()
    test_resnet()
