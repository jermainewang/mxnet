"""Graph APIs."""
from __future__ import absolute_import

import ctypes
import json
import os as _os
import sys as _sys

from .attribute import AttrScope
from .base import _LIB
from .base import check_call, c_array, c_str, mx_uint, py_str
from .base import GraphHandle, SymbolHandle, NDArrayHandle
from .symbol import Symbol, Variable
from .ndarray import NDArray
from .name import NameManager

# Use different version of SymbolBase
# When possible, use cython to speedup part of computation.
try:
    if int(_os.environ.get("MXNET_ENABLE_CYTHON", True)) == 0:
        from ._ctypes.symbol import SymbolBase
    elif _sys.version_info >= (3, 0):
        from ._cy3.symbol import SymbolBase
    else:
        from ._cy2.symbol import SymbolBase
except ImportError:
    if int(_os.environ.get("MXNET_ENFORCE_CYTHON", False)) != 0:
        raise ImportError("Cython Module cannot be loaded but MXNET_ENFORCE_CYTHON=1")
    from ._ctypes.symbol import SymbolBase

def _create_graph_handle(symbol):
    ghandle = GraphHandle()
    check_call(_LIB.MXGraphCreate(
        symbol.handle,
        ctypes.byref(ghandle)))
    return ghandle

class Graph(object):
    def __init__(self, handle, name=None):
        self._handle = handle
        self._name = NameManager.current.get(name, 'graph')
        self._freezed = False

    @property
    def handle(self):
        return self._handle

    @property
    def name(self):
        return self._name

    def freeze(self):
        self._freezed = True
        self.specialize(graph_frozen=1)

    def __del__(self):
        check_call(_LIB.MXGraphFree(self.handle));

    def get_global_attr(self, key):
        ret = ctypes.c_char_p()
        check_call(_LIB.MXGraphGetGlobalAttrJSON(
            self._handle, c_str(key), ctypes.byref(ret)))
        if ret.value is None:
            return None
        else:
            json_str = py_str(ret.value)
            return json.loads(json_str)

    def get_node_attr(self, key):
        ret = ctypes.c_char_p()
        check_call(_LIB.MXGraphGetNodeAttrJSON(
            self._handle, c_str(key), ctypes.byref(ret)))
        if ret.value is None:
            return None
        else:
            json_str = py_str(ret.value)
            return json.loads(json_str)

    def get_node_entry_attr(self, key):
        ret = ctypes.c_char_p()
        check_call(_LIB.MXGraphGetNodeEntryAttrJSON(
            self._handle, c_str(key), ctypes.byref(ret)))
        if ret.value is None:
            return None
        else:
            json_str = py_str(ret.value)
            return json.loads(json_str)

    def specialize(self, **kwargs):
        keys = []
        vals = []
        for k, v in kwargs.items():
            keys.append(c_str(k))
            vals.append(c_str(json.dumps(v)))
        keys = c_array(ctypes.c_char_p, keys)
        vals = c_array(ctypes.c_char_p, vals)
        check_call(_LIB.MXGraphSpecialize(
            self._handle,
            mx_uint(len(keys)),
            keys, vals))

    def transform(self, pass_names, **kwargs):
        assert not self._freezed, \
                'The graph cannot be changed after a GraphSymbol is created.'
        passes = [c_str(n) for n in pass_names]
        passes = c_array(ctypes.c_char_p, passes)
        keys = []
        vals = []
        for k, v in kwargs.items():
            keys.append(c_str(k))
            vals.append(c_str(json.dumps(v)))
        keys = c_array(ctypes.c_char_p, keys)
        vals = c_array(ctypes.c_char_p, vals)
        out = GraphHandle()
        check_call(_LIB.MXGraphTransform(
            self._handle,
            mx_uint(len(passes)),
            passes,
            mx_uint(len(keys)),
            keys, vals,
            ctypes.byref(out)))
        return Graph(out)

    def create_input_arrays(self):
        array_hdls = ctypes.POINTER(NDArrayHandle)()
        num_inputs = ctypes.c_int(0)
        check_call(_LIB.MXGraphCreateInputArrays(
            self._handle,
            ctypes.byref(num_inputs),
            ctypes.byref(array_hdls)))
        if num_inputs.value == 1:
            return NDArray(ctypes.cast(array_hdls[0], NDArrayHandle))
        else:
            return [NDArray(ctypes.cast(array_hdls[i], NDArrayHandle))
                    for i in range(num_inputs.value)]

    def create_output_arrays(self):
        array_hdls = ctypes.POINTER(NDArrayHandle)()
        num_outputs = ctypes.c_int(0)
        check_call(_LIB.MXGraphCreateOutputArrays(
            self._handle,
            ctypes.byref(num_outputs),
            ctypes.byref(array_hdls)))
        if num_outputs.value == 1:
            return NDArray(ctypes.cast(array_hdls[0], NDArrayHandle))
        else:
            return [NDArray(ctypes.cast(array_hdls[i], NDArrayHandle))
                    for i in range(num_outputs.value)]

    def eval(self, inputs):
        output_handles = ctypes.POINTER(NDArrayHandle)()
        num_outputs = ctypes.c_int(0)
        check_call(_LIB.MXGraphEval(
            self._handle,
            ctypes.c_int(len(inputs)),
            c_array(NDArrayHandle, [arr.handle for arr in inputs]),
            ctypes.byref(num_outputs),
            ctypes.byref(output_handles)))
        if num_outputs.value == 1:
            return NDArray(ctypes.cast(output_handles[0], NDArrayHandle))
        else:
            return [NDArray(ctypes.cast(output_handles[i], NDArrayHandle))
                    for i in range(num_output.value)]

def create(symbol, name=None):
    handle = _create_graph_handle(symbol)
    return Graph(handle, name)

def symbolize(graph):
    """Currently graph symbol only allows keyword arguments for composition."""
    def _graph_symbol_creator(name=None, attr=None, **kwargs):
        kwargs.update(AttrScope.current.get(attr))
        keys = []
        vals = []
        sym_kwargs = dict()
        for k, v in kwargs.items():
            if isinstance(v, SymbolBase):
                sym_kwargs[k] = v
            else:
                keys.append(c_str(k))
                vals.append(c_str(str(v)))

        sym_handle = SymbolHandle()
        keys = c_array(ctypes.c_char_p, keys)
        vals = c_array(ctypes.c_char_p, vals)
        check_call(_LIB.MXSymbolCreateGraphSymbol(
            graph.handle,
            mx_uint(len(keys)),
            keys, vals,
            ctypes.byref(sym_handle)))

        ret = Symbol(sym_handle)
        name = NameManager.current.get(name, graph.name)
        ret._compose(name=name, **sym_kwargs)
        return ret
    graph.freeze()
    return _graph_symbol_creator
