"""Graph APIs."""
from __future__ import absolute_import

import ctypes

from .attribute import AttrScope
from .base import _LIB
from .base import check_call, c_array, c_str, mx_uint
from .base import GraphHandle, SymbolHandle
from .symbol import Symbol
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

class Graph(object):
    def __init__(self, handle, name):
        self._handle = handle
        self._name = name

    @property
    def handle(self):
        return self._handle

    @property
    def name(self):
        return self._name

    def __del__(self):
        check_call(_LIB.MXGraphFree(self.handle));

def create(symbol, name=None):
    ghandle = GraphHandle()
    check_call(_LIB.MXGraphCreate(
        ctypes.c_void_p(symbol.handle),
        ctypes.byref(ghandle)))
    name = NameManager.current.get(name, 'graph')
    return Graph(ghandle, name)

def symbolize(graph):
    def _graph_symbol_creator(inputs=None, name=None, attr=None, **kwargs):
        #assert len(inputs) == graph.num_inputs()
        if inputs:
            assert all([isinstance(ele, SymbolBase) for ele in inputs]), \
                'Argument "inputs" must be a list of symbols.'

        kwargs.update(AttrScope.current.get(attr))
        keys = []
        vals = []
        for k, v in kwargs.items():
            assert not isinstance(v, SymbolBase), \
                'Graph symbol does not allow symbol type kwargs.'
            keys.append(c_str(k))
            vals.append(c_str(v))

        sym_handle = SymbolHandle()
        keys = c_array(ctypes.c_char_p, keys)
        vals = c_array(ctypes.c_char_p, vals)
        check_call(_LIB.MXSymbolCreateGraphSymbol(
            ctypes.c_void_p(graph.handle),
            mx_uint(len(keys)),
            keys, vals,
            ctypes.byref(sym_handle)))

        ret = Symbol(sym_handle)
        name = NameManager.current.get(name, graph.name)
        if inputs:
            ret._compose(name=name)
        else:
            ret._compose(*inputs, name=name)
        return ret
    return _graph_symbol_creator
