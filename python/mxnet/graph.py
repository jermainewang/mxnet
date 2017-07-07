"""Graph APIs."""
from __future__ import absolute_import

import ctypes
import os as _os
import sys as _sys

from .attribute import AttrScope
from .base import _LIB
from .base import check_call, c_array, c_str, mx_uint
from .base import GraphHandle, SymbolHandle
from .symbol import Symbol, Variable
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

def _fill_missing_symbol_inputs(graph, sym_name, provided_args, provided_kwargs):
    if provided_args is None:
        # No need to fill in the kwargs from args.
        return
    required_args = graph.list_arguments()
    for i, name in enumerate(required_args):
        if i < len(provided_args):
            assert not name in provided_kwargs, \
                'Argument "%s" is specified twice' % name
            provided_kwargs[name] = provided_args[i]

class Graph(object):
    def __init__(self, symbol, name=None):
        self._symbol = symbol
        self._name = NameManager.current.get(name, 'graph')
        self._handle = _create_graph_handle(symbol)

    @property
    def handle(self):
        return self._handle

    @property
    def name(self):
        return self._name

    def __del__(self):
        check_call(_LIB.MXGraphFree(self.handle));

    def list_arguments(self):
        return self._symbol.list_arguments()

def symbolize(graph):
    def _graph_symbol_creator(sym_inputs=None, name=None, attr=None, **kwargs):
        if sym_inputs is not None:
            if not isinstance(sym_inputs, list):
                sym_inputs = [sym_inputs]
            assert all([isinstance(ele, SymbolBase) for ele in sym_inputs]), \
                'Argument "sym_inputs" must be a list of symbols.'

        kwargs.update(AttrScope.current.get(attr))
        keys = []
        vals = []
        sym_kwargs = dict()
        for k, v in kwargs.items():
            if isinstance(v, SymbolBase):
                sym_kwargs[k] = v
            else:
                keys.append(c_str(k))
                vals.append(c_str(v))

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
        _fill_missing_symbol_inputs(graph, name, sym_inputs, sym_kwargs)
        ret._compose(name=name, **sym_kwargs)
        return ret
    return _graph_symbol_creator
