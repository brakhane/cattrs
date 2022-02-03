from cattrs import (
    converters,
    disambiguators,
    dispatch,
    errors,
    gen,
    global_converter,
)
from cattrs.converters import Converter, GenConverter, UnstructureStrategy
from cattrs.gen import override

__all__ = (
    "Converter",
    "converters",
    "disambiguators",
    "dispatch",
    "errors",
    "gen",
    "GenConverter",
    "global_converter",
    "override",
    "preconf",
    "register_structure_hook_func",
    "register_structure_hook",
    "register_unstructure_hook_func",
    "register_unstructure_hook",
    "structure_attrs_fromdict",
    "structure_attrs_fromtuple",
    "structure",
    "unstructure",
    "UnstructureStrategy",
)


unstructure = global_converter.unstructure
structure = global_converter.structure
structure_attrs_fromtuple = global_converter.structure_attrs_fromtuple
structure_attrs_fromdict = global_converter.structure_attrs_fromdict
register_structure_hook = global_converter.register_structure_hook
register_structure_hook_func = global_converter.register_structure_hook_func
register_unstructure_hook = global_converter.register_unstructure_hook
register_unstructure_hook_func = (
    global_converter.register_unstructure_hook_func
)
