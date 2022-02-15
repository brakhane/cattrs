"""Utilities for union (sum type) disambiguation."""
from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import product
from operator import or_
from typing import (  # noqa: F401, imported for Mypy.
    Any,
    Callable,
    Dict,
    Literal,
    Mapping,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
    get_args,
)

from attr import NOTHING, fields, resolve_types

from cattr._compat import get_origin


def create_uniq_field_dis_func(*classes: Type) -> Callable:
    """Given attr classes, generate a disambiguation function.

    The function is based on unique fields."""
    if len(classes) < 2:
        raise ValueError("At least two classes required.")
    cls_and_attrs = [
        (cl, set(at.name for at in fields(get_origin(cl) or cl)))
        for cl in classes
    ]
    if len([attrs for _, attrs in cls_and_attrs if len(attrs) == 0]) > 1:
        raise ValueError("At least two classes have no attributes.")
    # TODO: Deal with a single class having no required attrs.
    # For each class, attempt to generate a single unique required field.
    uniq_attrs_dict = OrderedDict()  # type: Dict[str, Type]
    cls_and_attrs.sort(key=lambda c_a: -len(c_a[1]))

    fallback = None  # If none match, try this.

    for i, (cl, cl_reqs) in enumerate(cls_and_attrs):
        other_classes = cls_and_attrs[i + 1 :]
        if other_classes:
            other_reqs = reduce(or_, (c_a[1] for c_a in other_classes))
            uniq = cl_reqs - other_reqs
            if not uniq:
                m = "{} has no usable unique attributes.".format(cl)
                raise ValueError(m)
            # We need a unique attribute with no default.
            cl_fields = fields(get_origin(cl) or cl)
            for attr_name in uniq:
                if getattr(cl_fields, attr_name).default is NOTHING:
                    break
            else:
                raise ValueError(f"{cl} has no usable non-default attributes.")
            uniq_attrs_dict[attr_name] = cl
        else:
            fallback = cl

    def dis_func(data):
        # type: (Mapping) -> Optional[Type]
        if not isinstance(data, Mapping):
            raise ValueError("Only input mappings are supported.")
        for k, v in uniq_attrs_dict.items():
            if k in data:
                return v
        return fallback

    return dis_func


def _is_literal_type(t: Type) -> bool:
    return get_origin(t) is Literal

def _get_literal_args(t: Type[Any]) -> Tuple[Any]:
    return get_args(t)


class AnythingType():
    def __repr__(self):
        return "<anything>"

Anything = AnythingType()

def create_literal_field_dis_func(*classes: Type) -> Callable:
    if len(classes) < 2:
        raise ValueError("At least two classes required.")


    litfields: Dict[str, Dict[Any, Set[Type]]] = defaultdict(lambda: defaultdict(set[Type]))


    # First, find all fields that are defined as Literal in at least one class

    for cls in classes:
        resolve_types(cls) # returns immediately if already resolved, so not much overhead if not needed
        for field in fields(get_origin(cls) or cls):
            if _is_literal_type(field.type):
                for value in _get_literal_args(cast(Type[Any], field.type)):
                    litfields[field.name][value].add(cls)

    best_fields = list(litfields.keys())
    best_fields.sort(key=lambda k: len(litfields[k]), reverse=True)

    print("best", best_fields)

    # Now we need to check if some classes do not define a Literal for a specific field, in which
    # case they always match. Note that "not defining a Literal" can mean not defining the field
    # at all, or defining it as some arbitrary type.

    for name, valuemap in litfields.copy().items():
        wild_classes = set(classes)
        for clss in valuemap.values():
            for cls in clss:
                wild_classes.discard(cls)
        if wild_classes:
            for clss in valuemap.values():
                clss.update(wild_classes)
            litfields[name][Anything] = wild_classes.copy()
        else:
            break



    import graphviz
    dot = graphviz.Digraph(strict="true")

    dot.attr("node", shape="box")
    dot.attr("graph", ranksep="1.5", searchsize="500", mclimit="10", newrank="true", concentrate="true")
    #dot.attr("graph", splines="ortho")




    def fmt(x):
        nonlocal classes

        #return '\n'.join(sorted(cls.__name__ for cls in x))
        return "<" + '<br/>'.join(f"<font color='gray'>{cls.__name__}</font>" if cls not in x else cls.__name__ for cls in sorted(classes, key=lambda x:x.__name__)) + ">"

    idxhack = []

    def mktree(fns: list[str], classes: set[Type]):
        dot.node(str(idxhack), label=fmt(classes))
        if not fns:
            return classes

            if len(classes) > 1:
                return create_uniq_field_dis_func(*classes)
            else:
                return next(iter(classes))
        else:
            tree = {}
            for val, classes_ in litfields[fns[0]].items():
                idxhack.append(val)
                union = classes_ & classes


                short_circuit = True

                if short_circuit:
                    if len(union) > 1:
                        tree[val] = mktree(fns[1:], union)
                    else:
                        tree[val] = union
                        dot.node(str(idxhack), label=fmt(union))
                else:
                    if union:
                        tree[val] = mktree(fns[1:], union)

                idxhack.pop()
                if union:
                    dot.edge(str(idxhack), str(idxhack+[val]), label=f'{fns[0]}\n{val}')
            return tree







    from pprint import pprint
    mktree(best_fields, set(classes))
    pprint(litfields)

    dot.view()


    #pprint(litfields)



