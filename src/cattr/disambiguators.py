"""Utilities for union (sum type) disambiguation."""
from collections import OrderedDict, defaultdict
from enum import Enum
from functools import reduce, cache
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
)

from attr import NOTHING, fields, resolve_types, define

from cattr._compat import get_origin, get_args

from pprint import pprint

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

    dis_func.__qualname__ = f"create_uniq_field_dis_func.<generated>.dis_{'_'.join(cls.__qualname__ for cls in classes)}"

    return dis_func


def _is_literal_type(t: Type) -> bool:
    return get_origin(t) is Literal


def _get_literal_args(t: Type[Any]) -> Tuple[Any]:
    return get_args(t)


class _AnythingType:
    def __repr__(self):
        return "<anything>"

Anything = _AnythingType()


def _enum_value(x: Any) -> Any:
    return x.value if isinstance(x, Enum) else x


def create_literal_field_dis_func(
    *classes: Type, short_circuit=True
) -> Callable:
    if len(classes) < 2:
        raise ValueError("At least two classes required.")

    litfields: Dict[str, Dict[Any, Set[Type]]] = defaultdict(
        lambda: defaultdict(set[Type])
    )

    # First, find all fields that are defined as Literal in at least one class

    for cls in classes:
        resolve_types(
            cls
        )  # returns immediately if already resolved, so not much overhead if not needed
        for field in fields(get_origin(cls) or cls):
            if _is_literal_type(field.type):
                for value in _get_literal_args(cast(Type[Any], field.type)):
                    litfields[field.name][_enum_value(value)].add(cls)

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
            litfields[name][Anything] = wild_classes

    best_fields = list(litfields.keys())
    print(litfields)
    if short_circuit:
        # best_fields.sort(key=lambda k: len(litfields[k]), reverse=True)
        best_fields.sort(
            key=lambda k: sum(len(v) ** 2 for v in litfields[k].values()),
            reverse=False,
        )

    print("lit")
    pprint(litfields)
    print("best", best_fields)

    import graphviz

    dot = graphviz.Digraph(strict=False)

    dot.attr("node", shape="box")
    for a in ["node", "edge"]:
        dot.attr(a, fontname="Fira Code")
    dot.attr(
        "graph",
        ranksep="1.5",
        searchsize="500",
        mclimit="10",
        newrank="true",
        concentrate="true",
    )
    # dot.attr("graph", splines="ortho")

    def fmt(x):
        nonlocal classes

        # return '\n'.join(sorted(cls.__name__ for cls in x))
        return (
            "<"
            + '<br/>'.join(
                f"<font color='gray'>{cls.__name__}</font>"
                if cls not in x
                else cls.__name__
                for cls in sorted(classes, key=lambda x: x.__name__)
            )
            + ">"
        )

    idxhack = []

    cached_create_uniq_field_dis_func = cache(create_uniq_field_dis_func)

    class DisambiguationError(RuntimeError):
        pass

    def mktree(fns: list[str], classes: set[Type]):
        if not classes:
            return None
        dot.node(str(idxhack), label=fmt(classes))
        if len(classes) == 1 and short_circuit:
            return next(iter(classes))

        if not fns:
            if len(classes) > 1:
                try:
                    return cached_create_uniq_field_dis_func(*classes)
                except ValueError as e:
                    raise DisambiguationError(e, classes)
            else:
                return next(iter(classes))
        else:
            tree = {}
            fn, *tail = fns
            for val, classes_ in litfields[fn].items():
                idxhack.append(val)
                union = classes_ & classes

                try:
                    if not short_circuit:
                        tree[val] = mktree(tail, union)
                    else:
                        if union:
                            tree[val] = mktree(tail, union)
                except DisambiguationError as e:
                    raise DisambiguationError(*e.args, (fn, repr(val)))

                idxhack.pop()
                if union:
                    dot.edge(
                        str(idxhack),
                        str(idxhack + [val]),
                        label=fr"{fn}\n{val}",
                    )
            return tree

    try:
        tree = mktree(best_fields, set(classes))
    except DisambiguationError as e:
        orig_e, classes, *params = e.args
        params_str = ", ".join(f"{k}={v}" for k, v in params)
        raise ValueError(
            f"Unable to disambiguate between classes {classes} when {params_str}: {orig_e.args[0]}"
        ) from orig_e


    pprint(tree)
    dot.view()

    def dis_func(data: Mapping) -> Optional[Type]:
        if not isinstance(data, Mapping):
            raise ValueError("Only input mappings are supported.")

        def recurse(i, tree):
            try:
                fn = best_fields[i]
            except IndexError:
                return None
            val = data.get(fn, Anything)
            try:
                subtree = tree[val]
            except KeyError:
                try:
                    subtree = tree[Anything]
                except KeyError:
                    # No valid match found
                    return None
            if isinstance(
                subtree, type
            ):  # FIXME: is this always true, even for classes with custom metaclasses?
                return subtree
            elif isinstance(subtree, Callable):
                return subtree(data)
            else:
                return recurse(i + 1, subtree)

        return recurse(0, tree)

    return dis_func
