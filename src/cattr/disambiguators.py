"""Utilities for union (sum type) disambiguation."""
from collections import OrderedDict, defaultdict
from enum import Enum
from functools import reduce, cache
from operator import or_
from typing import (  # noqa: F401, imported for Mypy.
    Annotated,
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
    get_type_hints,
    TYPE_CHECKING,
)

from attr import NOTHING, fields, has, resolve_types

from cattr._compat import get_origin, get_args, is_annotated, is_literal

from cattr.errors import StructureHandlerNotFoundError


if TYPE_CHECKING:
    from cattr.converters import Converter


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

    # dis_func.__qualname__ = f"create_uniq_field_dis_func.<generated>.dis_{'_'.join(cls.__qualname__ for cls in classes)}"

    return dis_func



class _AnythingType:
    def __repr__(self):
        return "<anything>"


# Internal marker standing for "field can have any value, including
# being absent"
_ANYTHING = _AnythingType()

Fallback = object()


class FOO(RuntimeError): pass

HACK = []

def create_literal_field_dis_structure_func(
    union: Type, converter: 'Converter', minimize_checks=True
):
    classes = get_args(union)

    if len(classes) < 2:
        raise ValueError("A union cannot have a single type")


    disfields: Dict[str, Dict[Any, Set[Type]]] = defaultdict(
        lambda: defaultdict(set[Type])
    )

    fallback_only: Dict[str, Set[Type]] = defaultdict(set)

    # First, find all fields that can be used for disambiguation
    # These are either user defined via FIXME
    # or, in "auto mode", are fields defined as Literal in at least one class

    for cls in classes:
        disam_config = getattr(cls, "__test__", None)
        if disam_config:
            for k, v in disam_config.items():
                disfields[k][v].add(cls)
        else:
            hints = get_type_hints(get_origin(cls) or cls, include_extras=True)
            for name, typ in hints.items():
                if is_annotated(typ):
                    typ, *metadata = get_args(typ)
                    if any(md is Fallback for md in metadata):
                        fallback_only[name].add(cls)

                if is_literal(typ):
                    for value in get_args(typ):
                        disfields[name][
                            value.value if isinstance(value, Enum) else value
                        ].add(cls)

    if not disfields:
        raise ValueError(f"No disambiguation fields found in any of the classes {classes}")

    # Now we need to check if some classes do not define a disambiguation value
    # for a specific field, in which case they always match.
    # Note that "not defining a value" can mean not defining the field at all,
    # or defining it as some arbitrary type.

    for name, valuemap in disfields.copy().items():
        wild_classes = set(classes)
        for clss in valuemap.values():
            for cls in clss:
                wild_classes.discard(cls)
        if wild_classes:
            for clss in valuemap.values():
                clss.update(wild_classes - fallback_only[name])
            disfields[name][_ANYTHING] = wild_classes

    # if there is more than one literal fields, we might be able
    # to reduce the number of checks by checking the fields
    # that disambiguate the most.
    # The heuristic to go by sum([number of classes per value]^2) seems
    # to be a relatively good one based on my experiments.
    #
    # If we are checking every field value anyway, we don't need
    # to find an "optimal" order, and we just go with defintion order

    best_field_names = list(disfields.keys())
    if minimize_checks:
        best_field_names.sort(
            key=lambda k: sum(len(v) ** 2 for v in disfields[k].values()),
            reverse=False,
        )

    class DisambiguationError(RuntimeError):
        """
        Internal exception raised by mktree to signify an error
        and collect status information as it bubbles up the
        stack.

        args are original_exception, set_of_classes,
        [(fieldname, fieldvalue) for each frame]
        """

        pass

    def mktree(names: list[str], classes: set[Type]):
        global HACK

        if not classes:
            return None


        if (len(classes) == 1 and minimize_checks) or (not names):
            HACK.append(None)
            try:
                it = iter(classes)
                t = next(it)
                for cl in it:
                    t = Union[t, cl]
                func = converter._structure_func.dispatch(t)
                if func == converter._structure_error:
                    raise RuntimeError("No structure hook for this type exists.")
                return func
            except Exception as e:
                raise DisambiguationError(e, classes) from None
            finally:
                HACK.pop()

        else:
            tree = {}
            name, *tail = names
            for val, classes_ in disfields[name].items():
                union = classes_ & classes

                try:
                    if not minimize_checks:
                        tree[val] = mktree(tail, union)
                    else:
                        if union:
                            tree[val] = mktree(tail, union)
                except DisambiguationError as e:
                    raise DisambiguationError(
                        *e.args, (name, val)
                    ) from None

            return tree


    try:
        tree = mktree(best_field_names, set(classes))
    except DisambiguationError as e:
        # convert the internal exception into a ValueError with
        # useful error message
        orig_e, classes, *params = e.args
        params_str = ", ".join(f"{k!r}={v!r}" for k, v in params)

        if len(classes) > 1:
            error_msg = (
                f"Unable to disambiguate between types {classes} when "
                f"{params_str}: {orig_e.args[0]}\n"
                f"Hint: register a structure hook for Union[{', '.join(cls.__name__ for cls in classes)}]."
            )
        else:
            error_msg = (
                f"Unable to structure type {next(iter(classes))} when "
                f"{params_str}: {orig_e.args[0]}"
            )


        raise ValueError(error_msg) from orig_e

    def dis_func(data: Mapping, typ: Type) -> Optional[Type]:
        if not isinstance(data, Mapping):
            raise ValueError("Only input mappings are supported.")

        class NotFound(RuntimeError):
            pass

        def recurse(i, tree):
            try:
                fn = best_field_names[i]
            except IndexError:
                raise NotFound()
            val = data.get(fn, _ANYTHING)
            try:
                subtree = tree[val]
            except KeyError:
                try:
                    subtree = tree[_ANYTHING]
                except KeyError:
                    if val is _ANYTHING:
                        raise NotFound(f"{fn!r} is missing")
                    else:
                        raise NotFound(f"{fn!r}={val!r}")
            if isinstance(subtree, Callable):
                assert not isinstance(subtree, type)
                xx=subtree(data, typ)
                print(f"{subtree=} {data=} {xx=}")
                return xx
            else:
                try:
                    return recurse(i + 1, subtree)
                except NotFound as e:
                    raise NotFound(f"{fn!r}={val!r}", *e.args)

        try:
            return recurse(0, tree)
        except NotFound as e:
            raise ValueError(f"No class matches {', '.join(e.args)}") from None

    # dis_func.__qualname__ = f"create_literal_field_dis_func.<generated>.dis_{'_'.join(cls.__qualname__ for cls in classes)}"

    return dis_func
