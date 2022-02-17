from __future__ import annotations
import enum
from types import resolve_bases

from typing import Annotated, Any, Literal, Union, get_args, Type, get_origin
from attr import fields
import attrs
import cattrs


@attrs.define
class A:
    type: Literal[1]
    a: str


@attrs.define
class A2:
    type: Literal[1]
    b: str


@attrs.define
class B:
    type: Literal[2]
    a: str


@attrs.define
class C:
    type: Literal[2, 3]
    subtype: Literal[1, 42]


@attrs.define
class D:
    type: Literal[3]
    subtype: Literal[2]


@attrs.define
class DFafa(D):
    fafa: int


@attrs.define
class E:
    nothing_at_all: int


@attrs.define
class E2:
    also_no_type: str


@attrs.define
class F:
    subtype: Literal[3]


# cattrs.disambiguators.create_literal_field_dis_func(A, A2, B, C, D, DFafa, E, E2, F, minimize_checks=True)


class Op(int, enum.Enum):
    READY = 1
    STEADY = 2


@attrs.define
class Ready:
    op: Literal[Op.READY] = Op.READY
    t: Literal[None] = None


import attr


@attrs.define
class Steady:
    op: Literal[Op.STEADY]


@attrs.define
class Go:
    op: Literal[3]


@attrs.define
class Dispatch:
    op: Literal[42]
    t: str


@attrs.define
class DFoo(Dispatch):
    t: Literal["Foo"]


@attrs.define
class DBar(Dispatch):
    t: Literal["Bar"]


@attrs.define
class DBaz(Dispatch):
    t: Literal["Baz"]


@attrs.define
class DBazMember(DBaz):
    member: str


# d = cattrs.disambiguators.create_literal_field_dis_func(Ready, Steady, Go, DFoo, DBar, DBaz, DBazMember, minimize_checks=True)
# d = cattrs.disambiguators.create_literal_field_dis_func(DFoo, DBar, DBaz, DBazMember, short_circuit=True)

T = Union[Ready, Steady, Go, DFoo, DBar, DBaz, DBazMember]

# c = cattrs.GenConverter(omit_if_default=True, type_overrides={Literal[Op.READY]: cattrs.override(omit_if_default=False)})
c = cattrs.GenConverter(omit_if_default=True)


# cf = cattrs.disambiguators.create_literal_field_dis_func(*get_args(T))
# c.register_structure_hook(T, lambda obj,_: cf(obj)())

from cattr.disambiguators import HACK

def hook(union: type):
    return cattrs.disambiguators.create_literal_field_dis_structure_func(union, converter=c, minimize_checks=True)


c.register_structure_hook_factory(lambda x: not HACK and cattrs.converters.is_union_with_disam_class(x), hook)


def hook2(cl):
    origin = get_origin(cl)
    attribs = fields(origin or cl)
    if attrs.has(cl) and any(isinstance(a.type, str) for a in attribs):
        # PEP 563 annotations - need to be resolved.
        attrs.resolve_types(cl)

    h = cattrs.converters.make_dict_unstructure_fn(
        cl,
        c,
        _cattrs_omit_if_default=c.omit_if_default,
        **{
            field.name: cattrs.override(omit_if_default=False)
            for field in attribs
            if cattrs.converters.is_literal(field.type)
        },
    )

    return h


# c.register_unstructure_hook_factory(
#    attrs.has,
#    hook2
# )

c.register_structure_hook(Ready, lambda x, y: (x, y))

#print(c.structure({"op": 1, "t": None}, Ready))
#print(c.structure({"op": 1, "t": None}, T))
#print(c.structure({"op": 1, "t": None}, T))


#print(Ready().op)

import json

print("JS", json.dumps(c.unstructure(Ready(Op.READY, None))))


from typing import Callable, Tuple, TypeVar

_T = TypeVar("_T")


def __dataclass_transform__(
    *,
    eq_default: bool = True,
    order_default: bool = False,
    kw_only_default: bool = False,
    field_descriptors: Tuple[Union[type, Callable[..., Any]], ...] = (()),
) -> Callable[[_T], _T]:
    return lambda a: a


def wrap(dis={}, **kw):
    def wrapper(cls):
        res = cls
        if dis:
            v = getattr(res, "__test__", {})
            v.update(dis)
            res.__test__ = v
        return res

    return wrapper


@wrap(dis={"type": "foo"})
@attrs.define
class Foo:
    a: int


@wrap(dis={"type": "bar"})
@attrs.define
class Bar:
    a: int


@wrap(dis={"type": "baa"})
@attrs.define
class X(Bar):
    type: str


print("D", dir(X))

Foo(a=49)

print(c.structure({"type": "baa", "a": 42}, Foo | X))

@wrap({"t": "A"})
@attrs.define
class XXA:
    a: int

@wrap({"t": "B"})
@attrs.define
class XXB:
    a: int

from cattr.disambiguators import Fallback

@wrap()
@attrs.define
class XXC:
    b: int


# attrs.resolve_types(XXA)
# attrs.resolve_types(XXB)
# attrs.resolve_types(XXC)

# c.register_structure_hook(
#     XXA | XXC, lambda obj, t: XXA()
# )

# c.register_structure_hook(
#     XXB | XXC, lambda obj, t: XXB()
# )

print(c.structure({"t": "B", "a": 2}, XXA | XXB | XXC))