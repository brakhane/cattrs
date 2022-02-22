# from __future__ import annotations
from dataclasses import dataclass
import enum
from types import resolve_bases

from typing import Annotated, Any, Literal, Union, get_args, Type, get_origin
from attr import fields, define
import attrs
import cattrs
from cattr.annotations import Fallback, has_discriminator


if False:

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

    @attrs.define(kw_only=True)
    class Dispatch:
        op: Literal[42] = 42
        t: str

    @attrs.define
    class DFoo(Dispatch):
        t: Literal["Foo"]

    @attrs.define
    class DBar(Dispatch):
        t: Literal["Bar"]

    @attrs.define(kw_only=True)
    class DBaz(Dispatch):
        t: Literal["Baz"] = "Baz"

    @attrs.define
    class DBazMember(DBaz):
        member: str

    # d = cattrs.disambiguators.create_literal_field_dis_func(DFoo, DBar, DBaz, DBazMember, short_circuit=True)

    T = Union[Ready, Steady, Go, DFoo, DBar, DBaz, DBazMember]

    # c = cattrs.GenConverter(omit_if_default=True, type_overrides={Literal[Op.READY]: cattrs.override(omit_if_default=False)})
    from cattrs.converters import UnstructureStrategy
    from cattrs.gen import override

    c = cattrs.GenConverter(
        omit_if_default=True,
        forbid_extra_keys=True,
        type_overrides={Literal: override(omit_if_default=False)},
    )

    x = c.structure(
        {"op": 42, "t": "Baz", "member": "foo"},
        Union[Ready, Steady, Go, DFoo, DBar, DBaz, DBazMember],
    )

    print(f"{x=}")

    print(c.unstructure(x))

    # cf = cattrs.disambiguators.create_literal_field_dis_func(*get_args(T))
    # c.register_structure_hook(T, lambda obj,_: cf(obj)())

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

    # print(c.structure({"op": 1, "t": None}, Ready))
    # print(c.structure({"op": 1, "t": None}, T))
    # print(c.structure({"op": 1, "t": None}, T))

    # print(Ready().op)

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

    @has_discriminator({"type": "foo"})
    @attrs.define
    class Foo:
        a: int

    @has_discriminator({"type": "bar"})
    @attrs.define
    class Bar:
        a: int

    # @has_discriminator({"type": "X"})
    @attrs.define
    class X(Bar):
        b: int

    print(c.structure({"type": "bar", "a": 99, "b": 42}, Foo | Bar | X))

    @attr.define
    class XXA:
        a: Literal[1]

    @attr.define
    class XXB:
        a: Literal[2]

    from cattr.annotations import Fallback

    @attr.define
    class XXC:
        a: Annotated[int, Fallback]

    # attrs.resolve_types(XXA)
    # attrs.resolve_types(XXB)
    # attrs.resolve_types(XXC)

    # c.register_structure_hook(
    #     XXA | XXC, lambda obj, t: XXA()
    # )

    # c.register_structure_hook(
    #     XXB | XXC, lambda obj, t: XXB()
    # )

    print(c.structure({"a": 42}, XXA | XXB | XXC))

    import dataclasses

    @attrs.define
    class DCA:
        t: Literal[1]
        data: str

    @dataclasses.dataclass
    class DCB:
        t: Literal[2]
        data: str

    @attrs.define
    class DCA2:
        t: Literal[1]
        data: str
        more_data: str

    print(
        c.structure(
            {"t": 1, "data": "foo", "more_data": "MD"}, DCA | DCB | DCA2
        )
    )

    @has_discriminator({"_type": "ClassA"})
    @define
    class ClassA:
        a_string: str

    @has_discriminator()  # {"_type": "ClassB"})
    @define
    class ClassB:
        a_string: str


# print(c.structure({"_type": "ClassA", "a_string": "foo"}, ClassA | ClassB))


@has_discriminator({"type": "person"})
@define
class Person:
    #    type: Literal["person"]
    name: str


@has_discriminator({"type": "animal", "subtype": "dog"})
@define
class Dog:
    #    type: Literal["animal"]
    #    subtype: Literal["dog"]
    name: str
    owner: Person

    def speak(self):
        return "Woof!"


@has_discriminator({"type": "animal", "subtype": "cat"})
@define
class Cat:
    #    type: Literal["animal"]
    #    subtype: Literal["cat"]
    name: str
    owner: Person

    def speak(self):
        return "Meow!"


c = cattrs.Converter(
    #    omit_if_default=True,
    #    forbid_extra_keys=True,
    #    type_overrides={Literal: override(omit_if_default=False)},
)

attrs.resolve_types(Dog)
attrs.resolve_types(Cat)
attrs.resolve_types(Person)

print(
    c.structure(
        {
            "type": "animal",
            "subtype": "cat",
            "name": "Garfield",
            "owner": {"type": "person", "name": "John"},
        },
        Person | Dog | Cat,
    )
)


@dataclass
class DC:
    a: int
    b: str


print(c.unstructure(DC(a=42, b="foo")))
print(c.structure({"a": 42, "b": "foo"}, DC))
print(c.structure(2, Literal[2]))
