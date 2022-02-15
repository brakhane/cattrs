from __future__ import annotations
import enum

from typing import Literal, Union, get_args
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
    subtype: Literal[1, 3, 42]

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



#cattrs.disambiguators.create_literal_field_dis_func(A, A2, B, C, D, DFafa, E, E2, F, short_circuit=True)


class Op(int, enum.Enum):
    READY = 1
    STEADY = 2

@attrs.define
class Ready:
    op: Literal[Op.READY]
    t: Literal[None]

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



d = cattrs.disambiguators.create_literal_field_dis_func(Ready, Steady, Go, DFoo, DBar, DBaz, DBazMember, short_circuit=True)
#d = cattrs.disambiguators.create_literal_field_dis_func(DFoo, DBar, DBaz, DBazMember, short_circuit=True)

T = Union[Ready, Steady, Go, DFoo, DBar, DBaz, DBazMember]

#conv = cattrs.disambiguators.create_literal_field_dis_func(*get_args(T))

#cattrs.register_structure_hook(T, lambda obj,_: conv(obj) )


print(cattrs.structure({"op": 1, "t_": None, "m_ember": "foo"}, T))




