from enum import IntEnum

import attr
import pytest

from cattr import Converter, GenConverter, UnstructureStrategy


class E(IntEnum):
    ONE = 1
    TWO = 2


@attr.define
class C:
    a: int
    b: float
    c: str
    d: bytes
    e: E
    f: int
    g: float
    h: str
    i: bytes
    j: E
    k: int
    l: float
    m: str
    n: bytes
    o: E
    p: int
    q: float
    r: str
    s: bytes
    t: E
    u: int
    v: float
    w: str
    x: bytes
    y: E
    z: int
    aa: float
    ab: str
    ac: bytes
    ad: E


@pytest.mark.parametrize("converter_cls", [Converter, GenConverter])
@pytest.mark.parametrize(
    "unstructure_strat",
    [UnstructureStrategy.AS_DICT, UnstructureStrategy.AS_TUPLE],
)
def test_unstructure_attrs_primitives(
    benchmark, converter_cls, unstructure_strat
):
    """Benchmark a large (30 attributes) attrs class containing primitives."""

    c = converter_cls(unstruct_strat=unstructure_strat)

    benchmark(
        c.unstructure,
        C(
            1,
            1.0,
            "a small string",
            "test".encode(),
            E.ONE,
            2,
            2.0,
            "a small string",
            "test".encode(),
            E.TWO,
            3,
            3.0,
            "a small string",
            "test".encode(),
            E.ONE,
            4,
            4.0,
            "a small string",
            "test".encode(),
            E.TWO,
            5,
            5.0,
            "a small string",
            "test".encode(),
            E.ONE,
            6,
            6.0,
            "a small string",
            "test".encode(),
            E.TWO,
        ),
    )


@pytest.mark.parametrize("converter_cls", [Converter, GenConverter])
@pytest.mark.parametrize(
    "unstructure_strat",
    [UnstructureStrategy.AS_DICT, UnstructureStrategy.AS_TUPLE],
)
def test_structure_attrs_primitives(
    benchmark, converter_cls, unstructure_strat
):
    """Benchmark a large (30 attributes) attrs class containing primitives."""

    c = converter_cls(unstruct_strat=unstructure_strat)

    inst = C(
        1,
        1.0,
        "a small string",
        "test".encode(),
        E.ONE,
        2,
        2.0,
        "a small string",
        "test".encode(),
        E.TWO,
        3,
        3.0,
        "a small string",
        "test".encode(),
        E.ONE,
        4,
        4.0,
        "a small string",
        "test".encode(),
        E.TWO,
        5,
        5.0,
        "a small string",
        "test".encode(),
        E.ONE,
        6,
        6.0,
        "a small string",
        "test".encode(),
        E.TWO,
    )

    raw = c.unstructure(inst)

    benchmark(c.structure, raw, C)
