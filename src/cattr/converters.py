from collections import Counter, defaultdict
from collections.abc import MutableSet as AbcMutableSet
from dataclasses import Field
from enum import Enum
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    get_args,
    get_type_hints,
)

from attr import Attribute
from attr import has as attrs_has
from attr import resolve_types

from ._compat import (
    FrozenSetSubscriptable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
    Set,
    fields,
    get_origin,
    has,
    has_with_generic,
    is_annotated,
    is_bare,
    is_counter,
    is_frozenset,
    is_generic,
    is_generic_attrs,
    is_hetero_tuple,
    is_literal,
    is_mapping,
    is_mutable_set,
    is_protocol,
    is_sequence,
    is_tuple,
    is_union_type,
)
from .annotations import Fallback
from .disambiguators import create_uniq_field_dis_func
from .dispatch import MultiStrategyDispatch
from .errors import StructureHandlerNotFoundError
from cattrs.gen import (
    AttributeOverride,
    make_dict_structure_fn,
    make_dict_unstructure_fn,
    make_hetero_tuple_unstructure_fn,
    make_iterable_unstructure_fn,
    make_mapping_structure_fn,
    make_mapping_unstructure_fn,
)

NoneType = type(None)
T = TypeVar("T")
V = TypeVar("V")


class UnstructureStrategy(Enum):
    """`attrs` classes unstructuring strategies."""

    AS_DICT = "asdict"
    AS_TUPLE = "astuple"


def _subclass(typ):
    """a shortcut"""
    return lambda cls: issubclass(cls, typ)


def is_discriminated_class(typ):
    return isinstance(typ, type) and hasattr(typ, "__cattr_discriminators__")


def is_class_with_literal_types(typ):
    return isinstance(typ, type) and any(
        is_literal(t) for t in get_type_hints(typ).values()
    )


def is_union_with_some_discriminated_or_literal_classes(typ):
    return is_union_type(typ) and any(
        is_discriminated_class(get_origin(e) or e)
        or is_class_with_literal_types(get_origin(e) or e)
        for e in typ.__args__
    )


def is_attrs_union(typ):
    return is_union_type(typ) and all(
        has(get_origin(e) or e) for e in typ.__args__
    )


def is_attrs_union_or_none(typ):
    return is_union_type(typ) and all(
        e is NoneType or has(get_origin(e) or e) for e in typ.__args__
    )


def is_optional(typ):
    return (
        is_union_type(typ)
        and NoneType in typ.__args__
        and len(typ.__args__) == 2
    )


class _DisambiguationError(RuntimeError):
    """
    Internal exception raised by
    _literal_field_dis_structure_hook_factory.mktree to signify an error
    and collect status information as it bubbles up the
    stack.

    args are original_exception, set_of_classes,
    [(fieldname, fieldvalue) for each frame]
    """


class _NotFound(RuntimeError):
    """
    Internal exception raised by
    _literal_field_dis_structure_hook_factory.struct_func.recurse to
    collect parameter values as it bubbles up the stack.

    args are strings of "key=value" form.
    """


class _AnythingType:
    def __repr__(self):
        return "<anything>"


# Internal marker standing for "field can have any value, including
# being absent"
_ANYTHING = _AnythingType()


class Converter(object):
    """Converts between structured and unstructured data."""

    __slots__ = (
        "_dis_func_cache",
        "_unstructure_func",
        "_unstructure_attrs",
        "_structure_attrs",
        "_dict_factory",
        "_union_struct_registry",
        "_structure_func",
        "_prefer_attrib_converters",
        "_unstruct_strat",
        "_minimize_union_discriminator_checks",
    )

    def __init__(
        self,
        dict_factory: Callable[[], Any] = dict,
        unstruct_strat: UnstructureStrategy = UnstructureStrategy.AS_DICT,
        prefer_attrib_converters: bool = False,
        minimize_union_disambiguation_checks: bool = False,
    ) -> None:
        unstruct_strat = UnstructureStrategy(unstruct_strat)
        self._prefer_attrib_converters = prefer_attrib_converters
        self._minimize_union_discriminator_checks = (
            minimize_union_disambiguation_checks
        )

        # Create a per-instance cache.
        if unstruct_strat is UnstructureStrategy.AS_DICT:
            self._unstructure_attrs = self.unstructure_attrs_asdict
            self._structure_attrs = self.structure_attrs_fromdict
        else:
            self._unstructure_attrs = self.unstructure_attrs_astuple
            self._structure_attrs = self.structure_attrs_fromtuple

        self._dis_func_cache = lru_cache()(self._get_dis_func)

        self._unstructure_func = MultiStrategyDispatch(
            self._unstructure_identity
        )
        self._unstructure_func.register_cls_list(
            [
                (bytes, self._unstructure_identity),
                (str, self._unstructure_identity),
            ]
        )
        self._unstructure_func.register_func_list(
            [
                (
                    is_protocol,
                    lambda o: self.unstructure(o, unstructure_as=o.__class__),
                ),
                (is_mapping, self._unstructure_mapping),
                (is_sequence, self._unstructure_seq),
                (is_mutable_set, self._unstructure_seq),
                (is_frozenset, self._unstructure_seq),
                (_subclass(Enum), self._unstructure_enum),
                (has, self._unstructure_attrs),
                (is_union_type, self._unstructure_union),
            ]
        )

        # Per-instance register of to-attrs converters.
        # Singledispatch dispatches based on the first argument, so we
        # store the function and switch the arguments in self.loads.
        self._structure_func = MultiStrategyDispatch(self._structure_error)
        self._structure_func.register_func_list(
            [
                (
                    lambda cl: cl is Any or cl is Optional or cl is None,
                    lambda v, _: v,
                ),
                (is_generic_attrs, self._gen_structure_generic, True),
                (is_literal, self._structure_literal),
                (is_sequence, self._structure_list),
                (is_mutable_set, self._structure_set),
                (is_frozenset, self._structure_frozenset),
                (is_tuple, self._structure_tuple),
                (is_mapping, self._structure_dict),
                (
                    is_attrs_union_or_none,
                    self._gen_attrs_union_structure,
                    True,
                ),
                (
                    is_union_with_some_discriminated_or_literal_classes,
                    self._discriminated_union_structure_hook_factory,
                    True,
                ),
                (
                    lambda t: is_union_type(t)
                    and t in self._union_struct_registry,
                    self._structure_union,
                ),
                (is_optional, self._structure_optional),
                (has, self._structure_attrs),
            ]
        )
        # Strings are sequences.
        self._structure_func.register_cls_list(
            [
                (str, self._structure_call),
                (bytes, self._structure_call),
                (int, self._structure_call),
                (float, self._structure_call),
                (Enum, self._structure_call),
            ]
        )

        self._dict_factory = dict_factory

        # Unions are instances now, not classes. We use different registries.
        self._union_struct_registry: Dict[
            Any, Callable[[Any, Type[T]], T]
        ] = {}

    def unstructure(self, obj: Any, unstructure_as=None) -> Any:
        return self._unstructure_func.dispatch(
            obj.__class__ if unstructure_as is None else unstructure_as
        )(obj)

    @property
    def unstruct_strat(self) -> UnstructureStrategy:
        """The default way of unstructuring ``attrs`` classes."""
        return (
            UnstructureStrategy.AS_DICT
            if self._unstructure_attrs == self.unstructure_attrs_asdict
            else UnstructureStrategy.AS_TUPLE
        )

    def register_unstructure_hook(
        self, cls: Any, func: Callable[[T], Any]
    ) -> None:
        """Register a class-to-primitive converter function for a class.

        The converter function should take an instance of the class and return
        its Python equivalent.
        """
        if attrs_has(cls):
            resolve_types(cls)
        if is_union_type(cls):
            self._unstructure_func.register_func_list(
                [(lambda t: t == cls, func)]
            )
        else:
            self._unstructure_func.register_cls_list([(cls, func)])

    def register_unstructure_hook_func(
        self, check_func: Callable[[Any], bool], func: Callable[[T], Any]
    ):
        """Register a class-to-primitive converter function for a class, using
        a function to check if it's a match.
        """
        self._unstructure_func.register_func_list([(check_func, func)])

    def register_unstructure_hook_factory(
        self,
        predicate: Callable[[Any], bool],
        factory: Callable[[Any], Callable[[Any], Any]],
    ) -> None:
        """
        Register a hook factory for a given predicate.

        A predicate is a function that, given a type, returns whether the factory
        can produce a hook for that type.

        A factory is a callable that, given a type, produces an unstructuring
        hook for that type. This unstructuring hook will be cached.
        """
        self._unstructure_func.register_func_list([(predicate, factory, True)])

    def register_structure_hook(
        self, cl: Any, func: Callable[[Any, Type[T]], T]
    ):
        """Register a primitive-to-class converter function for a type.

        The converter function should take two arguments:
          * a Python object to be converted,
          * the type to convert to

        and return the instance of the class. The type may seem redundant, but
        is sometimes needed (for example, when dealing with generic classes).
        """
        if attrs_has(cl):
            resolve_types(cl)
        if is_union_type(cl):
            self._union_struct_registry[cl] = func
            self._structure_func.clear_cache()
        else:
            self._structure_func.register_cls_list([(cl, func)])

    def register_structure_hook_func(
        self,
        check_func: Callable[[Type[T]], bool],
        func: Callable[[Any, Type[T]], T],
    ):
        """Register a class-to-primitive converter function for a class, using
        a function to check if it's a match.
        """
        self._structure_func.register_func_list([(check_func, func)])

    def register_structure_hook_factory(
        self,
        predicate: Callable[[Type[T]], bool],
        factory: Callable[[Type[T]], Callable[[Any, Type[T]], T]],
    ) -> None:
        """
        Register a hook factory for a given predicate.

        A predicate is a function that, given a type, returns whether the factory
        can produce a hook for that type.

        A factory is a callable that, given a type, produces a structuring
        hook for that type. This structuring hook will be cached.
        """
        self._structure_func.register_func_list([(predicate, factory, True)])

    def structure(self, obj: Any, cl: Type[T]) -> T:
        """Convert unstructured Python data structures to structured data."""

        return self._structure_func.dispatch(cl)(obj, cl)

    # Classes to Python primitives.
    def unstructure_attrs_asdict(self, obj) -> Dict[str, Any]:
        """Our version of `attrs.asdict`, so we can call back to us."""
        attrs = fields(obj.__class__)
        dispatch = self._unstructure_func.dispatch
        rv = self._dict_factory()
        for a in attrs:
            name = a.name
            v = getattr(obj, name)
            rv[name] = dispatch(a.type or v.__class__)(v)
        return rv

    def unstructure_attrs_astuple(self, obj) -> Tuple[Any, ...]:
        """Our version of `attrs.astuple`, so we can call back to us."""
        attrs = fields(obj.__class__)
        dispatch = self._unstructure_func.dispatch
        res = list()
        for a in attrs:
            name = a.name
            v = getattr(obj, name)
            res.append(dispatch(a.type or v.__class__)(v))
        return tuple(res)

    def _unstructure_enum(self, obj):
        """Convert an enum to its value."""
        return obj.value

    def _unstructure_identity(self, obj):
        """Just pass it through."""
        return obj

    def _unstructure_seq(self, seq):
        """Convert a sequence to primitive equivalents."""
        # We can reuse the sequence class, so tuples stay tuples.
        dispatch = self._unstructure_func.dispatch
        return seq.__class__(dispatch(e.__class__)(e) for e in seq)

    def _unstructure_mapping(self, mapping):
        """Convert a mapping of attr classes to primitive equivalents."""

        # We can reuse the mapping class, so dicts stay dicts and OrderedDicts
        # stay OrderedDicts.
        dispatch = self._unstructure_func.dispatch
        return mapping.__class__(
            (dispatch(k.__class__)(k), dispatch(v.__class__)(v))
            for k, v in mapping.items()
        )

    def _unstructure_union(self, obj):
        """
        Unstructure an object as a union.

        By default, just unstructures the instance.
        """
        return self._unstructure_func.dispatch(obj.__class__)(obj)

    # Python primitives to classes.

    def _structure_error(self, _, cl):
        """At the bottom of the condition stack, we explode if we can't handle it."""
        msg = (
            "Unsupported type: {0!r}. Register a structure hook for "
            "it.".format(cl)
        )
        raise StructureHandlerNotFoundError(msg, type_=cl)

    def _gen_structure_generic(self, cl):
        """Create and return a hook for structuring generics."""
        fn = make_dict_structure_fn(
            cl,
            self,
            _cattrs_prefer_attrib_converters=self._prefer_attrib_converters,
        )
        return fn

    def _gen_attrs_union_structure(self, cl):
        """Generate a structuring function for a union of attrs classes (and maybe None)."""
        dis_fn = self._get_dis_func(cl)
        has_none = NoneType in cl.__args__

        if has_none:

            def structure_attrs_union(obj, _):
                if obj is None:
                    return None
                return self.structure(obj, dis_fn(obj))

        else:

            def structure_attrs_union(obj, _):
                return self.structure(obj, dis_fn(obj))

        return structure_attrs_union

    @staticmethod
    def _structure_call(obj, cl):
        """Just call ``cl`` with the given ``obj``.

        This is just an optimization on the ``_structure_default`` case, when
        we know we can skip the ``if`` s. Use for ``str``, ``bytes``, ``enum``,
        etc.
        """
        return cl(obj)

    @staticmethod
    def _structure_literal(val, type):
        vals = {
            (x.value if isinstance(x, Enum) else x): x for x in type.__args__
        }
        try:
            return vals[val]
        except KeyError:
            raise Exception(f"{val} not in literal {type}") from None

    # Attrs classes.

    def structure_attrs_fromtuple(
        self, obj: Tuple[Any, ...], cl: Type[T]
    ) -> T:
        """Load an attrs class from a sequence (tuple)."""
        conv_obj = []  # A list of converter parameters.
        for a, value in zip(fields(cl), obj):  # type: ignore
            # We detect the type by the metadata.
            converted = self._structure_attribute(a, value)
            conv_obj.append(converted)

        return cl(*conv_obj)  # type: ignore

    def _structure_attribute(
        self, a: Union[Attribute, Field], value: Any
    ) -> Any:
        """Handle an individual attrs attribute."""
        type_ = a.type
        attrib_converter = getattr(a, "converter", None)
        if self._prefer_attrib_converters and attrib_converter:
            # A attrib converter is defined on this attribute, and prefer_attrib_converters is set
            # to give these priority over registered structure hooks. So, pass through the raw
            # value, which attrs will flow into the converter
            return value
        if type_ is None:
            # No type metadata.
            return value

        try:
            return self._structure_func.dispatch(type_)(value, type_)
        except StructureHandlerNotFoundError:
            if attrib_converter:
                # Return the original value and fallback to using an attrib converter.
                return value
            else:
                raise

    def structure_attrs_fromdict(
        self, obj: Mapping[str, Any], cl: Type[T]
    ) -> T:
        """Instantiate an attrs class from a mapping (dict)."""
        # For public use.

        conv_obj = {}  # Start with a fresh dict, to ignore extra keys.
        for a in fields(cl):  # type: ignore
            name = a.name

            try:
                val = obj[name]
            except KeyError:
                continue

            if name[0] == "_":
                name = name[1:]

            conv_obj[name] = self._structure_attribute(a, val)

        return cl(**conv_obj)  # type: ignore

    def _structure_list(self, obj, cl):
        """Convert an iterable to a potentially generic list."""
        if is_bare(cl) or cl.__args__[0] is Any:
            return [e for e in obj]
        else:
            elem_type = cl.__args__[0]
            return [
                self._structure_func.dispatch(elem_type)(e, elem_type)
                for e in obj
            ]

    def _structure_set(self, obj, cl):
        """Convert an iterable into a potentially generic set."""
        if is_bare(cl) or cl.__args__[0] is Any:
            return set(obj)
        else:
            elem_type = cl.__args__[0]
            return {
                self._structure_func.dispatch(elem_type)(e, elem_type)
                for e in obj
            }

    def _structure_frozenset(self, obj, cl):
        """Convert an iterable into a potentially generic frozenset."""
        if is_bare(cl) or cl.__args__[0] is Any:
            return frozenset(obj)
        else:
            elem_type = cl.__args__[0]
            dispatch = self._structure_func.dispatch
            return frozenset(dispatch(elem_type)(e, elem_type) for e in obj)

    def _structure_dict(self, obj, cl):
        """Convert a mapping into a potentially generic dict."""
        if is_bare(cl) or cl.__args__ == (Any, Any):
            return dict(obj)
        else:
            key_type, val_type = cl.__args__
            if key_type is Any:
                val_conv = self._structure_func.dispatch(val_type)
                return {k: val_conv(v, val_type) for k, v in obj.items()}
            elif val_type is Any:
                key_conv = self._structure_func.dispatch(key_type)
                return {key_conv(k, key_type): v for k, v in obj.items()}
            else:
                key_conv = self._structure_func.dispatch(key_type)
                val_conv = self._structure_func.dispatch(val_type)
                return {
                    key_conv(k, key_type): val_conv(v, val_type)
                    for k, v in obj.items()
                }

    def _structure_optional(self, obj, union):
        if obj is None:
            return None
        union_params = union.__args__
        other = (
            union_params[0] if union_params[1] is NoneType else union_params[1]
        )
        # We can't actually have a Union of a Union, so this is safe.
        return self._structure_func.dispatch(other)(obj, other)

    def _structure_union(self, obj, union):
        """Deal with structuring a union."""
        handler = self._union_struct_registry[union]
        return handler(obj, union)

    def _structure_tuple(self, obj, tup: Type[T]):
        """Deal with converting to a tuple."""
        if tup in (Tuple, tuple):
            tup_params = None
        else:
            tup_params = tup.__args__
        has_ellipsis = tup_params and tup_params[-1] is Ellipsis
        if tup_params is None or (has_ellipsis and tup_params[0] is Any):
            # Just a Tuple. (No generic information.)
            return tuple(obj)
        if has_ellipsis:
            # We're dealing with a homogenous tuple, Tuple[int, ...]
            tup_type = tup_params[0]
            conv = self._structure_func.dispatch(tup_type)
            return tuple(conv(e, tup_type) for e in obj)
        else:
            # We're dealing with a heterogenous tuple.
            return tuple(
                self._structure_func.dispatch(t)(e, t)
                for t, e in zip(tup_params, obj)
            )

    @staticmethod
    def _get_dis_func(union):
        # type: (Type) -> Callable[..., Type]
        """Fetch or try creating a disambiguation function for a union."""
        union_types = union.__args__
        if NoneType in union_types:  # type: ignore
            # We support unions of attrs classes and NoneType higher in the
            # logic.
            union_types = tuple(
                e for e in union_types if e is not NoneType  # type: ignore
            )

        if not all(has(get_origin(e) or e) for e in union_types):
            raise StructureHandlerNotFoundError(
                "Only unions of attrs classes supported "
                "currently. Register a loads hook manually.",
                type_=union,
            )
        return create_uniq_field_dis_func(*union_types)

    def _discriminated_union_structure_hook_factory(
        self, union_: Type[T]
    ) -> Callable[[Any, T], T]:
        classes = get_args(union_)

        if len(classes) < 2:
            raise ValueError("A union cannot have a single type")

        # map from discriminator field name → (value → set of classes that match)
        disfields: Dict[str, Dict[Any, MutableSet[Type]]] = defaultdict(
            lambda: defaultdict(set[Type])
        )

        fallback_only: Dict[str, MutableSet[Type]] = defaultdict(set)

        # First, find all fields that can be used for disambiguation
        # These are either user defined via @has_discriminator
        # or, in "auto mode", are fields defined as Literal in at least one
        # class of the union

        for cls in classes:
            disam_config = getattr(cls, "__cattr_discriminators__", None)

            if disam_config:
                if self._unstructure_attrs == self.unstructure_attrs_astuple:
                    raise ValueError(
                        f"Cannot use @has_discriminator classes with "
                        f"{UnstructureStrategy.AS_TUPLE}"
                    )
                for k, v in disam_config.items():
                    disfields[k][v].add(cls)

            else:
                hints = get_type_hints(
                    get_origin(cls) or cls, include_extras=True
                )
                for name, typ in hints.items():
                    if is_annotated(typ):
                        typ, *metadata = get_args(typ)
                        if any(md is Fallback for md in metadata):
                            fallback_only[name].add(cls)

                    if is_literal(typ):
                        for value in get_args(typ):
                            disfields[name][
                                value.value
                                if isinstance(value, Enum)
                                else value
                            ].add(cls)

        if not disfields:
            # This shouldn't happen, as our handler will only be called
            # when at least one disambiguation field exists
            raise ValueError(
                f"No disambiguation fields found in any of the classes {classes}"
            )

        # Now we need to check if some classes do not define a disambiguation
        # value for a specific field, in which case they always match.
        # Note that "not defining a value" can mean not defining the field at
        # all, or defining it as some arbitrary type.

        for name, valuemap in disfields.copy().items():
            wild_classes = set(classes)
            for clss in valuemap.values():
                for cls in clss:
                    wild_classes.discard(cls)
            if wild_classes:
                for clss in valuemap.values():
                    clss.update(wild_classes - fallback_only[name])
                disfields[name][_ANYTHING] = wild_classes

        # if there is more than one disambiguation field, we might be able
        # to reduce the number of checks by checking the fields
        # that disambiguate the most.
        # The heuristic to go by sum([number of classes per value]^2) seems
        # to be a relatively good one based on my experiments.
        #
        # However, if we are going to check every field value anyway,
        # we don't need to find an "optimal" order, and we just go with
        # defintion order

        best_field_names = list(disfields.keys())
        if self._minimize_union_discriminator_checks:
            best_field_names.sort(
                key=lambda k: sum(len(v) ** 2 for v in disfields[k].values()),
                reverse=False,
            )

        # Now we need to create the decision tree.
        # The tree is stored as a dict, with the keys
        # representing the "match value", and the value
        # either being another map (for further checks),
        # or a structuring hook.
        #
        # Example:
        # Class A matches spam=1; B matches spam=2, eggs=1
        # and both C and D match spam=2, eggs=2,
        # and "best_fields" are ["spam", "eggs"], then the final tree will
        # look something like this:
        # {1: structure_A, 2: {1: structure_B, 2: structure_Union_CD}}
        #
        # Note that structure_Union_CD will be structuring the Union of C and D,
        # and that the value of eggs is not checked in case spam=1.

        def mktree(depth: int, classes: set[Type]):

            # see below
            skip_handlers = frozenset(
                [
                    self._discriminated_union_structure_hook_factory,
                    self._structure_error,
                ]
            )

            if not classes:
                raise ValueError("called with empty classes")

            if depth >= len(best_field_names) or (
                self._minimize_union_discriminator_checks and len(classes) == 1
            ):
                # no more checks left, we now need to find
                # a structure hook.
                try:
                    it = iter(classes)
                    t = next(it)
                    for cl in it:
                        t = Union[t, cl]

                    # if it's another Union, there's a good chance
                    # we would be chosen as structuring hook, resulting
                    # in endless recursion. We already know we cannot
                    # distinguish between the remaining classes,
                    # so we tell dispatch to ignore us.
                    # We also want to immediately fail if there's no
                    # match, so we also skip the "last resort"
                    # handler which would raise an Exception later
                    # when executed.
                    func = self._structure_func.dispatch(
                        t, skip_handlers=skip_handlers
                    )
                    num_annotated = sum(
                        hasattr(cls, "__cattr_discriminators__")
                        for cls in classes
                    )
                    if num_annotated:
                        if num_annotated < len(classes):
                            raise ValueError(
                                "Cannot have ambiguity between classes that are "
                                "annotated with has_discriminator and those "
                                "that aren't."
                            )

                        def remove_discriminators_and_dispatch(obj):
                            for name in best_field_names:
                                obj.pop(name, None)
                            return func(obj, t)

                        return remove_discriminators_and_dispatch
                    else:
                        return lambda obj: func(obj, t)

                except Exception as e:
                    raise _DisambiguationError(e, classes) from None

            else:
                tree = {}
                name = best_field_names[depth]
                for val, classes_ in disfields[name].items():
                    union = classes_ & classes
                    try:
                        if union:
                            print(f"{union=} {name}={val!r}")
                            tree[val] = mktree(depth + 1, union)
                    except _DisambiguationError as e:
                        raise _DisambiguationError(
                            *e.args, (name, val)
                        ) from None

                return tree

        try:
            tree = mktree(0, set(classes))
        except _DisambiguationError as e:
            # convert the internal exception into a ValueError with
            # a helpful error message
            orig_e, classes, *params = e.args
            params_str = ", ".join(f"{k!r}={v!r}" for k, v in params)

            if len(classes) > 1:
                error_msg = (
                    f"Unable to disambiguate between types {classes} when "
                    f"{params_str}: {orig_e.args[0]}\n"
                    f"Hint: register a structure hook for "
                    f"Union[{', '.join(cls.__name__ for cls in classes)}], "
                    f"or mark one or more classes as a fallback."
                )
            else:
                error_msg = (
                    f"Unable to structure type {next(iter(classes))} when "
                    f"{params_str}: {orig_e.args[0]}"
                )

            raise ValueError(error_msg) from orig_e

        # now that the decision tree is build, we can create our
        # structure function and return it. This will simply walk through the
        # tree and execute the specific structure handler.
        # If no handlers match the values, we raise an exception

        def struct_func(data: Mapping, typ: Type) -> Optional[Type]:
            if not isinstance(data, Mapping):
                raise ValueError("Only input mappings are supported.")

            def recurse(i, tree):
                try:
                    fn = best_field_names[i]
                except IndexError:
                    raise _NotFound()
                val = data.get(fn, _ANYTHING)
                try:
                    subtree = tree[val]
                except KeyError:
                    try:
                        subtree = tree[_ANYTHING]
                    except KeyError:
                        if val is _ANYTHING:
                            raise _NotFound(f"{fn!r}=<missing>")
                        else:
                            raise _NotFound(f"{fn!r}={val!r}")
                if isinstance(subtree, Callable):
                    # structure handler, call it
                    return subtree(data)
                else:
                    # another subtree
                    try:
                        return recurse(i + 1, subtree)
                    except _NotFound as e:
                        raise _NotFound(f"{fn!r}={val!r}", *e.args)

            try:
                return recurse(0, tree)
            except _NotFound as e:
                raise ValueError(
                    f"No class matches {', '.join(e.args)}"
                ) from None

        return struct_func


class GenConverter(Converter):
    """A converter which generates specialized un/structuring functions."""

    __slots__ = (
        "omit_if_default",
        "forbid_extra_keys",
        "type_overrides",
        "_unstruct_collection_overrides",
    )

    def __init__(
        self,
        dict_factory: Callable[[], Any] = dict,
        unstruct_strat: UnstructureStrategy = UnstructureStrategy.AS_DICT,
        omit_if_default: bool = False,
        forbid_extra_keys: bool = False,
        type_overrides: Mapping[Type, AttributeOverride] = {},
        unstruct_collection_overrides: Mapping[Type, Callable] = {},
        prefer_attrib_converters: bool = False,
    ):
        super().__init__(
            dict_factory=dict_factory,
            unstruct_strat=unstruct_strat,
            prefer_attrib_converters=prefer_attrib_converters,
        )
        self.omit_if_default = omit_if_default
        self.forbid_extra_keys = forbid_extra_keys
        self.type_overrides = dict(type_overrides)

        self._unstruct_collection_overrides = unstruct_collection_overrides

        # Do a little post-processing magic to make things easier for users.
        co = unstruct_collection_overrides

        # abc.Set overrides, if defined, apply to abc.MutableSets and sets
        if Set in co:
            if MutableSet not in co:
                co[MutableSet] = co[Set]
                co[AbcMutableSet] = co[Set]  # For 3.7/3.8 compatibility.
            if FrozenSetSubscriptable not in co:
                co[FrozenSetSubscriptable] = co[Set]

        # abc.MutableSet overrrides, if defined, apply to sets
        if MutableSet in co:
            if set not in co:
                co[set] = co[MutableSet]

        if FrozenSetSubscriptable in co:
            co[frozenset] = co[
                FrozenSetSubscriptable
            ]  # For 3.7/3.8 compatibility.

        # abc.Sequence overrides, if defined, can apply to MutableSequences, lists and tuples
        if Sequence in co:
            if MutableSequence not in co:
                co[MutableSequence] = co[Sequence]
            if tuple not in co:
                co[tuple] = co[Sequence]

        # abc.MutableSequence overrides, if defined, can apply to lists
        if MutableSequence in co:
            if list not in co:
                co[list] = co[MutableSequence]

        # abc.Mapping overrides, if defined, can apply to MutableMappings
        if Mapping in co:
            if MutableMapping not in co:
                co[MutableMapping] = co[Mapping]

        # abc.MutableMapping overrides, if defined, can apply to dicts
        if MutableMapping in co:
            if dict not in co:
                co[dict] = co[MutableMapping]

        # builtins.dict overrides, if defined, can apply to counters
        if dict in co:
            if Counter not in co:
                co[Counter] = co[dict]

        if unstruct_strat is UnstructureStrategy.AS_DICT:
            # Override the attrs handler.
            self.register_unstructure_hook_factory(
                has_with_generic, self.gen_unstructure_attrs_fromdict
            )
            self.register_structure_hook_factory(
                has_with_generic, self.gen_structure_attrs_fromdict
            )
        self.register_unstructure_hook_factory(
            is_annotated, self.gen_unstructure_annotated
        )
        self.register_unstructure_hook_factory(
            is_hetero_tuple, self.gen_unstructure_hetero_tuple
        )
        self.register_unstructure_hook_factory(
            is_sequence, self.gen_unstructure_iterable
        )
        self.register_unstructure_hook_factory(
            is_mapping, self.gen_unstructure_mapping
        )
        self.register_unstructure_hook_factory(
            is_mutable_set,
            lambda cl: self.gen_unstructure_iterable(cl, unstructure_to=set),
        )
        self.register_unstructure_hook_factory(
            is_frozenset,
            lambda cl: self.gen_unstructure_iterable(
                cl, unstructure_to=frozenset
            ),
        )
        self.register_structure_hook_factory(
            is_annotated, self.gen_structure_annotated
        )
        self.register_structure_hook_factory(
            is_mapping, self.gen_structure_mapping
        )
        self.register_structure_hook_factory(
            is_counter, self.gen_structure_counter
        )

    def gen_unstructure_annotated(self, type):
        origin = type.__origin__
        h = self._unstructure_func.dispatch(origin)
        return h

    def gen_structure_annotated(self, type):
        origin = type.__origin__
        h = self._structure_func.dispatch(origin)
        return h

    def gen_unstructure_attrs_fromdict(self, cl: Type[T]) -> Dict[str, Any]:
        origin = get_origin(cl)
        attribs = fields(origin or cl)
        if attrs_has(cl) and any(isinstance(a.type, str) for a in attribs):
            # PEP 563 annotations - need to be resolved.
            resolve_types(cl)

        attrib_overrides = {}
        for a in attribs:
            v = self.type_overrides.get(a.type) or self.type_overrides.get(
                get_origin(a.type)
            )
            if v:
                attrib_overrides[a.name] = v

        h = make_dict_unstructure_fn(
            cl,
            self,
            _cattrs_omit_if_default=self.omit_if_default,
            **attrib_overrides,
        )
        return h

    def gen_structure_attrs_fromdict(self, cl: Type[T]) -> T:
        attribs = fields(get_origin(cl) if is_generic(cl) else cl)
        if attrs_has(cl) and any(isinstance(a.type, str) for a in attribs):
            # PEP 563 annotations - need to be resolved.
            resolve_types(cl)

        attrib_overrides = {}
        for a in attribs:
            v = self.type_overrides.get(a.type) or self.type_overrides.get(
                get_origin(a.type)
            )
            if v:
                attrib_overrides[a.name] = v
        h = make_dict_structure_fn(
            cl,
            self,
            _cattrs_forbid_extra_keys=self.forbid_extra_keys,
            _cattrs_prefer_attrib_converters=self._prefer_attrib_converters,
            **attrib_overrides,
        )
        # only direct dispatch so that subclasses get separately generated
        return h

    def gen_unstructure_iterable(self, cl: Any, unstructure_to=None):
        unstructure_to = self._unstruct_collection_overrides.get(
            get_origin(cl) or cl, unstructure_to or list
        )
        h = make_iterable_unstructure_fn(
            cl, self, unstructure_to=unstructure_to
        )
        self._unstructure_func.register_cls_list([(cl, h)], direct=True)
        return h

    def gen_unstructure_hetero_tuple(self, cl: Any, unstructure_to=None):
        unstructure_to = self._unstruct_collection_overrides.get(
            get_origin(cl) or cl, unstructure_to or tuple
        )
        h = make_hetero_tuple_unstructure_fn(
            cl, self, unstructure_to=unstructure_to
        )
        self._unstructure_func.register_cls_list([(cl, h)], direct=True)
        return h

    def gen_unstructure_mapping(
        self, cl: Any, unstructure_to=None, key_handler=None
    ):
        unstructure_to = self._unstruct_collection_overrides.get(
            get_origin(cl) or cl, unstructure_to or dict
        )
        h = make_mapping_unstructure_fn(
            cl, self, unstructure_to=unstructure_to, key_handler=key_handler
        )
        self._unstructure_func.register_cls_list([(cl, h)], direct=True)
        return h

    def gen_structure_counter(self, cl: Any):
        h = make_mapping_structure_fn(
            cl, self, structure_to=Counter, val_type=int
        )
        self._structure_func.register_cls_list([(cl, h)], direct=True)
        return h

    def gen_structure_mapping(self, cl: Any):
        h = make_mapping_structure_fn(cl, self)
        self._structure_func.register_cls_list([(cl, h)], direct=True)
        return h
