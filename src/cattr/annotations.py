class _FallbackType:
    def __repr__(self):
        return "Fallback"


Fallback = _FallbackType()
"""
Can be used to annotate types in discriminated classes as fallback.

Example:

class A:
    t: Literal[1]

class Other:
    t: Annotated[int, Fallback]

cattrs.structure({"t": 42}, Union[A, Other])
"""


def has_discriminator(discriminators={}, replace=False):
    """
    Adds discriminators to class, this allows it to be identified in a
    union.

    Note that marking a class this way will turn off automatic disambiguation
    via `Literal` types.

    `discriminators` is a dict, with the key being the discriminator, and
    the value the discriminator value. These are simply added to the
    unstructured object, and are checked when a union needs to be
    disambiguated. The values will be removed before structuring

    When inheriting from a marked class, you can mark the child as well.
    If `replace` is False (the default), then the key/values are added
    to the inherited values, if True, the old values are overwritten.
    """

    def wrapper(cls):
        v = getattr(cls, "__cattr_discriminators__", {})
        if discriminators:
            if replace:
                v = discriminators
            else:
                v = v.copy()
                v.update(discriminators)
        cls.__cattr_discriminators__ = v
        return cls

    return wrapper
