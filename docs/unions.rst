========================
Tips for handling unions
========================

This sections contains information for advanced union handling.

As mentioned in the structuring section, ``cattrs`` is able to handle simple
unions of ``attrs`` classes automatically. More complex cases require
converter customization or discriminators (since there are many ways of handling
unions).

Unstructuring unions with extra metadata
****************************************

Let's assume a simple scenario of two classes, ``ClassA`` and ``ClassB``, both
of which have no distinct fields and so cannot be used automatically with
``cattrs``.

.. code-block:: python

    @define
    class ClassA:
        a_string: str

    @define
    class ClassB:
        a_string: str


A naive approach to unstructuring either of these would yield identical
dictionaries, and not enough information to restructure the classes.

.. code-block:: python

    >>> converter.unstructure(ClassA("test"))
    {'a_string': 'test'}  # Is this ClassA or ClassB? Who knows!


What we can do is ensure some extra information is present in the
unstructured data, and then use that information to help structure later.


The manual way
--------------

First, we register an unstructure hook for the `Union[ClassA, ClassB]` type.

.. code-block:: python

    >>> converter.register_unstructure_hook(
    ...     Union[ClassA, ClassB],
    ...     lambda o: {"_type": type(o).__name__,  **converter.unstructure(o)}
    ... )
    >>> converter.unstructure(ClassA("test"), unstructure_as=Union[ClassA, ClassB])
    {'_type': 'ClassA', 'a_string': 'test'}

Note that when unstructuring, we had to provide the `unstructure_as` parameter
or `cattrs` would have just applied the usual unstructuring rules to `ClassA`,
instead of our special union hook.

Now that the unstructured data contains some information, we can create a
structuring hook to put it to use:

.. code-block:: python

    >>> converter.register_structure_hook(
    ...     Union[ClassA, ClassB],
    ...     lambda o, _: converter.structure(o, ClassA if o["_type"] == "ClassA" else ClassB)
    ... )
    >>> converter.structure({"_type": "ClassA", "a_string": "test"}, Union[ClassA, ClassB])
    ClassA(a_string='test')

The automated way via discriminators
------------------------------------

Writing these hooks can become tedious pretty quickly. Luckily, ``cattrs`` now
has built-in support for discriminated unions. This feature is pretty powerful
and flexible, but also has some subtleties you need to be aware of. You should
take some time to read this chapter to get familiar how this feature works.

There are two ways we can let ``cattrs`` take care of our problem. The
first is using a special decorator:

Using the ``has_discriminator`` decorator
"""""""""""""""""""""""""""""""""""""""""


.. code-block:: python

    @has_discriminator({"_type": "ClassA"})
    @define
    class ClassA:
        a_string: str

    @has_discriminator({"_type": "ClassB"})
    @define
    class ClassB:
        a_string: str

That's all! ``cattrs`` will now happily structure ``Union[ClassA, ClassB]`` as
long as the unstructured data contains a ``_type`` key.

Note that we cannot omit the type definition on ``ClassB``:

.. code-block:: python

    @has_discriminator()
    @define
    class ClassB:
        a_string: str

As soon as we try to structure into ``Union[ClassA, ClassB]`` for the first time,
we will get an error::

    ValueError: Unable to disambiguate between types {<class '__main__.ClassB'>,
    <class '__main__.ClassA'>} when '_type'='ClassA'

Note that this error will occur before ``cattrs`` even tries to structure the
first data. (All possible conflicts will be checked before structuring is
started for the first time. So you won't get nasty surprises at runtime once you receive
different data on live than in your test instance.) It realized that
``ClassB`` has no discriminator, so technically it is also a match for
``{"_type": "ClassA"}``. There is a way for us to tell ``cattrs`` to only
use ``ClassB`` when it's not ``ClassA``, but for technical reasons this requires
us to use the ``typing.Literal`` approach, more on that later.

There's also no need to register an unstructuring hook. ``cattrs`` will automatically
add the ``_type`` value when unstructuring, and automatically remove it before
structuring (this is so you can use ``forbid_extra_keys`` without 
``GenConverter`` complaining about the "unknown" ``_type`` key)

Some Internet APIs can get pretty crazy with multiple types of discriminators,
luckily ``cattrs`` handles those with ease:

.. code-block:: python

    @has_discriminator({"type": "person"})
    @define
    class Person:
        name: str

    @has_discriminator({"type": "animal", "subtype": "dog"})
    @define
    class Dog:
        name: str
        owner: Person

        def speak(self):
            return "Woof!"

    @has_discriminator({"type": "animal", "subtype": "cat"})
    @define
    class Cat:
        name: str
        owner: Person

        def speak(self):
            return "Meow!"

    obj = cattrs.structure({
        "type": "animal",
        "subtype": "cat",
        "name": "Garfield",
        "owner": { "type": "person", "name": "John" }
    }, Person | Dog | Cat)
    print(obj)  # Cat(name='Garfield', owner=Person(name='John'))
    obj.speak() # Meow!





If we wanted, we could also have added a base Animal class:

.. code-block:: python

    @has_discriminator({"type": "animal"})
    @define
    class Animal:
        name: str
        owner: Person

    # "type": "animal" is automatically inherited from base class
    @has_discriminator({"subtype": "dog"})
    @define
    class Dog(Animal):
        def speak(self):
            return "Woof!"

    @has_discriminator({"subtype": "cat"})
    @define
    class Cat(Animal):
        def speak(self):
            return "Meow!"



In the future, `cattrs` will gain additional tools to make union handling even
easier and automate generating these hooks.
