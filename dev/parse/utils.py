import inspect
import re
from typing import Optional


def snake_to_kebab(text: str) -> str:
    """

    Examples
    --------

    >>> snake_to_kebab('donut_eat_animals')
    'donut-eat-animals'

    """
    return text.replace("_", "-")


def find_method_docstring(klass, method: str) -> Optional[str]:
    """Look through a class' ancestors for the first non-empty method docstring.

    Since Python 3.5, inspect.getdoc is supposed to do exactly this. However, it doesn't seem to
    work for Cython classes.

    Examples
    --------

    >>> class Parent:
    ...
    ...     def foo(self):
    ...         '''foo method'''

    >>> class Child(Parent):
    ...
    ...     def foo(self):
    ...         ...

    >>> find_method_docstring(Child, 'foo')
    'foo method'

    """

    for ancestor in inspect.getmro(klass):
        try:
            ancestor_meth = getattr(ancestor, method)
        except AttributeError:
            break
        if doc := inspect.getdoc(ancestor_meth):
            return doc


def find_method_signature(klass, method: str) -> Optional[inspect.Signature]:
    """Look through a class' ancestors and fill out the methods signature.

    A class method has a signature. But it might now always be complete. When a parameter is not
    annotated, we might want to look through the ancestors and determine the annotation. This is
    very useful when you have a base class that has annotations, and child classes that are not.

    Examples
    --------

    >>> class Parent:
    ...
    ...     def foo(self, x: int) -> int:
    ...         ...

    >>> find_method_signature(Parent, 'foo')
    <Signature (self, x: int) -> int>

    >>> class Child(Parent):
    ...
    ...     def foo(self, x, y: float) -> str:
    ...         ...

    >>> find_method_signature(Child, 'foo')
    <Signature (self, x: int, y: float) -> str>

    """

    m = getattr(klass, method)
    sig = inspect.signature(m)

    params = []

    for param in sig.parameters.values():

        if param.name == "self" or param.annotation is not param.empty:
            params.append(param)
            continue

        for ancestor in inspect.getmro(klass):
            try:
                ancestor_meth = inspect.signature(getattr(ancestor, m.__name__))
            except AttributeError:
                break
            try:
                ancestor_param = ancestor_meth.parameters[param.name]
            except KeyError:
                break
            if ancestor_param.annotation is not param.empty:
                param = param.replace(annotation=ancestor_param.annotation)
                break

        params.append(param)

    return_annotation = sig.return_annotation
    if return_annotation is inspect._empty:
        for ancestor in inspect.getmro(klass):
            try:
                ancestor_meth = inspect.signature(getattr(ancestor, m.__name__))
            except AttributeError:
                break
            if ancestor_meth.return_annotation is not inspect._empty:
                return_annotation = ancestor_meth.return_annotation
                break

    return sig.replace(parameters=params, return_annotation=return_annotation)
