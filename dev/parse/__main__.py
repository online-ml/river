"""This script is responsible for building the API reference. The API reference is located in
docs/api. The script scans through all the modules, classes, and functions. It processes
the __doc__ of each object and formats it so that MkDocs can process it in turn.

"""
import argparse
import collections
import doctest
import functools
import importlib
import inspect
import os
import pathlib
import re
import shutil

from numpydoc.docscrape import ClassDoc, FunctionDoc


def md_line(text):
    return f"{text}\n"


def h1(text):
    return md_line(f"# {text}")


def h2(text):
    return md_line(f"## {text}")


def h3(text):
    return md_line(f"### {text}")


def h4(text):
    return md_line(f"#### {text}")


def link(caption, href):
    return f"[{caption}]({href})"


def code(text):
    return f"`{text}`"


def li(text):
    return f"- {text}\n"


def snake_to_kebab(text: str) -> str:
    """

    Examples
    --------

    >>> snake_to_kebab('donut_eat_animals')
    'donut-eat-animals'

    """
    return text.replace("_", "-")


def find_method_docstring(klass, method: str) -> str:
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


def find_method_signature(klass, method: str) -> inspect.Signature:
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


class Linkifier:
    PATTERN = re.compile(r"`?(\w+\.)+\w+`?")

    def __init__(self, library):
        self.library = library

        path_index = {}
        rename_index = {}

        # Either modules are defined in the module's __init__.py...
        modules = dict(inspect.getmembers(importlib.import_module(library), inspect.ismodule))
        # ... either they're defined in an api.py file
        modules.update(
            dict(inspect.getmembers(importlib.import_module(f"{library}.api"), inspect.ismodule))
        )

        def index_module(mod_name, mod, path):
            path = os.path.join(path, mod_name)
            dotted_path = path.replace("/", ".")

            for func_name, func in inspect.getmembers(mod, inspect.isfunction):
                for e in (
                    f"{mod_name}.{func_name}",
                    f"{dotted_path}.{func_name}",
                    f"{func.__module__}.{func_name}",
                ):
                    path_index[e] = os.path.join(path, snake_to_kebab(func_name))
                    rename_index[e] = f"{dotted_path}.{func_name}"

            for klass_name, klass in inspect.getmembers(mod, inspect.isclass):
                for e in (
                    f"{mod_name}.{klass_name}",
                    f"{dotted_path}.{klass_name}",
                    f"{klass.__module__}.{klass_name}",
                ):
                    path_index[e] = os.path.join(path, klass_name)
                    rename_index[e] = f"{dotted_path}.{klass_name}"

            for submod_name, submod in inspect.getmembers(mod, inspect.ismodule):
                if submod_name not in mod.__all__ or submod_name == "typing":
                    continue
                for e in (f"{mod_name}.{submod_name}", f"{dotted_path}.{submod_name}"):
                    path_index[e] = os.path.join(path, snake_to_kebab(submod_name))

                # Recurse
                index_module(submod_name, submod, path=path)

        for mod_name, mod in modules.items():
            index_module(mod_name, mod, path="")

        # Prepend the name of the library to each index entry
        for k in list(path_index.keys()):
            path_index[f"{self.library}.{k}"] = path_index[k]
        for k in list(rename_index.keys()):
            rename_index[f"{self.library}.{k}"] = rename_index[k]

        # HACK: replace underscores with dashes in links
        for k, v in path_index.items():
            path_index[k] = v.replace("_", "-")

        self.path_index = path_index
        self.rename_index = rename_index

    def linkify(self, text, prefix):
        def replace(x):
            # HACK
            if "collections" in x.group():
                return x.group()

            if text.count("```", 0, x.start()) % 2 == 1:
                return x.group()

            y = x.group().strip("`")
            if path := self.path_index.get(y):
                name = self.rename_index.get(y, y)
                name = f"`{name}`'" if x.group().startswith("`") else name
                name = name.strip("'")
                return f"[{name}]({prefix}{path})"
            return x.group()

        return self.PATTERN.sub(replace, text)


def concat_lines(lines):
    return inspect.cleandoc(
        " ".join(
            # Either empty space or list item
            f"{l}\n\n" if (l == "") or (l.strip().startswith("-")) else l
            for l in lines
        )
    )


def print_docstring(obj, file):
    """Prints a classes's docstring to a file."""

    doc = ClassDoc(obj) if inspect.isclass(obj) else FunctionDoc(obj)

    printf = functools.partial(print, file=file)

    printf(h1(obj.__name__))
    printf(md_line(concat_lines(doc["Summary"])))
    printf(md_line(concat_lines(doc["Extended Summary"])))

    # We infer the type annotations from the signatures, and therefore rely on the signature
    # instead of the docstring for documenting parameters
    try:
        signature = inspect.signature(obj)
    except ValueError:
        signature = (
            inspect.Signature()
        )  # TODO: this is necessary for Cython classes, but it's not correct
    params_desc = {param.name: " ".join(param.desc) for param in doc["Parameters"]}

    # Parameters
    if signature.parameters:
        printf(h2("Parameters"))
    for param in signature.parameters.values():
        # Name
        printf(f"- **{param.name}**\n")
        # Type annotation
        if param.annotation is not param.empty:
            anno = inspect.formatannotation(param.annotation)
            anno = anno.strip("'")
            printf(f"     *Type* → *{anno}*\n")
        # Default value
        if param.default is not param.empty:
            printf(f"     *Default* → `{param.default}`\n")
        printf("\n", file=file)
        # Description
        desc = params_desc[param.name]
        if desc:
            printf(f"    {desc}\n")
    printf("")

    # Attributes
    if doc["Attributes"]:
        printf(h2("Attributes"))
    for attr in doc["Attributes"]:
        # Name
        printf(f"- **{attr.name}**", end="")
        # Type annotation
        if attr.type:
            printf(f" (*{attr.type}*)", end="")
        printf("\n", file=file)
        # Description
        desc = " ".join(attr.desc)
        if desc:
            printf(f"    {desc}\n")
    printf("")

    # Examples
    if doc["Examples"]:
        printf(h2("Examples"))

        parser = doctest.DocTestParser()
        lines = parser.parse(inspect.cleandoc("\n".join(doc["Examples"])))
        in_code = False

        for line in lines:
            if isinstance(line, doctest.Example):
                example = line

                # Start code fences for new examples
                if not in_code:
                    printf("```python")
                    in_code = True

                printf(example.source, end="")
                if example.want:
                    # Close code fences for example source
                    printf("```")
                    # Enclose the output in code fences
                    printf("```")
                    printf(example.want, end="")
                    printf("```")
                    in_code = False

            elif text := line:
                # Close code fences for example source
                if in_code and text.strip():
                    printf("```")
                    in_code = False
                printf(text, end="")
        else:
            # Close code fences for example source
            if in_code:
                printf("```")
                in_code = False

    printf("")

    import river.base
    import river.ensemble

    # Methods
    if inspect.isclass(obj) and doc["Methods"]:
        printf(h2("Methods"))
        printf_indent = lambda x, **kwargs: printf(f"    {x}", **kwargs)

        for meth in doc["Methods"]:
            base_method_names = {"clone", "mutate"}

            if (
                issubclass(obj, river.base.Base)
                and not obj is river.base.Base
                and meth.name in base_method_names
            ):
                continue

            container_method_names = {
                "append",
                "clear",
                "copy",
                "count",
                "extend",
                "index",
                "insert",
                "pop",
                "popitem",
                "keys",
                "fromkeys",
                "get",
                "remove",
                "reverse",
                "sort",
                "setdefault",
                "values",
                "update",
                "items",
            }

            if (
                issubclass(obj, (collections.UserList, collections.UserDict))
                and not obj is river.base.Ensemble
                and meth.name in container_method_names
            ):
                continue

            printf(md_line(f'???- abstract "{meth.name}"'))

            # Parse method docstring
            docstring = find_method_docstring(klass=obj, method=meth.name)
            if not docstring:
                continue
            meth_doc = FunctionDoc(func=None, doc=docstring)

            printf_indent(md_line(" ".join(meth_doc["Summary"])))
            if meth_doc["Extended Summary"]:
                printf_indent(md_line(" ".join(meth_doc["Extended Summary"])))

            # We infer the type annotations from the signatures, and therefore rely on the signature
            # instead of the docstring for documenting parameters
            signature = find_method_signature(obj, meth.name)
            params_desc = {param.name: " ".join(param.desc) for param in doc["Parameters"]}

            # Parameters
            if len(signature.parameters) > 1:  # signature is never empty, but self doesn't count
                printf_indent("**Parameters**\n")
            for param in signature.parameters.values():
                if param.name == "self":
                    continue
                # Name
                printf_indent(f"- **{param.name}**", end="")
                # Type annotation
                if param.annotation is not param.empty:
                    printf_indent(f" — *{inspect.formatannotation(param.annotation)}*", end="")
                # Default value
                if param.default is not param.empty:
                    printf_indent(f" — defaults to `{param.default}`", end="")
                printf_indent("", file=file)
                # Description
                desc = params_desc.get(param.name)
                if desc:
                    printf_indent(f"    {desc}")
            printf_indent("")

            # Returns
            if meth_doc["Returns"]:
                printf_indent("**Returns**\n")
                return_val = meth_doc["Returns"][0]
                if signature.return_annotation is not inspect._empty:
                    if inspect.isclass(signature.return_annotation):
                        printf_indent(f"*{signature.return_annotation.__name__}*: ", end="")
                    else:
                        printf_indent(f"*{signature.return_annotation}*: ", end="")
                printf_indent(return_val.type)
                printf_indent("")

            # Add a space between methods
            printf("<span />")

    # Notes
    if doc["Notes"]:
        printf(h2("Notes"))
        printf(md_line("\n".join(doc["Notes"])))

    # References
    if doc["References"]:
        printf(md_line("\n".join(doc["References"])))


def print_module(mod, path, overview, depth=0, verbose=False):
    mod_name = mod.__name__.split(".")[-1]

    # Create a directory for the module
    mod_slug = snake_to_kebab(mod_name)
    mod_path = path.joinpath(mod_slug)
    mod_short_path = str(mod_path).replace("docs/api/", "")
    os.makedirs(mod_path, exist_ok=True)
    with open(mod_path.joinpath(".pages"), "w") as f:
        f.write(f"title: {mod_name}")

    # Add the module to the overview
    if depth == 0:
        print(h2(mod_name), file=overview)
    elif depth == 1:
        print(h3(mod_name), file=overview)
    elif depth == 2:
        print(h4(mod_name), file=overview)
    else:
        raise ValueError("Module depth must be between 0 and 2, you went too deep!")
    if mod.__doc__:
        print(md_line(mod.__doc__), file=overview)

    # Extract all public classes and functions
    ispublic = lambda x: x.__name__ in mod.__all__ and not x.__name__.startswith("_")
    classes = inspect.getmembers(mod, lambda x: inspect.isclass(x) and ispublic(x))
    funcs = inspect.getmembers(mod, lambda x: inspect.isfunction(x) and ispublic(x))

    # Overview
    if hasattr(mod, "_docs_overview"):
        mod._docs_overview(functools.partial(print, file=overview))
    else:
        if classes and funcs:
            print("\n**Classes**\n", file=overview)
        for _, c in classes:
            slug = snake_to_kebab(c.__name__)
            print(
                li(link(c.__name__, f"../{mod_short_path}/{slug}")),
                end="",
                file=overview,
            )
        if classes and funcs:
            print("\n**Functions**\n", file=overview)
        for _, f in funcs:
            slug = snake_to_kebab(f.__name__)
            print(
                li(link(f.__name__, f"../{mod_short_path}/{slug}")),
                end="",
                file=overview,
            )

    # Docstrings
    for _, c in classes:
        if verbose:
            print(f"{mod_name}.{c.__name__}")
        slug = snake_to_kebab(c.__name__)
        with open(mod_path.joinpath(slug).with_suffix(".md"), "w") as file:
            print_docstring(obj=c, file=file)

    for _, f in funcs:
        if verbose:
            print(f"{mod_name}.{f.__name__}")
        slug = snake_to_kebab(f.__name__)
        with open(mod_path.joinpath(slug).with_suffix(".md"), "w") as file:
            print_docstring(obj=f, file=file)

    # Sub-modules
    for name, submod in inspect.getmembers(mod, inspect.ismodule):
        # We only want to go through the public submodules, such as optim.schedulers
        if (
            name in ("tags", "typing", "inspect", "skmultiflow_utils")
            or name not in mod.__all__
            or name.startswith("_")
        ):
            continue

        if verbose:
            print(f"{mod_name}.{name}")

        print_module(
            mod=submod,
            path=mod_path,
            overview=overview,
            depth=depth + 1,
            verbose=verbose,
        )

    print("", file=overview)


def print_library(library: str, output_dir: pathlib.Path, verbose=False):
    # Create a directory for the API reference
    shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir.joinpath(".pages"), "w") as f:
        f.write("title: API reference 🍱\narrange:\n  - overview.md\n  - ...\n")

    overview = open(output_dir.joinpath("overview.md"), "w")
    print(h1("Overview"), file=overview)

    for mod_name, mod in inspect.getmembers(
        importlib.import_module(f"{library}.api"), inspect.ismodule
    ):
        if mod_name.startswith("_") or mod_name == "api":
            continue
        if verbose:
            print(mod_name)
        print_module(mod, path=output_dir, overview=overview, verbose=verbose)


def linkify_docs(library: str, docs_dir: pathlib.Path, verbose=False):
    shutil.rmtree(docs_dir.joinpath("linkified"), ignore_errors=True)
    shutil.copytree(src=docs_dir, dst=docs_dir.joinpath("linkified"), dirs_exist_ok=True)
    linkifier = Linkifier(library=library)

    for page in docs_dir.glob("**/*.md"):
        # Ignore files in the linkified directory
        if str(page).startswith("docs/linkified"):
            continue

        text = page.read_text()
        prefix = "../" * (str(page).count("/") - 1)

        if "/api/" not in str(page):
            prefix = f"../{prefix}api/"

        if "benchmarks" not in str(page):
            if verbose:
                print(f"Adding links to {page}")
            text = linkifier.linkify(text, prefix=prefix)

        # Write back text to file
        linkified_page = pathlib.Path(str(page).replace("docs/", "docs/linkified/"))
        linkified_page.write_text(text)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("library", nargs="?", help="the library to document")
    parser.add_argument("--out", default="docs", help="where to dump the docs")
    parser.add_argument("--verbose", dest="verbose", action="store_true")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    print_library(
        library=args.library,
        output_dir=pathlib.Path(args.out) / "api",
        verbose=args.verbose,
    )
    linkify_docs(library=args.library, docs_dir=pathlib.Path(args.out), verbose=args.verbose)


if __name__ == "__main__":
    main()
