import importlib
import inspect
import os
import pathlib
import re
import shutil
import typing


def snake_to_kebab(snake: str) -> str:
    return snake.replace('_', '-')

def pascal_to_kebab(string):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', string)
    string = re.sub('(.)([0-9]+)', r'\1-\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', string).lower()

def line(s):
    return f'{s}\n'

def paragraph(s):
    return line(f'{s}\n')

def h1(s):
    return paragraph(f'# {s}')

def h2(s):
    return paragraph(f'## {s}')

def h3(s):
    return paragraph(f'### {s}')

def h4(s):
    return paragraph(f'#### {s}')

def link(caption, href):
    return f'[{caption}]({href})'

def code(s):
    return f'`{s}`'


def split_sections(docstring: str) -> typing.Dict[str, str]:
    """Parses the sections of a docstring.

    The docstring is assumed to follow the Google docstring convention.

    """

    lines = iter(docstring.splitlines())
    sections = {}
    admonitions = []

    section = 'Description'
    content = []

    for l in lines:

        # Parse admonitions
        if l.startswith('.. '):
            admonition = ''
            kind = l.lstrip('.. ').rstrip('::')
            for l in lines:
                if not l:
                    break
                admonition += f' {l.rstrip()}'
            admonitions.append((kind, ' '.join(admonition.split())))
            continue

        if not l.startswith(' ') and l.endswith(':') and ' ' not in l:
            sections[section] = inspect.cleandoc('\n'.join(content))
            section = l.split(':')[0]
            content = []
            continue
        content.append(l)

    sections[section] = inspect.cleandoc('\n'.join(content))
    sections['Admonitions'] = admonitions

    return sections


def parse_params_section(params_doc: str) -> typing.Dict[str, str]:
    """Parses params documentation and returns a dict with one element per parameter."""
    descs = {}
    lines = iter(params_doc.splitlines())
    content = next(lines)
    for l in lines:
        if not l.startswith(' '):
            try:
                name, desc = content.split(':', 1)
            except ValueError:
                name, desc = content, ''
            descs[name] = ' '.join(desc.split())
            content = l
            continue
        content += l
    try:
        name, desc = content.split(':', 1)
    except ValueError:
        name, desc = content, ''
    descs[name] = ' '.join(desc.split())

    return descs


def extract_methods(klass):

    def is_public_method(obj):
        return inspect.isroutine(obj) and not obj.__name__.startswith('_')
    methods = dict(inspect.getmembers(klass, predicate=is_public_method))

    # The inspect.signature method does not take into account class inheritance
    # for type annotations like inspect.getdoc does for docstring, so we need
    # to a bit of hacking.

    for name, meth in methods.items():

        sig = inspect.signature(meth)
        if any(p.annotation is not p.empty for p in sig.parameters.values()):
            continue

        for ancestor in inspect.getmro(klass)[1:]:
            try:
                ancestor_meth = getattr(ancestor, name)
            except AttributeError:
                continue
            sig = inspect.signature(ancestor_meth)
            if any(p.annotation is not p.empty for p in sig.parameters.values()):
                methods[name] = ancestor_meth
                break

    return methods.values()


def get_cymeth_doc(meth, klass):
    """Since Python 3.5, inspect.getdoc is supposed to return the docstring from a parent class
    if a klass has none. However this doesn't seem to work for Cython classes.

    """

    doc = None

    for ancestor in inspect.getmro(klass):
        ancestor_meth = getattr(ancestor, meth.__name__)
        doc = inspect.getdoc(ancestor_meth)
        if doc:
            break

    return doc


def extract_doc(obj) -> str:
    """Extracts documentation from a Python object.

    The object is assumed to follow the Google docstring convention. The return value will be a
    blob of MarkDown that MkDocs can parse.

    """

    docstring = inspect.getdoc(obj)
    sections = split_sections(docstring)

    # md is the name given to the return value, it stands for MarkDown
    md = h1(obj.__name__)

    # Description
    desc = sections.get('Description')
    if desc:
        md += paragraph(desc)

    # Parameters
    try:
        signature = inspect.signature(obj)
    except ValueError:
        signature = None
    if signature and signature.parameters:
        md += h2('Parameters')
        params_desc = parse_params_section(sections['Parameters'])
        for param in signature.parameters.values():

            # Parameter name
            li = f'- **{param.name}**'

            # Type annotation
            if param.annotation is not param.empty:
                li += f' (*{inspect.formatannotation(param.annotation)}*)'

            # Default value
            if param.default is not param.empty:
                li += f' – defaults to `{param.default}`'

            # Parameter description
            md += line(li)
            desc = params_desc.get(param.name)
            if desc:
                md += line(f'\n    {desc}\n')

    # Returns
    if signature:
        returns = signature.return_annotation
        if not returns is signature.empty:
            md += h2('Returns')
            md += paragraph(f'*{inspect.formatannotation(returns)}*')
            return_comment = sections.get('Returns')
            if return_comment:
                md += paragraph(return_comment)

    # Attributes
    attrs = sections.get('Attributes')
    if attrs:
        md += h2('Attributes')
        for name, desc in parse_params_section(attrs).items():
            try:
                name, ann = name.split(' ')
                ann = ann.lstrip('(').rstrip(')')
            except ValueError:
                ann = None
            el = f'- **{name}**'
            if ann:
                el += f' (*{ann}*)'
            md += line(el)
            if desc:
                md += line(f'\n    {desc}\n')

    # Methods
    if inspect.isclass(obj):
        md += h2('Methods')
        md += paragraph('???- note "Click to expand"')
        tab = ' ' * 4

        for i, meth in enumerate(extract_methods(klass=obj)):
            md += tab + h3(meth.__name__)

            # Parse method docstring
            if type(meth).__name__ == 'cython_function_or_method':
                doc = get_cymeth_doc(meth, klass=obj)
            else:
                doc = inspect.getdoc(meth)

            meth_sections = split_sections(doc)

            # Method description
            desc = meth_sections.get('Description')
            if desc:
                md += tab +paragraph(desc)

            signature = inspect.signature(meth)

            # Parameters
            if list(signature.parameters) != ['self']:

                md += tab + h4('Parameters')
                print(meth)
                params_desc = parse_params_section(meth_sections['Parameters'])
                for param in signature.parameters.values():

                    if param.name == 'self':
                        continue

                    # Parameter name
                    li = f'- **{param.name}**'

                    # Type annotation
                    if param.annotation is not param.empty:
                        li += f' (*{inspect.formatannotation(param.annotation)}*)'

                    # Default value
                    if param.default is not param.empty:
                        li += f' – defaults to `{param.default}`'

                    # Parameter description
                    md += tab + line(li)
                    desc = params_desc.get(param.name)
                    if desc:
                        md += tab + line(f'\n    {desc}\n')

            # Returns
            if not signature.return_annotation is signature.empty:
                md += tab + h4('Returns')
                md += tab + paragraph(f'*{inspect.formatannotation(signature.return_annotation)}*')
                return_comment = meth_sections.get('Returns')
                if return_comment:
                    md += tab + paragraph(return_comment)

    # Example
    example = sections.get('Example') or sections.get('Examples')
    if example:
        md += h2('Example')

        in_code = False
        after_space = False

        for l in example.splitlines():

            if in_code and after_space and l and not l.startswith('>>>') and not l.startswith('...'):
                md += line('```')
                in_code = False
                after_space = False

            if not in_code and l.startswith('>>>'):
                md += line('```python')
                in_code = True

            after_space = False
            if not l:
                after_space = True

            md += line(l)

        if in_code:
            md += line('```')

    # Admonitions
    for (kind, content) in sections.get('Admonitions', []):
        md += line(f'!!! {kind}')
        md += paragraph(f'    {content}')

    # References
    references = sections.get('References')
    if references:
        md += h2('References')
        md += paragraph(references)

    return md


def write_module(mod, where):

    mod_name = mod.__name__.split('.')[-1]
    mod_header = inspect.getdoc(mod).partition('\n')[0].rstrip('.')
    mod_path = where.joinpath(snake_to_kebab(mod_name))

    # Create a directory for the module
    os.makedirs(mod_path, exist_ok=True)
    with open(mod_path.joinpath('.pages'), 'w') as f:
        f.write(f'title: {mod_header}')

    # Go through the functions
    for name, func in inspect.getmembers(mod, inspect.isfunction):
        print(name)
        path = mod_path.joinpath(snake_to_kebab(name))
        with open(path.with_suffix('.md'), 'w') as f:
            doc = extract_doc(obj=func)
            f.write(doc)

    # Go through the classes
    for name, klass in inspect.getmembers(mod, inspect.isclass):
        print(name)
        path = mod_path.joinpath(pascal_to_kebab(name))
        with open(path.with_suffix('.md'), 'w') as f:
            doc = extract_doc(obj=klass)
            f.write(doc)

    # Go through the submodules
    for name, submod in inspect.getmembers(mod, inspect.ismodule):
        # We only want to go through the public submodules, such as optim.schedulers
        if name not in mod.__all__:
            continue
        write_module(mod=submod, where=mod_path)  # we're recursing



if __name__ == '__main__':

    api_path = pathlib.Path('docs/api-reference')

    # Create a directory for the API reference
    shutil.rmtree(api_path, ignore_errors=True)
    os.makedirs(api_path, exist_ok=True)
    with open(api_path.joinpath('.pages'), 'w') as f:
        f.write('title: API reference')

    # Load all of creme's modules
    print('Loading modules...', end=' ', flush=True)
    modules = dict(inspect.getmembers(importlib.import_module('creme'), inspect.ismodule))
    modules = {
        'linear_model': modules['linear_model'],
        'stream': modules['stream'],
        'optim': modules['optim']
    }
    print('done')

    for name, mod in modules.items():
        print(f'{name}...', end=' ', flush=True)
        write_module(mod=mod, where=api_path)
        print('done')

# TODO: link to classes in type annotations (maybe using https://facelessuser.github.io/pymdown-extensions/extensions/pathconverter/)
# TODO: display children and parents inheritance (maybe using inspect.getclasstree and inspect.mro,
# possibly with mermaid as is done in the superfences section here: https://facelessuser.github.io/pymdown-extensions/)
# TODO: type hinting for Cython classes
