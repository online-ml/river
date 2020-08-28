"""This script is responsible for building the API reference. The API reference is located in
docs/api-reference. The script scans through all the modules, classes, and functions. It processes
the __doc__ of each object and formats it so that MkDocs can process it in turn.

"""
import functools
import importlib
import inspect
import os
import pathlib
import re
import shutil

from numpydoc.docscrape import ClassDoc
from numpydoc.docscrape import FunctionDoc


def paragraph(text):
    return f'{text}\n'

def h1(text):
    return paragraph(f'# {text}')

def h2(text):
    return paragraph(f'## {text}')

def h3(text):
    return paragraph(f'### {text}')

def h4(text):
    return paragraph(f'#### {text}')

def link(caption, href):
    return f'[{caption}]({href})'

def code(text):
    return f'`{text}`'

def li(text):
    return f'- {text}\n'

def snake_to_kebab(text):
    return text.replace('_', '-')

def inherit_docstring(c, meth):
    """Since Python 3.5, inspect.getdoc is supposed to return the docstring from a parent class
    if a class has none. However this doesn't seem to work for Cython classes.

    """

    doc = None

    for ancestor in inspect.getmro(c):
        try:
            ancestor_meth = getattr(ancestor, meth)
        except AttributeError:
            break
        doc = inspect.getdoc(ancestor_meth)
        if doc:
            break

    return doc

def snake_to_kebab(snake: str) -> str:
    return snake.replace('_', '-')

def pascal_to_kebab(string):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', string)
    string = re.sub('(.)([0-9]+)', r'\1-\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', string).lower()


class Linkifier:

    def __init__(self):

        path_index = {}
        name_index = {}

        modules = dict(inspect.getmembers(importlib.import_module('creme'), inspect.ismodule))
        modules = {
            'base': modules['base'],
            'linear_model': modules['linear_model'],
            'stream': modules['stream'],
            'optim': modules['optim']
        }

        def index_module(mod_name, mod, path):
            path = os.path.join(path, mod_name).replace('/', '.')

            for func_name, func in inspect.getmembers(mod, inspect.isfunction):
                for e in (
                    f'{mod_name}.{func_name}',
                    f'{path}.{func_name}',
                    f'{func.__module__}.{func_name}'
                ):
                    path_index[e] = os.path.join(path, snake_to_kebab(func_name))
                    name_index[e] = f'{path}.{func_name}'

            for klass_name, klass in inspect.getmembers(mod, inspect.isclass):
                for e in (
                    f'{mod_name}.{klass_name}',
                    f'{path}.{klass_name}',
                    f'{klass.__module__}.{klass_name}'
                ):
                    path_index[e] = os.path.join(path, klass_name)
                    name_index[e] = f'{path}.{klass_name}'

            for submod_name, submod in inspect.getmembers(mod, inspect.ismodule):
                if submod_name not in mod.__all__ or submod_name == 'typing':
                    continue
                for e in (f'{mod_name}.{submod_name}', f'{path}.{submod_name}'):
                    path_index[e] = os.path.join(path, snake_to_kebab(submod_name))

                # Recurse
                index_module(submod_name, submod, path=path)

        for mod_name, mod in modules.items():
            index_module(mod_name, mod, path='')

        # Prepend the location of the API reference to each path
        for k, v in path_index.items():
            path_index[k] = os.path.join('/api-reference', v)

        # Prepend creme to each index entry
        for k in list(path_index.keys()):
            path_index[f'creme.{k}'] = path_index[k]
        for k in list(name_index.keys()):
            name_index[f'creme.{k}'] = name_index[k]

        self.path_index = path_index
        self.name_index = name_index

    def linkify(self, text, use_fences):
        path = self.path_index.get(text)
        name = self.name_index.get(text)
        if path and name:
            if use_fences:
                return f'[`{name}`]({path})'
            return f'[{name}]({path})'
        return None

    def linkify_fences(self, text):
        between_fences = re.compile('`[\w\.]+\.\w+`')
        return between_fences.sub(lambda x: self.linkify(x.group().strip('`'), True) or x.group(), text)

    def linkify_dotted(self, text):
        dotted = re.compile('\w+\.[\.\w]+')
        return dotted.sub(lambda x: self.linkify(x.group(), False) or x.group(), text)


def concat_lines(lines):
    return inspect.cleandoc(' '.join('\n\n' if line == '' else line for line in lines))


def print_docstring(obj, file):
    """Prints a classes's docstring to a file.

    """

    doc = ClassDoc(obj) if inspect.isclass(obj) else FunctionDoc(obj)

    printf = functools.partial(print, file=file)

    printf(h1(obj.__name__))
    printf(linkifier.linkify_fences(paragraph(concat_lines(doc['Summary']))))
    printf(linkifier.linkify_fences(paragraph(concat_lines(doc['Extended Summary']))))

    # We infer the type annotations from the signatures, and therefore rely on the signature
    # instead of the docstring for documenting parameters
    signature = inspect.signature(obj)
    params_desc = {param.name: ' '.join(param.desc) for param in doc['Parameters']}

    # Parameters
    printf(h2('Parameters'))
    for param in signature.parameters.values():
        # Name
        printf(f'- **{param.name}**', end='')
        # Type annotation
        if param.annotation is not param.empty:
            anno = inspect.formatannotation(param.annotation)
            anno = linkifier.linkify_dotted(anno)
            printf(f' (*{anno}*)', end='')
        # Default value
        if param.default is not param.empty:
            printf(f' – defaults to `{param.default}`', end='')
        printf('\n', file=file)
        # Description
        desc = params_desc[param.name]
        if desc:
            printf(f'    {desc}\n')
    printf('')

    # Attributes
    if doc['Attributes']:
        printf(h2('Attributes'))
        for attr in doc['Attributes']:
            # Name
            printf(f'- **{attr.name}**', end='')
            # Type annotation
            printf(f' (*{attr.type}*)', end='')
            printf('', file=file)
            # Description
            desc = ' '.join(attr.desc)
            if desc:
                printf(f'    {desc}')
    printf('')

    # Examples
    if doc['Examples']:
        printf(h2('Examples'))

        in_code = False
        after_space = False

        for line in inspect.cleandoc('\n'.join(doc['Examples'])).splitlines():

            if (
                in_code and after_space and line
                and not line.startswith('>>>') and not line.startswith('...')
            ):
                printf('```\n')
                in_code = False
                after_space = False

            if not in_code and line.startswith('>>>'):
                printf('```python')
                in_code = True

            after_space = False
            if not line:
                after_space = True

            printf(line)

        if in_code:
            printf('```')
    printf('')

    # Methods
    if inspect.isclass(obj):
        printf(h2('Methods'))
        printf_indent = lambda x, **kwargs: printf(f'    {x}', **kwargs)

        for meth in doc['Methods']:

            printf(paragraph(f'???- note "{meth.name}"'))

            # Parse method docstring
            docstring = inherit_docstring(c=obj, meth=meth.name)
            if not docstring:
                continue
            meth_doc = FunctionDoc(func=None, doc=docstring)

            printf_indent(paragraph(' '.join(meth_doc['Summary'])))
            if meth_doc['Extended Summary']:
                printf_indent(paragraph(' '.join(meth_doc['Extended Summary'])))

            # We infer the type annotations from the signatures, and therefore rely on the signature
            # instead of the docstring for documenting parameters
            signature = inspect.signature(getattr(obj, meth.name))
            params_desc = {param.name: ' '.join(param.desc) for param in doc['Parameters']}

            # Parameters
            for param in signature.parameters.values():
                if param.name == 'self':
                    continue
                # Name
                printf_indent(f'- **{param.name}**', end='')
                # Type annotation
                if param.annotation is not param.empty:
                    printf(f' (*{inspect.formatannotation(param.annotation)}*)', end='')
                # Default value
                if param.default is not param.empty:
                    printf(f' – defaults to `{param.default}`', end='')
                printf('', file=file)
                # Description
                desc = params_desc.get(param.name)
                if desc:
                    printf_indent(f'    {desc}')
            printf('')

    # References
    if doc['References']:
        printf('**References**')
        printf(paragraph('\n'.join(doc['References'])))


def print_module(mod, path, overview, is_submodule=False):

    mod_name = mod.__name__.split('.')[-1]

    # Create a directory for the module
    mod_slug = snake_to_kebab(mod_name)
    mod_path = path.joinpath(mod_slug)
    os.makedirs(mod_path, exist_ok=True)
    with open(mod_path.joinpath('.pages'), 'w') as f:
        f.write(f'title: {mod_name}')

    # Add the module to the overview
    if is_submodule:
        print(h3(mod_name), file=overview)
    else:
        print(h2(mod_name), file=overview)
    if mod.__doc__:
        print(paragraph(mod.__doc__), file=overview)

    # Extract all public classes and functions
    ispublic = lambda x: x.__name__ in mod.__all__
    classes = inspect.getmembers(mod, lambda x: inspect.isclass(x) and ispublic(x))
    funcs = inspect.getmembers(mod, lambda x: inspect.isfunction(x) and ispublic(x))

    # Classes

    if classes and funcs:
        print('**Classes**', file=overview)

    for _, c in classes:

        if c.__name__ not in ('LinearRegression', 'Optimizer'):
            continue
        print(f'{mod_name}.{c.__name__}')

        # Add the class to the overview
        slug = snake_to_kebab(c.__name__)
        print(li(link(c.__name__, f'/api-reference/{mod_slug}/{slug}')), end='', file=overview)

        # Write down the class' docstring
        with open(mod_path.joinpath(slug).with_suffix('.md'), 'w') as file:
            print_docstring(obj=c, file=file)

    # Functions

    if classes and funcs:
        print('**Functions**', file=overview)

    for _, f in funcs:

        if f.__name__ != 'progressive_val_score':
            continue
        print(f'{mod_name}.{f.__name__}')

        # Add the function to the overview
        slug = snake_to_kebab(f.__name__)
        print(li(link(f.__name__, f'/api-reference/{mod_slug}/{slug}')), end='', file=overview)

        # Write down the function' docstring
        with open(mod_path.joinpath(slug).with_suffix('.md'), 'w') as file:
            print_docstring(obj=f, file=file)

    # Sub-modules
    for name, submod in inspect.getmembers(mod, inspect.ismodule):
        # We only want to go through the public submodules, such as optim.schedulers
        if name not in mod.__all__:
            continue
        print_module(mod=submod, path=mod_path, overview=overview, is_submodule=True)

    print('', file=overview)


if __name__ == '__main__':

    api_path = pathlib.Path('docs/api-reference')

    # Create a directory for the API reference
    shutil.rmtree(api_path, ignore_errors=True)
    os.makedirs(api_path, exist_ok=True)
    with open(api_path.joinpath('.pages'), 'w') as f:
        f.write('title: API reference\narrange:\n  - overview.md\n  - ...\n')

    overview = open(api_path.joinpath('overview.md'), 'w')
    print(h1('Overview'), file=overview)

    linkifier = Linkifier()

    for mod_name, mod in inspect.getmembers(importlib.import_module('creme'), inspect.ismodule):

        if mod_name not in ('linear_model', 'evaluate', 'optim'):
            continue
        print(mod_name)

        print_module(mod, path=api_path, overview=overview)
