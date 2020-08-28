import functools
import importlib
import inspect
import os
import pathlib
import re


def snake_to_kebab(snake: str) -> str:
    return snake.replace('_', '-')

def pascal_to_kebab(string):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', string)
    string = re.sub('(.)([0-9]+)', r'\1-\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', string).lower()

def build_index():

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

    between_fences = re.compile('`[\w\.]+\.\w+`')

    # TODO: one expression for between fences ``
    # TODO: one expression for in a type annotation

    def linkify(text, fences):
        path = path_index.get(text)
        name = name_index.get(text)
        if path and name:
            if fences:
                return f'[`{name}`]({path})'
            return f'[{name}]({path})'
        return None

    for path in pathlib.Path('docs/api-reference/').rglob('*.md'):

        print(path)

        with open(path) as f:
            md = f.read()
            md = between_fences.sub(lambda x: linkify(x.group().strip('`'), True) or x.group(), md)

        with open(path, 'w') as f:
            f.write(md)
