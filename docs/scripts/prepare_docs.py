import importlib
import inspect
import os
import pathlib
import re
import shutil


def get_link(match):
    obj_name = match.group().strip('`')
    path = index.get(obj_name)
    if path:
        return f'[`{obj_name}`]({path})'
    return match.group()

def snake_to_kebab(snake: str) -> str:
    return snake.replace('_', '-')

def pascal_to_kebab(string):
    string = re.sub('(.)([A-Z][a-z]+)', r'\1-\2', string)
    string = re.sub('(.)([0-9]+)', r'\1-\2', string)
    return re.sub('([a-z0-9])([A-Z])', r'\1-\2', string).lower()


if __name__ == '__main__':

    shutil.rmtree('docs/build/', ignore_errors=True)
    shutil.copytree('docs', 'docs/build/')

    api_path = pathlib.Path('docs/build/api-reference')
    index = {}

    print('Loading modules...', end=' ', flush=True)
    modules = dict(inspect.getmembers(importlib.import_module('creme'), inspect.ismodule))
    modules = {
        'base': modules['base'],
        'linear_model': modules['linear_model'],
        'stream': modules['stream'],
        'optim': modules['optim']
    }
    print('done')

    def index_module(mod_name, mod, index, path):
        path = os.path.join(path, mod_name)

        for func_name, _ in inspect.getmembers(mod, inspect.isfunction):
            index[f'{mod_name}.{func_name}'] = os.path.join(path, snake_to_kebab(func_name))
            index[f'{path}.{func_name}'.replace('/', '.')] = index[f'{mod_name}.{func_name}']

        for klass_name, _ in inspect.getmembers(mod, inspect.isclass):
            index[f'{mod_name}.{klass_name}'] = os.path.join(path, pascal_to_kebab(klass_name))
            index[f'{path}.{klass_name}'.replace('/', '.')] = index[f'{mod_name}.{klass_name}']

        for submod_name, submod in inspect.getmembers(mod, inspect.ismodule):
            if submod_name not in mod.__all__ or submod_name == 'typing':
                continue
            index[f'{mod_name}.{submod_name}'] = os.path.join(path, snake_to_kebab(submod_name))
            index[f'{path}.{submod_name}'.replace('/', '.')] = index[f'{mod_name}.{submod_name}']

            # recurse
            index = index_module(submod_name, submod, index, path=path)

        return index

    for mod_name, mod in modules.items():
        index[mod_name] = snake_to_kebab(mod_name)
        index_module(mod_name, mod, index, path='')

    # prepend the location of the API reference to each path
    for k, v in index.items():
        index[k] = os.path.join('/api-reference', v)

    # prepend creme
    for k in list(index.keys()):
        index[f'creme.{k}'] = index[k]

    fence_pattern = re.compile('`(.+?)`')

    for path in pathlib.Path('docs/build').rglob('*.md'):

        print(f'Preparing {path}...', end=' ', flush=True)

        with open(path) as f:
            md = f.read()
            md = fence_pattern.sub(get_link, md)

        with open(path, 'w') as f:
            f.write(md)

        print('done')
