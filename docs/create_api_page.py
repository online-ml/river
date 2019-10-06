import importlib
import inspect


def write_module(mod, f):

    lines = mod.__doc__.splitlines()

    # Write the title, which is the first line of the module's docstring
    title = lines[0].rstrip('.')
    header = f'**{mod.__name__.split(".")[-1]}**: {title}\n'
    f.write(header)
    f.write('-' * len(header) + '\n\n')

    # Write the description, which is the rest of the lines of the module's docstring
    if len(lines) > 2:
        f.write(' '.join(lines[2:]) + '\n\n')

    # Print classes
    classes = [name for name, _ in inspect.getmembers(mod, inspect.isclass) if name in mod.__all__]
    if classes:
        f.write('.. rubric:: Classes\n\n')
        f.write(f'.. currentmodule:: creme.{mod_name}\n')
        f.write('''.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        \n''')
        for klass in sorted(classes):
            f.write(f'    {klass}\n')
        f.write('\n')

    # Print functions
    funcs = [name for name, _ in inspect.getmembers(mod, inspect.isfunction) if name in mod.__all__]
    if funcs:
        f.write('.. rubric:: Functions\n\n')
        f.write(f'.. currentmodule:: creme.{mod_name}\n')
        f.write('''.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: function.rst
        \n''')
        for func in sorted(funcs):
            f.write(f'    {func}\n')
        f.write('\n')

    # Print submodules
    sub_mods = [name for name, _ in inspect.getmembers(mod, inspect.ismodule) if name in mod.__all__]
    for sub_mod_name in sub_mods:
        sub_mod = importlib.import_module(f'{mod.__name__}.{sub_mod_name}')
        desc = sub_mod.__doc__.splitlines()[0].rstrip('.')

        f.write(desc + '\n')
        f.write('~' * len(desc) + '\n\n')
        f.write(f'''.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
        \n''')

        klasses = [name for name, _ in inspect.getmembers(sub_mod, inspect.isclass) if name in sub_mod.__all__]
        for klass in sorted(klasses):
            f.write(f'    {sub_mod_name}.{klass}\n')
        f.write('\n')


if __name__ == '__main__':

    modules = importlib.import_module('creme').__all__

    with open('api.rst', 'w') as f:

        print('Compiling api.rst...')

        f.write('API reference\n')
        f.write('=============\n\n')

        for mod_name in sorted(modules):

            print(f'Adding {mod_name}')

            mod = importlib.import_module(f'creme.{mod_name}', f)
            write_module(mod, f)

            f.write('\n')
        print('Finished!')
