import importlib
import inspect

from creme import base


CATEGORIES = {
    'Binary classification': base.BinaryClassifier,
    'Multi-class classification': base.BinaryClassifier,
    'Regression': base.Regressor
}


if __name__ == '__main__':

    print('Making supervised.rst...')

    with open('supervised.rst', 'w') as f:

        f.write('Supervised learners\n')
        f.write('===================\n\n')

        for category, klass in CATEGORIES.items():
            f.write(f'{category}\n')
            f.write(f'{"-" * len(category)}\n\n')

            for mod_name in sorted(importlib.import_module('creme').__all__):

                mod = importlib.import_module(f'creme.{mod_name}', f)
                klasses = dict(inspect.getmembers(mod, inspect.isclass))
                klasses = {name: k for name, k in klasses.items() if issubclass(k, klass)}

                if klasses:
                    f.write(f':mod:`{mod_name}`\n\n')
                    for name in klasses:
                        f.write(f'- `{mod_name}.{name}`\n')
                    f.write('\n')

        print('Finished!')
