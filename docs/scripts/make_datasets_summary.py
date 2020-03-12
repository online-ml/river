import csv
import importlib
import inspect


if __name__ == '__main__':

    print('Making datasets.csv...')

    with open('datasets.csv', 'w') as f:

        w = csv.DictWriter(f, ['Name', 'Category', 'Samples', 'Features'])
        w.writeheader()

        for name, dataset in inspect.getmembers(importlib.import_module(f'creme.datasets'), inspect.isclass):
            dataset = dataset()
            w.writerow({
                'Name': f'`creme.datasets.{name}`',
                'Category': dataset.category,
                'Samples': f'{dataset.n_samples:,d}',
                'Features': f'{dataset.n_features:,d}',
            })
