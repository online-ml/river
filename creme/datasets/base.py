import os
import shutil
import urllib
import zipfile

from .. import stream


def get_data_home(data_home=None):
    """Return the path of the creme data directory."""
    if data_home is None:
        data_home = os.environ.get('CREME_DATA', os.path.join('~', 'creme_data'))
        data_home = os.path.expanduser(data_home)
        if not os.path.exists(data_home):
            os.makedirs(data_home)
    return data_home


def fetch_csv_dataset(data_home, url, name, **iter_csv_params):

    data_home = get_data_home(data_home=data_home)

    # If the CSV file exists then iterate over it
    csv_path = os.path.join(data_home, f'{name}.csv')
    if os.path.exists(csv_path):
        return stream.iter_csv(csv_path, **iter_csv_params)

    # If the ZIP file exists then unzip it
    zip_path = os.path.join(data_home, f'{name}.zip')
    if os.path.exists(zip_path):
        print('Unzipping data...')

        # Unzip the ZIP file
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(data_home)

        # Delete the ZIP file now that the CSV file is available
        os.remove(zip_path)

        return fetch_csv_dataset(data_home, url, name, **iter_csv_params)

    # Download the ZIP file
    print('Downloading data...')
    with urllib.request.urlopen(url) as r, open(zip_path, 'wb') as f:
        shutil.copyfileobj(r, f)

    return fetch_csv_dataset(data_home, url, name, **iter_csv_params)
