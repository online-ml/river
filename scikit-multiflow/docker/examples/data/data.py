import pandas as pd
from sklearn import preprocessing
import urllib.request

# Download Electricity dataset.
# Author: M. Harries, J. Gama, A. Bifet
# Link: https://www.openml.org/d/151
url = 'https://www.openml.org/data/get_csv/2419/electricity-normalized.arff'
urllib.request.urlretrieve(url, 'elec.csv')

# Preprocess dataset
df = pd.read_csv('elec.csv')
df = df.drop(['date', 'day'], axis=1)
le = preprocessing.LabelEncoder()
df['class'] = le.fit_transform(df['class'])
df.to_csv('elec.csv', header=False, index=False)
