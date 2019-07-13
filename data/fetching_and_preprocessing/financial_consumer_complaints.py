# More info about the dataset can be found in the U.S. Government’s open data
# site (data.gov) https://catalog.data.gov/dataset/consumer-complaint-database
# or the Kaggle dataset page https://www.kaggle.com/sebastienverpile/consumercomplaintsdata/home

from urllib.request import urlretrieve
import pandas as pd
from os import remove

link = 'https://data.consumerfinance.gov/api/views/s6ew-h6mp/rows.csv'
raw_fname = 'data/financial_consumer_complaints_raw.csv'
output_fname = 'data/financial_consumer_complaints.csv'
separator = '|'

print('Downloading the Financial Consumer complaints dataset')
urlretrieve(link, raw_fname)

df = pd.read_csv('data/financial_consumer_complaints_raw.csv')

# Remove NA vals
print('All lines in raw data', len(df))
df = df[df['Consumer complaint narrative'].notna()]
print('Lines with existing text field', len(df))

# Keep only the text field and the product name
pertinent_cols = ['Consumer complaint narrative', 'Product']
df = df[pertinent_cols]

print("Products after consolidation: ", len(df.Product.value_counts()))
print("Breakdown:\n", df['Product'].value_counts().sort_values(ascending=False))

# preprocessing thanks to Susan Li from
# https://towardsdatascience.com/multi-class-text-classification-with-lstm-1590bee1bd17

# merge the various credit reporting products
df.loc[df['Product'] == 'Credit reporting, credit repair services, or other personal consumer reports', 'Product'] = 'Credit reporting'
# merge credit/prepaid card products
df.loc[df['Product'] == 'Credit card', 'Product'] = 'Credit/prepaid card'
df.loc[df['Product'] == 'Credit card or prepaid card', 'Product'] = 'Credit/prepaid card'
df.loc[df['Product'] == 'Prepaid card', 'Product'] = 'Credit/prepaid card'
# merge payday loan products
df.loc[df['Product'] == 'Payday loan, title loan, or personal loan', 'Product'] = 'Payday loan'
# merge virtual currency products
df.loc[df['Product'] == 'Money transfer, virtual currency, or money service', 'Product'] = 'Virtual currency'

# Remove "other" category which has too few cases anyway
df = df[df.Product != 'Other financial service']
df = df[df.Product != 'Money transfers']

print("Products after consolidation: ", len(df.Product.value_counts()))
print("Breakdown:\n", df['Product'].value_counts().sort_values(ascending=False))

# Remove the separator from the text fields
df['Consumer complaint narrative'].str.replace(separator, ' ')

# save to final output file
df.to_csv(output_fname, index=False, sep=separator)
remove(raw_fname)