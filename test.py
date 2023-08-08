import pandas as pd

# Merge processed data in heart+disease

df1 = pd.read_csv('./heart+disease/processed.cleveland.data', header=None)
# Add region name
df1['region'] = 'cleveland'

df2 = pd.read_csv('./heart+disease/processed.hungarian.data', header=None)
df2['region'] = 'hungarian'

df3 = pd.read_csv('./heart+disease/processed.switzerland.data', header=None)
df3['region'] = 'switzerland'

df4 = pd.read_csv('./heart+disease/processed.va.data', header=None)
df4['region'] = 'va'

df = pd.concat([df1, df2, df3, df4], ignore_index=True)
# Rename columns
df.columns = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num', 'region'
]

print(df.head())
# Save to csv
df.to_csv('./heart+disease/processed.csv', index=False)