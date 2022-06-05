import pandas as pd
df=pd.read_csv('adult-income-dataset.csv')
df['income_if_<=50k'] = df['income'].apply(lambda x: True if x == '<=50K' else False)
df['if_male'] = df['gender'].apply(lambda x: True if x == 'Male' else False)
df['income_if_<=50k']= df['income_if_<=50k'].astype(int)
df['if_male']= df['if_male'].astype(int)
df = df[df.workclass != '?']
df = df.reset_index(drop=True)

df.to_csv('evaluate_data.csv', index=False)