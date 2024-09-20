import pandas as pd

df = pd.read_excel('data/electricity_prices_2023.xlsx')

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

df['year'] = df['timestamp'].dt.year.astype('Int64')
df['month'] = df['timestamp'].dt.month.astype('Int64')
df['day'] = df['timestamp'].dt.day.astype('Int64')
df['hour'] = df['timestamp'].dt.hour.astype('Int64')
df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('Int64')

df.to_csv('data/electricity_prices_2023.csv', index=False)
