import pandas as pd

df = pd.read_csv('data/electricity_prices_2023.csv')

df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

df['timestamp'] = df['timestamp'].dt.tz_localize(
    'Europe/Helsinki', ambiguous='infer').dt.tz_convert('UTC')
df['timestamp'] = df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
df['timestamp'] = pd.to_datetime(df['timestamp'])

df['electricity_cost'] = df['electricity_cost'].str.replace(',', '.').astype('float64')
df['year'] = df['timestamp'].dt.year.astype('Int64')
df['month'] = df['timestamp'].dt.month.astype('Int64')
df['day'] = df['timestamp'].dt.day.astype('Int64')
df['hour'] = df['timestamp'].dt.hour.astype('Int64')
df['day_of_week'] = df['timestamp'].dt.dayofweek.astype('Int64')

df.to_csv('data/elec_prices_2023.csv', index=False)
