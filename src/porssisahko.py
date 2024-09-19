import pandas as pd

df = pd.read_excel('data/electricity_prices_2023.xlsx')

df['datetime'] = pd.to_datetime(df['datetime'], format='%d/%m/%Y %H:%M:%S', errors='coerce')

df['date'] = df['datetime'].dt.date
df['time'] = df['datetime'].dt.time

df = df.drop(columns=['datetime'])

print(df.iloc[:5, :])
print(df.head())