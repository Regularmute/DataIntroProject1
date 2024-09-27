import pandas as pd

def get_dummy_df():
    columns_to_fetch = ['year', 'month', 'day', 'hour', 'day_of_week', 'CO2', 'consumption',
                        'production', 'solar prediction', 'wind', 'hydro', 'district',
                        'electricity_cost', 'Vantaa Helsinki-Vantaan lentoasema temperature',
                        'Vaasa lentoasema temperature', 'Liperi Joensuu lentoasema temperature',
                        'Jyväskylä lentoasema temperature', 'Rovaniemi lentoasema AWOS temperature']
    dummy_df = pd.read_csv('/app/src/data_2023.csv', usecols=columns_to_fetch)
    return dummy_df
