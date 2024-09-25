import pandas as pd
from datetime import datetime


dow = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

# Single variable data
paths = {
    'CO2':'data/CO2_emis_coef_2023.csv',
    'consumption':'data/elec_cons_finland_2023.csv',
    'production':'data/elec_prod_finland_2023.csv',
    'solar prediction':'data/solar_prod_pred_2023.csv',
    'wind':'data/wind_prod_realtime_2023.csv',
    'hydro': 'data/hydro_prod_realtime_2023.csv',
    'district': 'data/dist_heat_prod_realtime_2023.csv',
    'electricity price': 'data/elec_prices_2023.csv',
    'temperatures': 'data/annual_weather_data_2023.csv'
}

dataframes = {}

headers = True

# Read original dataframes
for var in paths:
    # Read/drop electricity price data
    if var == 'electricity price':
        df = pd.read_csv(paths[var])    
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    elif var == 'temperatures':
        df = pd.read_csv(paths[var])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.drop(columns=['year','month','day','hour','minute','day_of_week'])
    # --- Fingrid data
    else:
        df = pd.read_csv(paths[var], sep=';')
        if 'datasetId' in df.columns:
            df = df.drop(columns=['datasetId'])
        df = df.drop(columns='endTime')
        df[var] = df[df.columns[1]]
        df = df.drop(df.columns[1], axis=1)
        df['timestamp'] = pd.to_datetime(df['startTime'])
        df = df.drop(columns='startTime')
    

    dataframes[var] = df


for mon in range(1, 13):

    for day in range(1, 32):
        
        fdf = pd.DataFrame()

        for var in dataframes:
            df = dataframes[var]
            df = df[(df.timestamp.dt.month == mon) & (df.timestamp.dt.day == day)]
            
            if df.empty:
                break

            df = df.groupby(df.timestamp.dt.hour).mean().reset_index(drop=True)

            df['date'] = df.timestamp.dt.date
            df['hour'] = df.timestamp.dt.hour
            df = df.drop(columns='timestamp')
            if 'day_of_week' in df.columns:
                df['day_of_week'] = df.day_of_week.astype(int)

            if fdf.empty:
                fdf = df
            else:
                df = df.drop(columns='date')
                fdf = pd.merge(fdf, df, on=['hour'])
            
        if fdf.empty:
            continue
        # read and handle FMI dataframe
        # df = pd.read_csv('data/annual_weather_data_2023.csv')
        # # df['day_of_week'] = df['day_of_week'].apply(lambda x: dow[x])
        # df = df.drop(columns=['minute', 'timestamp', 'day_of_week'])
        # fdf = pd.merge(fdf, df, on=['year', 'month', 'day', 'hour'])

        # write data (sequential write) to file       
        fdf.to_csv('data/data_2023.csv', mode='a', header=headers, index=False)

        if headers:
            headers = False

    print('Month',mon,'ready')
