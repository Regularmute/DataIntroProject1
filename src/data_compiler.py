import pandas as pd
from datetime import datetime


dow = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

paths = {
    'CO2':'data/CO2_emis_coef_2023.csv',
    'consumption':'data/elec_cons_finland_2023.csv',
    'production':'data/elec_prod_finland_2023.csv',
    'solar prediction':'data/solar_prod_pred_2023.csv',
    'wind':'data/wind_prod_realtime_2023.csv',
    'hydro': 'data/hydro_prod_realtime_2023.csv',
    'district': 'data/dist_heat_prod_realtime_2023.csv',
    # 'electricity price': 'data/electricity_prices_2023.csv'
}

dataframes = {}

headers = True

# Read original dataframes
for var in paths:
    if var == 'electricity price':
        df = pd.read_csv(paths[var])    
    else:
        df = pd.read_csv(paths[var], sep=';')
    if 'datasetId' in df.columns:
        df = df.drop(columns='datasetId')    
    if 'endTime' in df.columns:
        df = df.drop(columns='endTime')    
    
    df[var] = df[df.columns[1]]
    df = df.drop(df.columns[1], axis=1)

    if 'startTime' in df.columns:
        df['timestamp'] = pd.to_datetime(df['startTime'])
    elif 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    dataframes[var] = df


for mon in range(1, 2):

    for day in range(1, 2):
        
        fdf = pd.DataFrame()

        for var in dataframes:
            df = dataframes[var]

            df = df[(df.timestamp.dt.month == mon) & (df.timestamp.dt.day == day)]
            if df.empty:
                break
            if 'year' not in df.columns:
                df['year'] = df['timestamp'].dt.year
                df['month'] = df['timestamp'].dt.month
                df['day'] = df['timestamp'].dt.day
                df['hour'] = df['timestamp'].dt.hour
            
            df = df.drop(columns='timestamp')

            df = df.groupby([df.year, df.month, df.day, df.hour]).mean()
            if 'day_of_week' in df.columns:
                df['day_of_week'] = df.day_of_week.astype(int)
            if fdf.empty:
                fdf = df
            else:
                fdf = pd.merge(fdf, df, on=['year', 'month', 'day', 'hour'])


        if fdf.empty:
            continue

        # read and handle FMI dataframe
        df = pd.read_csv('data/jan23_annual_weather_data.csv')
        df['day_of_week'] = df['day_of_week'].apply(lambda x: dow[x])
        df = df.drop(columns=['minute', 'timestamp', 'day_of_week'])
        fdf = pd.merge(fdf, df, on=['year', 'month', 'day', 'hour'])

        # write data (sequential write) to file       
        fdf.to_csv('data/data_2023.csv', mode='a', header=headers, index=False)

        if headers:
            headers = False

    print('Month',mon,'ready')
