import pandas as pd
from datetime import datetime


paths = {
    'CO2':'data/CO2_emis_coef_2023.csv',
    'consumption':'data/elec_cons_finland_2023.csv',
    'production':'data/elec_prod_finland_2023.csv',
    'solar prediction':'data/solar_prod_pred_2023.csv',
    'wind':'data/wind_prod_realtime_2023.csv'
}


dataframes = {}

headers = True

# Read original dataframes
for var in paths:
    df = pd.read_csv(paths[var], sep=';')
    if 'datasetId' in df.columns:
        df = df.drop(columns='datasetId')    
    if 'endTime' in df.columns:
        df = df.drop(columns='endTime')    
    
    df[var] = df[df.columns[1]]
    df = df.drop(df.columns[1], axis=1)
    df['startTime'] = pd.to_datetime(df['startTime'])
    dataframes[var] = df


for mon in range(1, 13):

    for day in range(1,32):
        
        fdf = pd.DataFrame()

        for var in dataframes:
            df = dataframes[var]
            df = df[(df.startTime.dt.month == mon) & (df.startTime.dt.day == day)]
            if df.empty:
                break
            df = df.groupby(df.startTime.dt.hour).mean()

            df['year'] = df['startTime'].dt.year
            df['month'] = df['startTime'].dt.month
            df['day'] = df['startTime'].dt.day
            df['hour'] = df['startTime'].dt.hour
            df = df.drop(columns='startTime')

            if fdf.empty:
                fdf = df
            else:
                fdf = pd.merge(fdf, df, on=['year', 'month', 'day', 'hour'])
        if fdf.empty:
            continue
        fdf = fdf.set_index(['year', 'month', 'day', 'hour'])
        fdf.to_csv('data/data_2023.csv', mode='a', header=headers)
        if headers:
            headers = False

    print('Month',mon,'ready')
