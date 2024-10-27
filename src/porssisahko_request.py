import requests
import pandas as pd
import json
from io import StringIO


def get_elec_pred_by_url_and_date(url):
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.read_json(StringIO(json.dumps(r.json())))
        df['time_start'] = pd.to_datetime(df['time_start'], utc=True)

        df['time_start'] = df['time_start'].dt.tz_convert('Europe/Helsinki')
        print(df.head())
        date_time_string = df['time_start'].dt.strftime('%Y-%m-%d-%H:%M')
        formatted_date_time = date_time_string.str.split(
            r'[:\-\.TZ]', expand=True).iloc[:, 0:5]
        formatted_date_time.columns = ['year', 'month', 'day', 'hour', 'min']
        df['EUR_per_kWh'] = df['EUR_per_kWh'] * 100 * 1.255
        df = pd.concat([formatted_date_time, df.EUR_per_kWh], axis=1)

        return (df)
    else:
        print("Error", r.status_code)
