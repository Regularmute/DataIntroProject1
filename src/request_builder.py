import requests
import os
import pandas as pd
import json
from dotenv import load_dotenv
from io import StringIO

load_dotenv()
headers = {'Cache-Control':'no-cache', 'x-api-key': os.getenv('FINGRID_API_KEY')}

def get_fingrid_url(datasetId, s_year, s_month, s_day, s_hour, s_min, e_year, e_month, e_day, e_hour, e_min):
    startTime = f"{s_year}-{s_month:02d}-{s_day:02d}T{s_hour:02d}:{s_min:02d}:00Z"
    endTime = f"{e_year}-{e_month:02d}-{e_day:02d}T{e_hour:02d}:{e_min:02d}:00Z"
    pageSize = ((60-s_min)/3) + (e_hour-(s_hour+1))*20 + (e_min/3)
    return f'https://data.fingrid.fi/api/datasets/{datasetId}/data?startTime={startTime}&endTime={endTime}&format=json&pageSize={int(pageSize)}'

def load_fingrid_data():
    pass

def load_weather_data():
    pass

def main():
    url = get_fingrid_url(266,2024,6,10,9,15,2024,6,10,12,0)
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        df = pd.read_json(StringIO(json.dumps(r.json()['data'])))
        time = df.startTime.str.split(r'[:\-\.TZ]', expand=True).iloc[:,0:5]
        time.columns = ['year', 'month', 'day', 'hour', 'min']
        df = pd.concat([time, df.value], axis=1)
        print(df)
    else:
        print("Error", r.status_code)
        result_json = r.json()
        print(result_json['message'])
        print(result_json['details'])

if __name__ == "__main__":
    main()