import requests
import pandas as pd
import json
from io import StringIO

def get_elec_pred_by_url_and_date(url, date):
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.read_json(StringIO(json.dumps(r.json()['prices'])))
        formatted_date_time = df.startDate.str.split(r'[:\-\.TZ]', expand=True).iloc[:,0:5]
        formatted_date_time.columns = ['year', 'month', 'day', 'hour', 'min']
        df = pd.concat([formatted_date_time, df.price], axis=1)

        # Filter rows to include only those from tomorrow's date
        df['date'] = df[['year', 'month', 'day']].apply(lambda x: '-'.join(x), axis=1)
        df_tomorrow = df[df['date'] == date]

        # Drop the temporary 'date' column
        df_tomorrow = df_tomorrow.drop(columns=['date'])

        return(df_tomorrow)
    else:
        print("Error", r.status_code)
        result_json = r.json()
        print(result_json)