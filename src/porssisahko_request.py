import requests
import pandas as pd
import json
from io import StringIO

def get_elec_pred_by_url(url):
    r = requests.get(url)
    if r.status_code == 200:
        df = pd.read_json(StringIO(json.dumps(r.json()['prices'])))
        formatted_date_time = df.startDate.str.split(r'[:\-\.TZ]', expand=True).iloc[:,0:5]
        formatted_date_time.columns = ['year', 'month', 'day', 'hour', 'min']
        df = pd.concat([formatted_date_time, df.price], axis=1)
        return(df)
    else:
        print("Error", r.status_code)
        result_json = r.json()
        print(result_json)