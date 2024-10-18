from datetime import datetime, timedelta
import pandas as pd
from request_builder import get_fingrid_url, get_dataframe_by_url
from fmi_request import forecast_query, format_forecast_df
from fmi_config import forecast_places
from porssisahko_request import get_elec_pred_by_url_and_date

# Local time tomorrow_date
tomorrow_date = datetime.today().date() + timedelta(days=1)
print(datetime.today().time())
tomorrow_start_time = datetime(tomorrow_date.year,
                                      tomorrow_date.month,
                                      tomorrow_date.day,
                                      0, 0, 0).strftime('%Y-%m-%dT%H:%M:%S')
tomorrow_end_time = datetime(tomorrow_date.year,
                                      tomorrow_date.month,
                                      tomorrow_date.day,
                                      23, 59, 59).strftime('%Y-%m-%dT%H:%M:%S')
# Fingrid API returns results in UTC, fetches by UTC+2


predicted_elec_wind_prod_url = get_fingrid_url(246,
                                                tomorrow_date.year,
                                                tomorrow_date.month,
                                                tomorrow_date.day,
                                                0, 0,
                                                tomorrow_date.year,
                                                tomorrow_date.month,
                                                tomorrow_date.day,
                                                23, 59)
predicted_elec_wind_prod_df = get_dataframe_by_url(predicted_elec_wind_prod_url)

print("wind prediction")
print(predicted_elec_wind_prod_df.head())

predicted_elec_sun_prod_url = get_fingrid_url(247,
                                                tomorrow_date.year,
                                                tomorrow_date.month,
                                                tomorrow_date.day,
                                                0, 0,
                                                tomorrow_date.year,
                                                tomorrow_date.month,
                                                tomorrow_date.day,
                                                23, 59)
predicted_elec_sun_prod_df = get_dataframe_by_url(predicted_elec_sun_prod_url)

print("sun prediction")
print(predicted_elec_sun_prod_df.head())

predicted_elec_prices_url = 'https://api.porssisahko.net/v1/latest-prices.json'
predicted_elec_prices_df = get_elec_pred_by_url_and_date(predicted_elec_prices_url, tomorrow_date.strftime('%Y-%m-%d'))
print("electricity price prediction")
print(predicted_elec_prices_df.to_string())

predicted_weather_df = forecast_query(forecast_places, tomorrow_start_time, tomorrow_end_time)
formatted_weather_df = format_forecast_df(predicted_weather_df)

print("weather prediction")
print(formatted_weather_df.head())
formatted_weather_df.to_csv('data/weather_prediction.csv', index=False)
