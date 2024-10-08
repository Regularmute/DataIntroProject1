from datetime import datetime, timedelta
import pandas as pd
from request_builder import get_fingrid_url, get_dataframe_by_url
from fmi_request import forecast_query
from fmi_config import forecast_places
from porssisahko_request import get_elec_pred_by_url

tomorrow_date = datetime.today().date() - timedelta(days=1)
predicted_elec_consumption_url = get_fingrid_url(165,
                                                 tomorrow_date.year,
                                                 tomorrow_date.month,
                                                 tomorrow_date.day,
                                                 0, 0,
                                                 tomorrow_date.year,
                                                 tomorrow_date.month,
                                                 tomorrow_date.day,
                                                 23, 59)

predicted_elec_consumption_df = get_dataframe_by_url(predicted_elec_consumption_url)
print("consumption")
print(predicted_elec_consumption_df.head())



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

predicted_elec_production_url = get_fingrid_url(241,
                                                 tomorrow_date.year,
                                                 tomorrow_date.month,
                                                 tomorrow_date.day,
                                                 0, 0,
                                                 tomorrow_date.year,
                                                 tomorrow_date.month,
                                                 tomorrow_date.day,
                                                 23, 59)

predicted_elec_production_df = get_dataframe_by_url(predicted_elec_production_url)

print("production")
print(predicted_elec_production_df.head())

predicted_elec_prices_url = 'https://api.porssisahko.net/v1/latest-prices.json'
predicted_elec_prices_df = get_elec_pred_by_url(predicted_elec_prices_url)
print("elcetricity price prediction")
print(predicted_elec_prices_df.head())

predicted_weather_df = forecast_query(forecast_places)
tomorrow_weather_df = predicted_weather_df[predicted_weather_df['timestamp'].dt.date == tomorrow_date + timedelta(days=2)]
print("weather prediction")
print(tomorrow_weather_df.head())