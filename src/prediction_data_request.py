from datetime import datetime, timedelta
from request_builder import get_fingrid_url, get_dataframe_by_url
from fmi_request import forecast_query, format_forecast_df
from fmi_config import forecast_places
from porssisahko_request import get_elec_pred_by_url_and_date

# Fingrid API returns results in UTC, fetches by UTC+2

def get_forecast_wind_prod(date):
    predicted_elec_wind_prod_url = get_fingrid_url(246,
                                                    date.year,
                                                    date.month,
                                                    date.day,
                                                    0, 0,
                                                    date.year,
                                                    date.month,
                                                    date.day,
                                                    23, 59)
    predicted_elec_wind_prod_df = get_dataframe_by_url(predicted_elec_wind_prod_url)
    return predicted_elec_wind_prod_df

def get_forecast_sun_prod(date):
    predicted_elec_sun_prod_url = get_fingrid_url(247,
                                                    date.year,
                                                    date.month,
                                                    date.day,
                                                    0, 0,
                                                    date.year,
                                                    date.month,
                                                    date.day,
                                                    23, 59)
    predicted_elec_sun_prod_df = get_dataframe_by_url(predicted_elec_sun_prod_url)

    return predicted_elec_sun_prod_df

def get_forecast_elec_prices(date):
    predicted_elec_prices_url = 'https://api.porssisahko.net/v1/latest-prices.json'
    predicted_elec_prices_df = get_elec_pred_by_url_and_date(predicted_elec_prices_url, date.strftime('%Y-%m-%d'))
    return predicted_elec_prices_df

def get_forecast_weather(date):
    start_time = datetime(date.year,
                            date.month,
                            date.day,
                            0, 0, 0).strftime('%Y-%m-%dT%H:%M:%S')
    end_time = datetime(date.year,
                            date.month,
                            date.day,
                            23, 59, 59).strftime('%Y-%m-%dT%H:%M:%S')

    predicted_weather_df = forecast_query(forecast_places, start_time, end_time)
    formatted_weather_df = format_forecast_df(predicted_weather_df)

    return formatted_weather_df
