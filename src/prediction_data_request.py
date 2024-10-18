from datetime import datetime, timedelta
from request_builder import get_fingrid_url, get_dataframe_by_url
from fmi_request import forecast_query, format_forecast_df, history_query
from fmi_config import forecast_places, stations, timestep
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

def fmi_get_prev_x_days(x=5):
    end_time = datetime.now().replace(hour=23, minute=0, second=0,
                                      microsecond=0) - timedelta(days=1)

    start_time = end_time.replace(
        hour=0, minute=0, second=0, microsecond=0) - timedelta(days=4)

    start_iso = start_time.isoformat(timespec="seconds")
    end_iso = end_time.isoformat(timespec="seconds")

    try:
        df = history_query(start_iso, end_iso, timestep, stations)
    except Exception as e:
        raise e
    df.to_csv('data/fmi_prev_days.csv', index=False)

    return df

def wind_prod_get_prev_x_days(x=5):
    end_time = datetime.now().replace(hour=23, minute=0, second=0,
                                      microsecond=0) - timedelta(days=1)

    start_time = end_time.replace(
        hour=0, minute=0, second=0, microsecond=0) - timedelta(days=4)

    try:
        historical_elec_wind_prod_url = get_fingrid_url(246,
                                                        start_time.year,
                                                        start_time.month,
                                                        start_time.day,
                                                        0, 0,
                                                        end_time.year,
                                                        end_time.month,
                                                        end_time.day,
                                                        23, 59)
        historical_elec_wind_prod_df = get_dataframe_by_url(historical_elec_wind_prod_url)
        return historical_elec_wind_prod_df
    except Exception as e:
        raise e

def sun_prod_get_prev_x_days(x=5):
    end_time = datetime.now().replace(hour=23, minute=0, second=0,
                                      microsecond=0) - timedelta(days=1)

    start_time = end_time.replace(
        hour=0, minute=0, second=0, microsecond=0) - timedelta(days=4)

    try:
        historical_elec_sun_prod_url = get_fingrid_url(247,
                                                        start_time.year,
                                                        start_time.month,
                                                        start_time.day,
                                                        0, 0,
                                                        end_time.year,
                                                        end_time.month,
                                                        end_time.day,
                                                        23, 59)
        historical_elec_sun_prod_df = get_dataframe_by_url(historical_elec_sun_prod_url)
        return historical_elec_sun_prod_df
    except Exception as e:
        raise e
