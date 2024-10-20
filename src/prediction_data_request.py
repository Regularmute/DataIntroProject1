from datetime import datetime, timedelta
from request_builder import get_fingrid_url, get_dataframe_by_url
from fmi_request import forecast_query, format_forecast_df, history_query
from fmi_config import forecast_places, stations, timestep
from porssisahko_request import get_elec_pred_by_url_and_date
import pandas as pd
import requests

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
    predicted_elec_wind_prod_df = get_dataframe_by_url(
        predicted_elec_wind_prod_url)
    predicted_elec_wind_prod_df.rename(columns={'value': 'wind'}, inplace=True)
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
    predicted_elec_sun_prod_df = get_dataframe_by_url(
        predicted_elec_sun_prod_url)
    predicted_elec_sun_prod_df.rename(
        columns={'value': 'solar prediction'}, inplace=True)
    return predicted_elec_sun_prod_df


def get_forecast_elec_prices(date):
    predicted_elec_prices_url = 'https://api.porssisahko.net/v1/latest-prices.json'
    predicted_elec_prices_df = get_elec_pred_by_url_and_date(
        predicted_elec_prices_url, date.strftime('%Y-%m-%d'))
    predicted_elec_prices_df.rename(
        columns={'price': 'electricity_cost'}, inplace=True)
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

    predicted_weather_df = forecast_query(
        forecast_places, start_time, end_time)
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
        historical_elec_wind_prod_df = get_dataframe_by_url(
            historical_elec_wind_prod_url)
        historical_elec_wind_prod_df.rename(
            columns={'value': 'wind'}, inplace=True)
        return historical_elec_wind_prod_df
    except Exception as e:
        raise e


def co2_get_prev_x_days(x=5):
    end_time = datetime.now().replace(hour=23, minute=0, second=0,
                                      microsecond=0) - timedelta(days=1)

    start_time = end_time.replace(
        hour=0, minute=0, second=0, microsecond=0) - timedelta(days=4)
    try:
        historical_co2_url = get_fingrid_url(266,
                                             start_time.year,
                                             start_time.month,
                                             start_time.day,
                                             0, 0,
                                             end_time.year,
                                             end_time.month,
                                             end_time.day,
                                             23, 59)
        historical_co2_df = get_dataframe_by_url(historical_co2_url)
        historical_co2_df.rename(columns={'value': 'CO2'}, inplace=True)
        return historical_co2_df
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
        historical_elec_sun_prod_df = get_dataframe_by_url(
            historical_elec_sun_prod_url)
        historical_elec_sun_prod_df.rename(
            columns={'value': 'solar prediction'}, inplace=True)
        return historical_elec_sun_prod_df
    except Exception as e:
        raise e


def fetch_prices_for_day(year, month, day):
    url = f"https://www.sahkonhintatanaan.fi/api/v1/prices/{year}/{month}-{day}.json"
    response = requests.get(url)

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch data for {year}-{month}-{day}")
        return None


def get_prices_for_last_x_days(x=5):
    df_all_days = pd.DataFrame()
    today = datetime.now()

    for i in range(1, x+1):
        date = today - timedelta(days=i)
        year = date.year
        month = f"{date.month:02d}"
        day = f"{date.day:02d}"
        prices_data = fetch_prices_for_day(year, month, day)

        if prices_data:
            df_day = pd.DataFrame(prices_data)
            df_all_days = pd.concat([df_all_days, df_day], ignore_index=True)

    df_all_days['time_start'] = pd.to_datetime(df_all_days['time_start'])
    df_all_days['timestamp'] = df_all_days['time_start']
    df_all_days['year'] = df_all_days['time_start'].dt.year
    df_all_days['month'] = df_all_days['time_start'].dt.month
    df_all_days['day'] = df_all_days['time_start'].dt.day
    df_all_days['hour'] = df_all_days['time_start'].dt.hour

    # convert unit and add VAT (to be comparable with other data)
    df_all_days['electricity_cost'] = (
        df_all_days['EUR_per_kWh']*100*1.255).astype(float)

    df_all_days.drop(
        columns=['time_start', 'time_end', 'EUR_per_kWh'], inplace=True)

    return df_all_days
