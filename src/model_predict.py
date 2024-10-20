import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from prediction_data_request import (
    get_forecast_wind_prod,
    get_forecast_sun_prod,
    get_forecast_elec_prices,
    get_forecast_weather,
    fmi_get_prev_x_days,
    wind_prod_get_prev_x_days,
    sun_prod_get_prev_x_days,
    get_prices_for_last_x_days,
    co2_get_prev_x_days

)

from data_combiner import combine

from data_functions import perform_linear_regression, predict_co2, check_data_quality


def get_forecasts(date):
    weather_forecast = get_forecast_weather(date)
    wind_prod_forecast = get_forecast_wind_prod(date)
    sun_prod_forecast = get_forecast_sun_prod(date)
    elec_prices_forecast = get_forecast_elec_prices(date)

    # MUST BE REMOVED!!!
    # # forecast is not available before 14:00, this line is only to allow functionality testint
    # elec_prices_forecast = simulate_el_prices_forecast(date)
    # REMOVE ABOVE LINE - FOR TESTING ONLY

    return [weather_forecast, wind_prod_forecast, sun_prod_forecast, elec_prices_forecast]


def get_prev_x_days(x=5):
    weather_history = fmi_get_prev_x_days(x)
    wind_prod_history = wind_prod_get_prev_x_days(x)
    solar_prod_history = sun_prod_get_prev_x_days(x)
    elec_prices_history = get_prices_for_last_x_days(x)
    co2_history = co2_get_prev_x_days(x)
    co2_history = interpolate_missing_co2(co2_history)
    return [weather_history, wind_prod_history, solar_prod_history, elec_prices_history, co2_history]


def convert_columns_to_int(df):
    columns_to_convert = ['year', 'month', 'day', 'hour', 'min']
    for col in columns_to_convert:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(
                'Int64')
    if 'min' in df.columns:
        df = df.drop(columns='min')
    return df


def get_combined_forecasts(date):
    forecasts = get_forecasts(date)
    forecasts = [convert_columns_to_int(df) for df in forecasts]

    df_handler = combine(dataframes=forecasts, get_handler=True)
    df_handler.initialize(avg_temp=True, create_dates=True,
                          to_periodic=True, normalize=True, drop_columns=True, decay=False)
    return df_handler.get_initialized_dataframe()


def get_combined_history(x=5):
    history = get_prev_x_days(x)
    history = [convert_columns_to_int(df) for df in history]

    df_handler = combine(dataframes=history, get_handler=True)
    df_handler.initialize(avg_temp=True, create_dates=True,
                          to_periodic=True, normalize=True, drop_columns=True, decay=False)
    return df_handler.get_initialized_dataframe()


def simulate_co2_data():
    start = datetime(2024, 10, 15, 0, 0)
    end = datetime(2024, 10, 19, 23, 54)
    timestamps = pd.date_range(start=start, end=end, freq='3T')

    co2_values = np.random.uniform(0, 20, size=len(timestamps))

    df = pd.DataFrame({
        'year': timestamps.year,
        'month': timestamps.month,
        'day': timestamps.day,
        'hour': timestamps.hour,
        'min': timestamps.minute,
        'CO2': co2_values
    })

    return df


def simulate_el_prices_forecast(date):
    start = date
    end = date + timedelta(days=1) - timedelta(hours=1)
    timestamps = pd.date_range(start=start, end=end, freq='1H')
    el_prices = np.random.uniform(0, 20, size=len(timestamps))
    df = pd.DataFrame({
        'timestamp': timestamps,
        'year': timestamps.year,
        'month': timestamps.month,
        'day': timestamps.day,
        'hour': timestamps.hour,
        'min': timestamps.minute,
        'electricity_cost': el_prices
    })

    return df


def interpolate_missing_co2(df):
    df = df.rename(columns={'min': 'minute'})

    df['timestamp'] = pd.to_datetime(
        df[['year', 'month', 'day', 'hour', 'minute']])

    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    full_timestamps = pd.date_range(
        start=start_time, end=end_time, freq='3min')

    full_df = pd.DataFrame({'timestamp': full_timestamps})

    df = pd.merge(full_df, df[['timestamp', 'CO2']],
                  on='timestamp', how='left')

    df['CO2'] = df['CO2'].interpolate(method='linear')

    df['year'] = df['timestamp'].dt.year.astype(str)
    df['month'] = df['timestamp'].dt.month.astype(str).str.zfill(2)
    df['day'] = df['timestamp'].dt.day.astype(str).str.zfill(2)
    df['hour'] = df['timestamp'].dt.hour.astype(str).str.zfill(2)
    df['minute'] = df['timestamp'].dt.minute.astype(str).str.zfill(2)

    df = df.drop(columns=['timestamp'])
    df = df[['year', 'month', 'day', 'hour', 'minute', 'CO2']]

    return df.rename(columns={'minute': 'min'})


def get_forecasts_and_predict():
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow = today + timedelta(days=1)
    print(f"tomorrow: {tomorrow}")
    combined_forecasts = get_combined_forecasts(tomorrow)
    combined_history = get_combined_history()
    model = perform_linear_regression(combined_history, 'CO2', print_=False)
    predicted_co2 = predict_co2(model, combined_forecasts, print_=False)

    return predicted_co2


if __name__ == "__main__":

    prediction = get_forecasts_and_predict()
    print("printout from main")
    print(prediction)
