import pandas as pd
import prediction_data_request as pdr
import data_combiner
import datetime as dt

def get_dummy_df():
    columns_to_fetch = ['year', 'month', 'day', 'hour', 'day_of_week', 'CO2', 'consumption',
                        'production', 'solar prediction', 'wind', 'hydro', 'district',
                        'electricity_cost', 'Vantaa Helsinki-Vantaan lentoasema temperature',
                        'Vaasa lentoasema temperature', 'Liperi Joensuu lentoasema temperature',
                        'Jyväskylä lentoasema temperature', 'Rovaniemi lentoasema AWOS temperature']
    dummy_df = pd.read_csv('/app/src/data_2023.csv', usecols=columns_to_fetch)
    return dummy_df

def main():
    tomorrow = dt.datetime.today().date() + dt.timedelta(days=1)

    combined_df = data_combiner.combine([pdr.get_forecast_wind_prod(tomorrow),
                           pdr.get_forecast_sun_prod(tomorrow),
                           pdr.get_forecast_elec_prices(tomorrow),
                           pdr.get_forecast_weather(tomorrow),
                           pdr.fmi_get_prev_x_days(5),
                           pdr.wind_prod_get_prev_x_days(5),
                           pdr.sun_prod_get_prev_x_days(5),
                           pdr.get_prices_for_last_x_days(5)])
    combined_df.to_csv('/app/src/combined_data.csv', index=False)
    print(combined_df.head(24), combined_df.shape)


if __name__ == "__main__":
    main()
