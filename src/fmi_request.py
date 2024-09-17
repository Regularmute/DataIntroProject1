"""Module for requesting data from FMI API."""

import datetime as dt
import pandas as pd
from fmi_config import bbox, stations, timestep, annual_data, annual_data, forecast_places

from fmiopendata.wfs import download_stored_query


def fmi_history_query(start_time, end_time, timestep):
    """Raw query to FMI API for historical weather data."""
    obs = download_stored_query("fmi::observations::weather::multipointcoverage",
                                args=[f"bbox={bbox}",
                                      "timestep=" + timestep,
                                      "starttime=" + start_time,
                                      "endtime=" + end_time])
    return obs


def history_query(start_time, end_time, timestep, stations):
    """Make pandas DataFrame from historical weather data."""

    observations = fmi_history_query(start_time, end_time, timestep)
    data = []

    for t in observations.data.keys():
        timestamp = pd.to_datetime(t)
        row = {
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'hour': timestamp.hour,
            'minute': timestamp.minute
        }
        for station in stations:
            station_col = f"{station} temperature"
            row[station_col] = observations.data[t][station]["Air temperature"]["value"]
        data.append(row)

    df = pd.DataFrame(data)
    return df


def split_time_intervals(start_time, end_time, interval_hours=48):
    """Split time interval into smaller intervals.
    FMI API limits to 168 hours, in practice needs to be shorter due to server load."""
    intervals = []
    current_start = start_time

    while current_start < end_time:
        current_end = min(
            current_start + dt.timedelta(hours=interval_hours), end_time)
        intervals.append((current_start, current_end))
        current_start = current_end
    return intervals


def collect_yearly_data(start_time, end_time, timestep, stations):
    """Collecs historical weather data for a year (or other given interval)."""
    intervals = split_time_intervals(start_time, end_time)
    all_data = pd.DataFrame()

    for start, end in intervals:
        start_iso = start.isoformat(timespec="seconds") + "Z"
        end_iso = end.isoformat(timespec="seconds") + "Z"
        print(f"Collecting data from {start_iso} to {end_iso}")
        df = history_query(start_iso, end_iso, timestep, stations)
        all_data = pd.concat([all_data, df], ignore_index=True)

    return all_data


def save_dataframe_to_csv(df, filename):
    """History dataframe to CSV"""
    df.to_csv(filename, index=False)


def load_dataframe_from_csv(filename):
    """History dataframe from CSV"""
    return pd.read_csv(filename)


def load_test():
    """driver or tester for loading history data"""
    start_time = dt.datetime(2023, 1, 1, 0, 0, 0)
    end_time = dt.datetime(2023, 12, 31, 0, 0, 0)

    temperature_history = collect_yearly_data(
        start_time, end_time, timestep, stations)
    save_dataframe_to_csv(temperature_history, annual_data)
    loaded_data = load_dataframe_from_csv(annual_data)
    print(loaded_data)


def fmi_forecast_query(place):
    """Raw query to FMI API for forecast data."""

    fcst = download_stored_query(f"fmi::forecast::harmonie::surface::point::multipointcoverage&place={place.lower()}",
                                 args=[f"bbox={bbox}"])
    return fcst


def forecast_query(places):
    """Makes pandas DataFrame from forecast data."""
    data = []
    forecasts = {place: fmi_forecast_query(place) for place in places}

    for t in forecasts[places[0]].data.keys():
        timestamp = pd.to_datetime(t)
        row = {
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'hour': timestamp.hour,
            'minute': timestamp.minute
        }
        for place in places:
            row[f"{place} temperature"] = forecasts[place].data[t][place]["Air temperature"]["value"]
        data.append(row)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":

    df = forecast_query(forecast_places)
    print(df)
    fc = fmi_forecast_query("helsinki")

    load_test()

#    print(fc.data.keys())
