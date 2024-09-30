"""Module for requesting data from FMI API."""

import datetime as dt
import pandas as pd

from fmi_config import bbox, stations, timestep, annual_data, annual_data

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
            'timestamp': timestamp,
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'hour': timestamp.hour,
            'minute': timestamp.minute,
            'day_of_week': timestamp.day_name()
        }
        for station in stations:
            station_col = f"{station} temperature"
            row[station_col] = observations.data[t][station]["Air temperature"]["value"]
        data.append(row)

    df = pd.DataFrame(data)
    return df


def split_time_intervals(start_time, end_time, interval_hours=72):
    """Split time interval into smaller intervals.
    FMI API limits to 168 hours, in practice needs to be shorter due to server load."""
    intervals = []
    current_start = start_time

    while current_start < end_time:
        current_end = min(
            current_start + dt.timedelta(hours=interval_hours), end_time)
        intervals.append((current_start, current_end))
        current_start = current_end + dt.timedelta(hours=1)
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
    end_time = dt.datetime(2023, 1, 9, 23, 0, 0)

    temperature_history = collect_yearly_data(
        start_time, end_time, timestep, stations)
    save_dataframe_to_csv(temperature_history, annual_data)
    loaded_data = load_dataframe_from_csv(annual_data)
    print(loaded_data)


def load_pieces(fmi_pieces):
    """Load historical weather data as monthly pieces"""
    for start, end, filename in fmi_pieces:

        data = pd.DataFrame()
        for start_time, end_time, filename in fmi_pieces:
            temperature_history = collect_yearly_data(
                start_time, end_time, timestep, stations)
            save_dataframe_to_csv(temperature_history, filename)
    # return data


def generate_fmi_pieces(start_year=2024, start_month=3, end_year=2024, end_month=9, start_counter=27):
    fmi_pieces = []
    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
              'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

    def get_last_day_of_month(year, month):
        if month == 12:
            return dt.datetime(year, 12, 31, 23, 0, 0)
        next_month = dt.datetime(year, month + 1, 1, 0, 0, 0)
        last_day = next_month - dt.timedelta(days=1)
        return dt.datetime(last_day.year, last_day.month, last_day.day, 23, 0, 0)

    counter = start_counter
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == start_year and month < start_month:
                continue
            if year == end_year and month > end_month:
                break
            start_date = dt.datetime(year, month, 1, 0, 0, 0)
            end_date = get_last_day_of_month(year, month)
            file_name = f'data/fmi_pieces/{counter:02d}_{months[month-1]}_{year}.csv'
            fmi_pieces.append((start_date, end_date, file_name))
            counter += 1

    return fmi_pieces


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
            'timestamp': timestamp,
            'year': timestamp.year,
            'month': timestamp.month,
            'day': timestamp.day,
            'hour': timestamp.hour,
            'minute': timestamp.minute,
            'day_of_week': timestamp.day_name()
        }
        for place in places:
            row[f"{place} temperature"] = forecasts[place].data[t][place]["Air temperature"]["value"]
        data.append(row)

    df = pd.DataFrame(data)
    return df


if __name__ == "__main__":
    time_splits = generate_fmi_pieces(
        start_year=2022, start_month=6, end_year=2024, end_month=9, start_counter=6)
    # print(time_splits)

    df = load_pieces(time_splits)

    # df = forecast_query(forecast_places)
    # print(df)
    # fc = fmi_forecast_query("helsinki")

    # load_test()

#    print(fc.data.keys())
