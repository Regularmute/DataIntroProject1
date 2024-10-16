from datetime import datetime
import data_functions as func
import calendar
import statsmodels.api as sm
import pandas as pd
import numpy as np
import random

dropped = [
    # 'CO2',
    # 'date',
    # 'hour',
    'consumption',
    'production',
    # 'solar prediction',
    # 'wind',
    'hydro',
    'district',
    # 'electricity_cost',
    'year',
    'month',
    'day',
    'day_of_week',
    'Vantaa Helsinki-Vantaan lentoasema temperature',
    'Vaasa lentoasema temperature',
    'Liperi Joensuu lentoasema temperature',
    'Jyväskylä lentoasema temperature',
    'Rovaniemi lentoasema AWOS temperature',
    'hour_cos',
    # 'hour_sin',
    'month_sin',
    # 'month_cos',
    # 'temp'
]

norm = [
    # 'CO2',
    # 'date',
    # 'hour',
    # 'consumption',
    # 'production',
    'solar prediction',
    'wind',
    # 'hydro',
    # 'district',
    'electricity_cost',
    # 'year',
    # 'month',
    # 'day',
    # 'day_of_week',
    # 'Vantaa Helsinki-Vantaan lentoasema temperature',
    # 'Vaasa lentoasema temperature',
    # 'Liperi Joensuu lentoasema temperature',
    # 'Jyväskylä lentoasema temperature',
    # 'Rovaniemi lentoasema AWOS temperature',
    'temp'
]

periodic = [
    # 'CO2',
    # 'date',
    'hour',
    # 'consumption',
    # 'production',
    # 'solar prediction',
    # 'wind',
    # 'hydro',
    # 'district',
    # 'electricity_cost',
    # 'year',
    'month',
    # 'day',
    'day_of_week',
]


class DFHandler:

    def __init__(self, path: str):

        # This switch is for renormalizing variables when new limited DF is spliced from previously normalized
        # Could be useful when continuous variables tend to have big differences between different time periods
        self.renormalize_limit = True

        self.full_df = func.read_full_df(path)
        self.initialized_df = self.full_df.copy()
        self.range_df = self.full_df.copy()
        self.normalized = False

    def load(self, year: 2023):
        pass

    def load_and_initialize(self, year: 2023):
        pass

    def reset(self):
        self.initialized_df = self.full_df.copy()

    def get_full_dataframe(self):
        return self.full_df

    def get_initialized_dataframe(self):
        return self.initialized_df

    def get_range_df(self):
        return self.range_df

    def set_renormalize_limit(self, value: bool):
        self.renormalize_limit = value

    def get_day(self, date=None):
        """
        Get all 24 hour data of a day in given year (default is 2023)
        """
        if not date:
            month = random.randint(1, 12)
            _, days_in_month = calendar.monthrange(2023, month)
            day = random.randint(1, days_in_month)
            date = datetime(2023, month, day).date()
        df = self.initialized_df[self.initialized_df['date'] == date].reset_index(
            drop=True)
        return self.drop_columns(df)

    def limit(self, s_year=None, s_month=None, e_year=None, e_month=None, s_day=1, e_day=None, start=None, end=None, print_=False):
        """
        Create a date range and splices initialized df into new one
        """
        if self.renormalize_limit and self.normalized:
            self.initialize(normalize=False)
        DF = self.initialized_df
        if start is None:
            start = datetime(s_year, s_month, s_day).date()
        if end is None:
            if not e_day:
                e_day = calendar.monthrange(e_year, min(e_month, 12))[1]
            end = datetime(
                e_year,
                min(e_month, 12),
                e_day
            ).date()
        if print_:
            print("GET DATA BETWEEN", start, "-", end)
        self.range_df = DF[(DF['date'] >= start) & (
            DF['date'] <= end)].reset_index(drop=True)
        if self.renormalize_limit and self.normalized:
            self.range_df = self.normalize(self.range_df)

        return self

    def create_dates(self):
        """
        Create actual date-objects from 'date' column
        """
        DF = self.initialized_df
        if 'date' in DF.columns:
            DF['date'] = pd.to_datetime(DF['date']).dt.date
            self.initialized_df = DF

    def avg_temperatures(self):
        """
        Create column for avg temperature of all temperature observations
        """
        DF = self.initialized_df
        DF['temp'] = DF.iloc[:, -5:].mean(axis=1)
        self.initialized_df = DF

    def drop_columns(self, DF=None):
        """
        Drop all columns defined in list 'dropped'. If no dataframe was passed, drop columns in class object self.initialized_df
        """
        if DF is not None:
            drop_columns = [
                column for column in DF.columns if column in dropped and column in DF]
            return DF.drop(columns=drop_columns)
        else:
            DF = self.initialized_df
            drop_columns = [
                column for column in DF.columns if column in dropped]
            self.initialized_df = DF.drop(columns=drop_columns)
            print('Dropped columns:', drop_columns,
                  '\n=============================')

    def to_periodic(self):
        """
        Create new column of periodic values for categorical data in columns listed in 'periodic'
        """
        DF = self.initialized_df
        for column in periodic:
            DF[column+'_sin'] = np.sin(2 * np.pi *
                                       DF[column] / (max(DF[column])+1))
            DF[column+'_cos'] = np.cos(2 * np.pi *
                                       DF[column] / (max(DF[column])+1))
        # DF = DF.drop(columns=periodic)
        self.initialized_df = DF
        print('To periodic (sin, cos):', periodic,
              '\n=============================')

    def normalize(self, df=None):
        """
        Normalize columns in listed in 'norm'
        """
        if df is None:
            DF = self.initialized_df
            self.normalized = True
        else:
            DF = df
        for column in norm:
            mean = DF[column].mean()
            sd = DF[column].std()
            DF[column] = (DF[column] - mean) / (2 * sd)
        if df is None:
            self.initialized_df = DF
            print('Normalized columns:', norm,
                  '\n=============================')
        else:
            df = DF
            if self.renormalize_limit:
                print('Re-Normalized columns:', norm,
                      '\n=============================')
            return df

    def initialize(self, avg_temp=True, create_dates=True, to_periodic=True, normalize=True, drop_columns=True):
        """
        Run all above
        """
        self.initialized_df = self.full_df.copy()
        if avg_temp:
            self.avg_temperatures()                      # Average temperatures
        if create_dates:
            self.create_dates()                          # Turn date-column to date-objects
        if to_periodic:
            # Switch catecorigal variables to period ones
            self.to_periodic()
        if normalize:
            self.normalize()
        if drop_columns:
            self.drop_columns()                          # Drop non required columns
