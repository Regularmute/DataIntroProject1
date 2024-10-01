from datetime import datetime
import data_functions as func
import calendar
import statsmodels.api as sm
import pandas as pd
from enum import Enum
import numpy as np

dropped = [
    # 'CO2',
    # 'date',
    # 'hour',
    # 'consumption',
    # 'production',
    # 'solar prediction',
    # 'wind',
    'hydro',
    'district',
    # 'electricity_cost',
    'year',
    'month',
    'day',
    # 'day_of_week',
    'Vantaa Helsinki-Vantaan lentoasema temperature',
    'Vaasa lentoasema temperature',
    'Liperi Joensuu lentoasema temperature',
    'Jyväskylä lentoasema temperature',
    'Rovaniemi lentoasema AWOS temperature',
    # 'temp'
]

norm = [
    # 'CO2',
    # 'date',
    # 'hour',
    'consumption',
    'production',
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
    # 'temp'
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
    # 'month',
    # 'day',
    'day_of_week',
]

RENORMALIZE_LIMIT = True    # This switch is for renormalizing variables when new limited DF is spliced from previously normalized
                            # Could be useful when continuous variables tend to have big differences between different time periods


class DFHandler:

    def __init__(self, path:str):
        self.full_df = func.read_full_df(path)
        self.initialized_df = self.full_df.copy()   
        self.range_df = self.full_df.copy()
        self.normalized = False

    def load(self, year:2023):
        pass

    def load_and_initialize(self, year:2023):
        pass

    def reset(self):
        self.initialized_df = self.full_df.copy()

    def get_full_dataframe(self):
        return self.full_df

    def get_initialized_dataframe(self):
        return self.initialized_df
    
    def get_range_df(self):
        return self.range_df

    def set_renormalize_limit(value:bool):
        RENORMALIZE_LIMIT = value
    
    def limit(self, s_year, s_month, e_year, e_month):
        """
        Create a date range and splices initialized df into new one
        """
        DF = self.initialized_df
        if RENORMALIZE_LIMIT and self.normalized:
            for column in norm:
                DF[column] = self.full_df[column]

        start = datetime(s_year, s_month, 1).date()
        end = datetime(
            e_year, 
            min(e_month, 12),
            calendar.monthrange(e_year, min(e_month, 12))[1]
        ).date()
        self.range_df = DF[(DF['date'] >= start) & (DF['date'] <= end)].reset_index(drop=True)
        
        if RENORMALIZE_LIMIT and self.normalized:
            self.normalize(self.range_df)
    
    
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
        if 'temp' in DF.columns:
            DF['temp'] = DF.iloc[:,-5:].mean(axis=1)
            self.initialized_df = DF


    def drop_columns(self):
        """
        Drop all columns defined in list 'dropped'
        """
        DF = self.initialized_df
        drop_columns = [column for column in DF.columns if column in dropped]
        self.initialized_df = DF.drop(columns=drop_columns)
        print('Dropped columns:',drop_columns,'\n=============================')


    def to_periodic(self):
        """
        Make new column with periodic function values for categorical data in columns of list 'periodic'
        """
        DF = self.initialized_df
        for column in periodic:
            DF[column+'_sin'] = np.sin(2 * np.pi * DF[column] / (max(DF[column])-1))
            DF[column+'_cos'] = np.cos(2 * np.pi * DF[column] / (max(DF[column])-1))
        DF = DF.drop(columns=periodic)
        self.initialized_df = DF
        print('To periodic (sin, cos):',periodic,'\n=============================')


    def normalize(self, df=None):
        """
        Normalize columns in global list 'norm'
        """
        if df is None:
            DF = self.initialized_df
        else:
            DF = df
        for column in norm:
            mean = DF[column].mean()
            sd = DF[column].std()
            DF[column] = (DF[column] - mean) / (2 * sd)
        if df is None:
            self.initialized_df = DF
            print('Normalized columns:',norm,'\n=============================')
        else:
            df = DF
            if RENORMALIZE_LIMIT:
                print('Re-Normalized columns:',norm,'\n=============================')
        self.normalized = True
        


    def initialize(self):
        """
        Run all above
        """
        self.avg_temperatures()                      # Average temperatures
        self.create_dates()                          # Turn date-column to date-objects
        self.to_periodic()                           # Switch catecorigal variables to period ones
        self.normalize()
        self.drop_columns()                          # Drop non required columns


