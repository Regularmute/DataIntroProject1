from datetime import datetime
import data_functions as func
import calendar
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

decaying = [
    'temp',
    # 'electricity_cost',
]



class DFHandler:

    def __init__(self, path=None, df=None):
        
        # This switch is for renormalizing variables when new limited DF is spliced from previously normalized
        # Could be useful when continuous variables tend to have big differences between different time periods
        self.renormalize_limit = True
        
        if df is not None:
            self.full_df = df
        elif path is not None:
            self.full_df = func.read_full_df(path)
        else:
            raise ValueError("Dataframe handler should be initialized with dataframe or path instance to csv-source")
        
        self.initialized_df = self.full_df.copy()   
        self.range_df = self.full_df.copy()
        self.normalized = False


    def reset(self):
        self.initialized_df = self.full_df.copy()

    def get_full_dataframe(self):
        return self.full_df

    def get_initialized_dataframe(self):
        return self.initialized_df
    
    def get_range_df(self):
        return self.range_df

    def set_renormalize_limit(self, value:bool):
        self.renormalize_limit = value

    def get_day(self, date=None):
        """
        Get all 24 hour data of a day in given year (default is 2023). For validation purposes.
        """
        if not date:
            month = random.randint(1,12)
            _, days_in_month = calendar.monthrange(2023, month)
            day = random.randint(1, days_in_month)
            date = datetime(2023, month, day).date()
        df = self.initialized_df[self.initialized_df['date'] == date].reset_index(drop=True)
        return self.drop_columns(df)
            
    
    def limit(self, s_year=None, s_month=None, e_year=None, e_month=None, s_day=1, e_day=None, start=None, end=None):
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
        self.range_df = DF[(DF['date'] >= start) & (DF['date'] <= end)].reset_index(drop=True)
        if self.renormalize_limit and self.normalized:
            self.range_df = self.normalize(self.range_df)
        if self.decay:
            self.run_decay()        
        return self
    
    
    def create_dates(self, df=None):
        """
        Create actual date-objects from 'date' column
        """
        if df is None:
            DF = self.initialized_df
        else:
            DF = df
        if 'date' in DF.columns:
            DF['date'] = pd.to_datetime(DF['date']).dt.date
            self.initialized_df = DF
        if df is None:
            self.initialized_df = DF
            return self
        return DF


    def avg_temperatures(self, df=None):
        """
        Create column for avg temperature of all temperature observations. Non-parametrical
        call averages temperature columns in existing class dataframe.
        """
        if df is None:
            DF = self.initialized_df
        else:
            DF = df
        DF['temp'] = DF[[col for col in DF.columns if 'temperature' in col]].mean(axis=1)
        if df is None:
            self.initialized_df = DF
            return self
        return DF


    def drop_columns(self, df=None):
        """
        Drop all columns defined in list 'dropped'. If no dataframe was passed, drop columns in class object self.initialized_df
        """
        if df is None:
            DF = self.initialized_df
            drop_columns = [column for column in DF.columns if column in dropped]
            self.initialized_df = DF.drop(columns=drop_columns)
            return self
        DF = df
        drop_columns = [column for column in DF.columns if column in dropped]
        return DF.drop(columns=drop_columns)


    def to_periodic(self, df=None):
        """
        Create new column of periodic values for categorical data in columns listed in 'periodic'
        """
        if df is None:
            DF = self.initialized_df
        else:
            DF = df
        for column in periodic:
            DF[column+'_sin'] = np.sin(2 * np.pi * DF[column] / (max(DF[column])+1))
            DF[column+'_cos'] = np.cos(2 * np.pi * DF[column] / (max(DF[column])+1))
        if df is None:
            self.initialized_df = DF
            return self
        return DF


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
            return self
        return DF


    def run_decay(self, df=None):
        """
        Run decay function to all dataframe columns defined in decaying list.
        """
        if df is None:
            DF = self.range_df
        else:
            DF = df
        for col in DF.columns:
            if col in decaying:
                DF[col] = self.decay(DF[col])
        if df is None:
            return DF
        self.range_df = DF
        return self


    def decay(self, values:pd.Series):
        """
        Decay function to reduce data values in ranged dataframe.Reduction will be
        done so that it's largest on first index and smallest in the last index.
        """
        timeframe = len(values)/24
        values *= (np.e ** (-(values.index/timeframe)))
        return values
        
        
    def initialize(self, df=None, avg_temp=True, create_dates=True, to_periodic=True, normalize=True, drop_columns=True, decay=False):
        """
        Run all above
        """
        if df is None:
            self.initialized_df = self.full_df.copy()
        if avg_temp:
            self.avg_temperatures(df)                      # Average temperatures
        if create_dates:
            self.create_dates(df)                          # Turn date-column to date-objects
        if to_periodic:
            self.to_periodic(df)                           # Switch catecorigal variables to period ones
        if normalize:
            self.normalize(df)                             # Normalize (center and scale)   
        if drop_columns:
            self.drop_columns(df)                          # Drop non required columns


    def combine(self, dataframes=None, init_handler=False):
        """
        Combine list of dataframes by the
        """
        if not isinstance(dataframes, list):
            raise ValueError("Please use list of dataframes as parameter")
        if len(dataframes) == 0:
            raise ValueError("No dataframes to combine")
        # Clean dataframes
        for i, df in enumerate(dataframes):
            if not isinstance(df, pd.DataFrame):
                raise ValueError("List should contain only pd.DataFrame type objects.")
            if 'datasetId' in df.columns:
                df = df.drop(columns=['datasetId'])
            if 'endTime' in df.columns:
                df = df.drop(columns='endTime')
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            if 'year' not in df.columns:
                df['year'] = df.timestamp.dt.year
                df['month'] = df.timestamp.dt.month
                df['day'] = df.timestamp.dt.month
                df['hour'] = df.timestamp.dt.hour

        dff = pd.DataFrame()

        dow = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        
        for df in dataframes:
            # Average hours
            if dff.empty:
                dff = df
                continue
            df = df.groupby(df.hour).mean().reset_index(drop=True)
            # Merge new dataframe
            dff = pd.merge(dff, df, on=['timestamp', 'year', 'month', 'day', 'hour'], how='outer')
            df['date'] = df.timestamp.dt.date
            df['hour'] = df.timestamp.dt.hour
            df = df.drop(columns='timestamp')
            if 'day_of_week' in df.columns and df.day_of_week.dtype == object:
                df['day_of_week'] = df.day_of_week.apply(lambda d: dow[d]).astype(int)
        
        if init_handler:
            return DFHandler(dff)
        
        return dff
