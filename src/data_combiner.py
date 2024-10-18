from dataframe_handler import DFHandler
import pandas as pd


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
            if 'month' not in df.columns:
                df['month'] = df.timestamp.dt.month
            if 'day' not in df.columns:
                df['day'] = df.timestamp.dt.day
            if 'hour' not in df.columns:    
                df['hour'] = df.timestamp.dt.hour

        dff = pd.DataFrame()

        dow = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
            'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
        
        for df in dataframes:
            # Average hours
            df = df.groupby(df.hour).mean().reset_index(drop=True)
            if dff.empty:
                dff = df
                continue
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