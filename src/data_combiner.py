from dataframe_handler import DFHandler
import pandas as pd


def combine(dataframes=None, get_handler=False):
    """
    Combine a list of DataFrames by 'date' and 'hour'. If the frequency is finer than one hour (e.g., 15 minutes), 
    the values will be averaged for each hour.

    Parameters:
        dataframes: List of pd.DataFrame-objects 
        get_handler (bool, optional): If True, returns a DFHandler-wrapped object. Default is False.

    Returns:
        pd.DataFrame or DFHandler: The combined DataFrame object. If get_handler is True, returns a DFHandler wrapper.
    """

    if not isinstance(dataframes, list):
        raise ValueError("Please use list of dataframes as a parameter")
    if len(dataframes) == 0:
        raise ValueError("Given list was empty")
    # Clean dataframes
    for i, df in enumerate(dataframes):
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"List should contain only pd.DataFrame type objects. Object in index {i} is not a Pandas DataFrame type")
        if 'timestamp' not in df.columns:
            raise ValueError(f"Column 'timestamp' is required. Not present in dataframe given in index {i}")
        # Drop some unneccessary fields (if existing)
        if 'datasetId' in df.columns:
            df = df.drop(columns=['datasetId'])
        if 'startTime' in df.columns:
            df = df.drop(columns='startTime')
        if 'endTime' in df.columns:
            df = df.drop(columns='endTime')
        if any('Vantaa' in col for col in df.columns):
            df = df.drop(columns='minute')
        elif 'day_of_week' in df.columns:
            df = df.drop(columns='day_of_week')
        # Ensure timestamp to be in correct format, check columns year, month, day, hour
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

        dataframes[i] = df

    dff = pd.DataFrame()

    dow = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2,
        'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

    for df in dataframes:
        # Average hours
        if 'day_of_week' in df.columns and df.day_of_week.dtype == object:
            df['day_of_week'] = df.day_of_week.apply(lambda d: dow[d]).astype(int)
        df = df.groupby(['day', 'hour']).mean().reset_index()
        df['date'] = df['timestamp'].dt.date
        df = df.drop(columns=['timestamp', 'year', 'month', 'day'])
        if dff.empty:
            dff = df
            continue
        # Merge new dataframe
        for col in df.columns:
            if col not in ['date', 'hour'] and col in dff.columns:
                raise ValueError(f"Dublicate column name '{col}' in two or more dataframes")
        dff = pd.merge(dff, df, on=['date', 'hour'], how='outer')
        

    if get_handler:
        return DFHandler(df=dff)

    return dff
