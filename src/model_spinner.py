from datetime import datetime
import data_functions as func
import calendar
import statsmodels.api as sm
import pandas as pd
from enum import Enum
import numpy as np
import dataframe_handler
import random


DFH = dataframe_handler.DFHandler


class Strategy(Enum):
    THREE_MONTHS_TAIL = 1
    ONE_MONTH_TAIL = 2


def spin(start:int, period:int, y:str):
    """
    Spilits yearly data into time periods of given size.
    """
    while start < 13:
        e_month = start+period-1
        DFH.limit(2023, start, 2023, e_month)
        df = DFH.get_range_df()
        # df = DFH.get_initialized_dataframe()
        print('=====================================================',df['date'][0],' - ',df['date'].iloc[-1])
        # func.perform_linear_regression(df.drop(columns='date'), y)
        func.cross_validate_linear_regression(df.drop(columns='date'), y, 10)
        # func.perform_linear_regression(df, y)
        vif = func.calculate_vif(df.drop(columns='date'))
        print(vif)
        start+=period





    pass

def validate_prediction():
    pass


if __name__ == '__main__':
    path = 'data/data_2023.csv'
    DFH = dataframe_handler.DFHandler(path)
    DFH.initialize()
    DFH.set_renormalize_limit(False)
    print(DFH.get_random_day(2023))
    spin(1, 12, 'CO2')