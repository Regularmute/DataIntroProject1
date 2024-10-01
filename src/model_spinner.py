from datetime import datetime
import data_functions as func
import calendar
import statsmodels.api as sm
import pandas as pd
from enum import Enum
import numpy as np
import dataframe_handler


DFH = dataframe_handler.DFHandler


class Strategy(Enum):
    THREE_MONTHS_TAIL = 1
    ONE_MONTH_TAIL = 2


def get_model():
    return sm.OLS()


def spin(start:int, period:int, y:str):
    while start < 13:
        e_month = start+period-1
        DFH.limit(2023, start, 2023, e_month)
        df = DFH.get_range_df()
        print('=====================================================',df['date'][0],' - ',df['date'].iloc[-1])
        func.perform_linear_regression(df.drop(columns='date'), y)
        start+=period


if __name__ == '__main__':
    path = 'data/data_2023.csv'
    DFH = dataframe_handler.DFHandler(path)
    DFH.initialize()
    spin(1, 2, 'CO2')