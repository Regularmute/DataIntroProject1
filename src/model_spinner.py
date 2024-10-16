from datetime import datetime, timedelta
import data_functions as func
import calendar
import statsmodels.api as sm
import pandas as pd
from enum import Enum
import numpy as np
import dataframe_handler
import random
import matplotlib.pyplot as plt


DFH = dataframe_handler.DFHandler


class Strategy(Enum):
    """
    Model validation strategyclasses
    """
    ONE_MONTH_RANDOM = 1                # Train model with one month, predict random days over the same year
    THREE_MONTHS_RANDOM = 2             # Same as above with 3 month training
    # Try to predict next n days given one month training
    ONE_MONTH_RANDOM_TAIL = 3
    THREE_MONTHS_RANDOM_TAIL = 4        # Same as above with 3 month training
    # Use whole year for training and predict random days
    ONE_YEAR_FULL = 5
    # Train model with one month, predict every day of the year
    ONE_MONTH_FULL = 6
    THREE_MONTHS_FULL = 7               # Same as above with three months


class PlotType(Enum):
    """
    Type of plotted data
    """
    METRICS = 1                         # Plot metrics
    PREDICTION = 2                      # Plot predictions vs true values


# def spin(start:int, period:int, y:str):
#     """
#     Splits yearly data into time periods of given size.
#     """
#     while start < 13:
#         e_month = start+period-1
#         df = DFH.limit(2023, start, 2023, e_month).get_range_df()
#         print('=====================================================',df['date'][0],' - ',df['date'].iloc[-1])
#         # func.perform_linear_regression(df.drop(columns='date'), y)
#         func.cross_validate_linear_regression(df.drop(columns='date'), y, 10)
#         # func.perform_linear_regression(df, y)
#         vif = func.calculate_vif(df.drop(columns='date'))
#         print(vif)
#         start+=period


def run_strategy(strat: Strategy, plot_type: PlotType, days=0, print_=True):
    """
    Run given strategy of model validation.
    """
    if not isinstance(strat, Strategy):
        raise ValueError(
            f"Invalid strategy type: {type(strat)}. Expected a Strategy enum.")

    if strat == Strategy.ONE_MONTH_RANDOM:
        for month in range(1, 2):
            data = DFH.limit(2023, month, 2023, month).get_range_df()
            # get model here
            model = func.perform_linear_regression(data, 'CO2', print_=False)
            for i in range(0, days):
                day = DFH.get_day()
                # Get prediction here
                actual, pred = func.predict_co2_for_day(
                    model, day, day.date[0], print_=False)
                comparison = func.compare_predictions(actual, pred, False)
                print("Month", month, " trained model ----> predict:",
                      day.date[0], "===============================================")
                print(pd.DataFrame([comparison[1]]))
                print(comparison[0])
                # func.compare_predictions(day.CO2, pred)
                # Do something with the prediction

    elif strat == Strategy.THREE_MONTHS_RANDOM_TAIL:
        result_df = pd.DataFrame()
        for i in range(0, days):
            while True:
                day = DFH.get_day()
                end = day.date[0] - timedelta(days=1)
                start = end - timedelta(days=90)
                if start.year == 2023:
                    break
            data = DFH.limit(start=start, end=end).get_range_df()
            model = func.perform_linear_regression(data, 'CO2', print_=False)
            actual, pred = func.predict_co2_for_day(
                model, day, day.date[0], print_=False)
            comparison = func.compare_predictions(actual, pred, print_=False)
            new_row = pd.DataFrame([comparison[1]])
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        if print_:
            print(
                "====================================================================================")
            print(
                f"Predicted {days} random days based on model trained with data from 90 previous days")
            print("AVG results:")
            print(result_df.mean())

    elif strat == Strategy.ONE_MONTH_RANDOM_TAIL:
        result_df = pd.DataFrame()
        for i in range(0, days):
            while True:
                day = DFH.get_day()
                end = day.date[0] - timedelta(days=1)
                start = end - timedelta(days=30)
                if start.year == 2023:
                    break
            data = DFH.limit(start=start, end=end).get_range_df()
            model = func.perform_linear_regression(data, 'CO2', print_=False)
            actual, pred = func.predict_co2_for_day(
                model, day, day.date[0], print_=False)
            comparison = func.compare_predictions(
                actual, pred, print_=False)
            new_row = pd.DataFrame([comparison[1]])
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        if print_:
            print(
                "====================================================================================")
            print(
                f"Predicted {days} random days based on model trained with data from 30 previous days")
            print("AVG results:")
            print(result_df.mean())

    elif strat == Strategy.ONE_YEAR_FULL:
        first_day = datetime(2023, 1, 1)
        if plot_type == PlotType.METRICS:
            # Four figures for each model
            fig, axs = plt.subplots(1, 5, figsize=(20, 20))
        else:
            # One figure for each model
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        data = DFH.get_initialized_dataframe()
        df = pd.DataFrame()
        # get model here
        model = func.perform_linear_regression(data, 'CO2', print_=False)
        for n in range(0, 365):
            get_day = (first_day+timedelta(days=n)).date()
            day = DFH.get_day(get_day)
            # Get prediction here
            actual, pred = func.predict_co2_for_day(
                model, day, day.date[0], print_=False)
            comparison = func.compare_predictions(actual, pred, print_=False)

            if plot_type == PlotType.METRICS:
                new_row = pd.DataFrame([comparison[1]])
            else:
                new_row = pd.DataFrame(
                    [{'Actual': actual.mean(), 'Predicted': pred.mean()}])

            df = pd.concat([df, new_row], ignore_index=True)

        print(df.shape)
        print(df.columns)
        print(f"max Spearman: {df['spearman'].max()}")
        print(f"min Spearman: {df['spearman'].min()}")
        print(f"avg Spearman: {df['spearman'].mean()}")

        # Count the number of days in each Spearman correlation range
        below_zero = df[df['spearman'] < 0].shape[0]
        between_zero_and_half = df[(df['spearman'] >= 0) & (
            df['spearman'] <= 0.5)].shape[0]
        above_half = df[df['spearman'] > 0.5].shape[0]

        # Print the counts
        print(
            f"Number of days with Spearman correlation below 0: {below_zero}")
        print(
            f"Number of days with Spearman correlation between 0 and 0.5: {between_zero_and_half}")
        print(
            f"Number of days with Spearman correlation above 0.5: {above_half}")

        if plot_type == PlotType.METRICS:
            plot_metrics(df, None, axs, [0, 365])
        else:
            plot_predictions(df, None, axs, [0, 365])

        plt.show()

    elif strat == Strategy.ONE_MONTH_FULL:
        first_day = datetime(2023, 1, 1)
        if plot_type == PlotType.METRICS:
            # Four figures for each model
            fig, axs = plt.subplots(12, 5, figsize=(20, 20))
        else:
            # One figure for each model
            fig, axs = plt.subplots(2, 6, figsize=(20, 20))

        for month in range(1, 13):
            data = DFH.limit(2023, month, 2023, month).get_range_df()
            # get model here
            model = func.perform_linear_regression(data, 'CO2', print_=False)
            df = pd.DataFrame()
            red_line_limit = [500, 0]
            for n in range(0, 365):
                get_day = (first_day+timedelta(days=n)).date()
                if get_day.month == month:
                    red_line_limit[0] = min(red_line_limit[0], n)
                    red_line_limit[1] = max(red_line_limit[1], n)
                day = DFH.get_day(get_day)
                # Get prediction here
                actual, pred = func.predict_co2_for_day(
                    model, day, day.date[0], print_=False)
                comparison = func.compare_predictions(
                    actual, pred, print_=False)

                if plot_type == PlotType.METRICS:
                    new_row = pd.DataFrame([comparison[1]])
                else:
                    new_row = pd.DataFrame(
                        [{'Actual': actual.mean(), 'Predicted': pred.mean()}])

                df = pd.concat([df, new_row], ignore_index=True)

            print(df.shape)
            print(df.columns)
            print(f"max Spearman: {df['spearman'].max()}")
            print(f"min Spearman: {df['spearman'].min()}")
            print(f"avg Spearman: {df['spearman'].mean()}")

            # Count the number of days in each Spearman correlation range
            below_zero = df[df['spearman'] < 0].shape[0]
            between_zero_and_half = df[(df['spearman'] >= 0) & (
                df['spearman'] <= 0.5)].shape[0]
            above_half = df[df['spearman'] > 0.5].shape[0]

            # Print the counts
            print(
                f"Number of days with Spearman correlation below 0: {below_zero}")
            print(
                f"Number of days with Spearman correlation between 0 and 0.5: {between_zero_and_half}")
            print(
                f"Number of days with Spearman correlation above 0.5: {above_half}")

            if plot_type == PlotType.METRICS:
                plot_metrics(df, month-1, axs, red_line_limit)
            else:
                plot_predictions(df, month-1, axs, red_line_limit)

        plt.show()

    elif strat == Strategy.THREE_MONTHS_FULL:
        first_day = datetime(2023, 1, 1)
        if plot_type == PlotType.METRICS:
            # Four figures for each model
            fig, axs = plt.subplots(10, 4, figsize=(20, 20))
        else:
            # One figure for each model
            fig, axs = plt.subplots(2, 6, figsize=(80, 20))

        for month in range(1, 11):
            data = DFH.limit(2023, month, 2023, month+2).get_range_df()
            # get model here
            model = func.perform_linear_regression(data, 'CO2', print_=True)
            df = pd.DataFrame()
            red_line_limit = [500, 0]
            for n in range(0, 365):
                get_day = (first_day+timedelta(days=n)).date()
                if get_day.month >= month and get_day.month <= month+2:
                    red_line_limit[0] = min(red_line_limit[0], n)
                    red_line_limit[1] = max(red_line_limit[1], n)
                day = DFH.get_day(get_day)
                # Get prediction here
                actual, pred = func.predict_co2_for_day(
                    model, day, day.date[0], print_=False)
                comparison = func.compare_predictions(
                    actual, pred, print_=False)

                if plot_type == PlotType.METRICS:
                    new_row = pd.DataFrame([comparison[1]])
                else:
                    new_row = pd.DataFrame(
                        [{'Actual': actual.mean(), 'Predicted': pred.mean()}])

                df = pd.concat([df, new_row], ignore_index=True)

            if plot_type == PlotType.METRICS:
                plot_metrics(df, month-1, axs, red_line_limit)
            else:
                plot_predictions(df, month-1, axs, red_line_limit)

        plt.show()


def plot_metrics(df, month, axs, red_line_limit):
    if month is None:
        axs[0].plot(df.index, df["mse"])
        axs[0].plot(red_line_limit, [0, 0], color='red', lw=2)
        axs[0].set_title(f"Year average MSE: {df['mse'].mean()}")
        axs[0].set_xticks([])

        axs[1].plot(df.index, df["rmse"])
        axs[1].plot(red_line_limit, [0, 0], color='red', lw=2)
        axs[1].set_title(f"Year average RMSE: {df['rmse'].mean()}")
        axs[1].set_xticks([])

        axs[2].plot(df.index, df["mape"])
        axs[2].plot(red_line_limit, [0, 0], color='red', lw=2)
        axs[2].set_title(f"Year average MAPE: {df['mape'].mean()}")
        axs[2].set_xticks([])

        axs[3].plot(df.index, df["r2"])
        axs[3].plot(red_line_limit, [0, 0], color='red', lw=2)
        axs[3].set_title(f"Year average R2: {df['r2'].mean()}")
        axs[3].set_xticks([])

        axs[4].plot(df.index, df["spearman"])
        axs[4].plot(red_line_limit, [0, 0], color='red', lw=2)
        axs[4].set_title(f"Year average Spearman: {df['spearman'].mean()}")
        axs[4].set_xticks([])
        return

    axs[month, 0].plot(df.index, df["mse"])
    axs[month, 0].plot(red_line_limit, [0, 0], color='red', lw=2)
    axs[month, 0].set_title(f"Month {month+1} average MSE: {df['mse'].mean()}")
    axs[month, 0].set_xticks([])

    axs[month, 1].plot(df.index, df["rmse"])
    axs[month, 1].plot(red_line_limit, [0, 0], color='red', lw=2)
    axs[month, 1].set_title(
        f"Month {month+1} average RMSE: {df['rmse'].mean()}")
    axs[month, 1].set_xticks([])

    axs[month, 2].plot(df.index, df["mape"])
    axs[month, 2].plot(red_line_limit, [0, 0], color='red', lw=2)
    axs[month, 2].set_title(
        f"Month {month+1} average MAPE: {df['mape'].mean()}")
    axs[month, 2].set_xticks([])

    axs[month, 3].plot(df.index, df["r2"])
    axs[month, 3].plot(red_line_limit, [0, 0], color='red', lw=2)
    axs[month, 3].set_title(f"Month {month+1} average R2: {df['r2'].mean()}")
    axs[month, 3].set_xticks([])

    axs[month, 4].plot(df.index, df["spearman"])
    axs[month, 4].plot(red_line_limit, [0, 0], color='red', lw=2)
    axs[month, 4].set_title(
        f"Month {month+1} average spearman: {df['spearman'].mean()}")
    axs[month, 4].set_xticks([])


def plot_predictions(df, month, axs, red_line_limit):
    if month is None:
        axs.plot(df.index, df["Actual"], color='blue')
        axs.plot(df.index, df["Predicted"], color='orange')
        axs.plot(red_line_limit, [0, 0], color='red', lw=2)
        axs.set_title(f'Full year prediction (orange)')
        return

    axs[int(month/6), int(month % 6)].plot(df.index,
                                           df["Actual"], color='blue')
    axs[int(month/6), int(month % 6)].plot(df.index,
                                           df["Predicted"], color='orange')
    axs[int(month/6), int(month % 6)].plot(red_line_limit,
                                           [0, 0], color='red', lw=2)
    axs[int(month/6), int(month % 6)
        ].set_title(f'{month+1} Prediction (orange)')


if __name__ == '__main__':
    path = 'data/data_2023.csv'
    DFH = dataframe_handler.DFHandler(path)
    DFH.initialize()
    # Don't use renormalizing on time limited dataframe
    DFH.set_renormalize_limit(False)
    # run_strategy(Strategy.ONE_MONTH_FULL, PlotType.METRICS, 50)
    run_strategy(Strategy.ONE_YEAR_FULL, PlotType.METRICS, 50)
    # run_strategy(Strategy.ONE_MONTH_FULL, PlotType.PREDICTION, 50)
