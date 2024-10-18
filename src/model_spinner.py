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

from sklearn.metrics import mean_absolute_percentage_error, r2_score


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
    YEAR_WITH_X_DAYS_PRIOR = 8  # New strategy


class PlotType(Enum):
    """
    Type of plotted data
    """
    METRICS = 1                         # Plot metrics
    PREDICTION = 2                      # Plot predictions vs true values
    NONE = 3                            # No plot


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

def run_strategy(strat: Strategy, plot_type: PlotType, days=0, print_=True, prior_days=30):
    """
    Run given strategy of model validation.
    """
    if not isinstance(strat, Strategy):
        raise ValueError(
            f"Invalid strategy type: {type(strat)}. Expected a Strategy enum.")

    if strat == Strategy.ONE_MONTH_RANDOM:
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        for month in range(1, 2):
            data = DFH.limit(2023, month, 2023, month).get_range_df()
            model = func.perform_linear_regression(data, 'CO2', print_=False)
            for i in range(0, days):
                day = DFH.get_day()
                actual, pred = func.predict_co2_for_day(
                    model, day, day.date[0], print_=False)
                comparison = func.compare_predictions(actual, pred, False)
                new_metrics_row = pd.DataFrame([comparison[1]])
                metrics_df = pd.concat(
                    [metrics_df, new_metrics_row], ignore_index=True)
                new_prediction_row = pd.DataFrame(
                    [{'Actual': actual.mean(), 'Predicted': pred.mean()}])
                predictions_df = pd.concat(
                    [predictions_df, new_prediction_row], ignore_index=True)
                if print_:
                    print("Month", month, " trained model ----> predict:",
                          day.date[0], "===============================================")
                    print(pd.DataFrame([comparison[1]]))
                    print(comparison[0])
        return metrics_df, predictions_df

    elif strat == Strategy.THREE_MONTHS_RANDOM_TAIL:
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
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
            new_metrics_row = pd.DataFrame([comparison[1]])
            metrics_df = pd.concat(
                [metrics_df, new_metrics_row], ignore_index=True)
            new_prediction_row = pd.DataFrame(
                [{'Actual': actual.mean(), 'Predicted': pred.mean()}])
            predictions_df = pd.concat(
                [predictions_df, new_prediction_row], ignore_index=True)
        if print_:
            print(
                "====================================================================================")
            print(
                f"Predicted {days} random days based on model trained with data from 90 previous days")
            print("AVG results:")
            print(metrics_df.mean())
        return metrics_df, predictions_df

    elif strat == Strategy.ONE_MONTH_RANDOM_TAIL:
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
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
            comparison = func.compare_predictions(actual, pred, print_=False)
            new_metrics_row = pd.DataFrame([comparison[1]])
            metrics_df = pd.concat(
                [metrics_df, new_metrics_row], ignore_index=True)
            new_prediction_row = pd.DataFrame(
                [{'Actual': actual.mean(), 'Predicted': pred.mean()}])
            predictions_df = pd.concat(
                [predictions_df, new_prediction_row], ignore_index=True)
        if print_:
            print(
                "====================================================================================")
            print(
                f"Predicted {days} random days based on model trained with data from 30 previous days")
            print("AVG results:")
            print(metrics_df.mean())
        return metrics_df, predictions_df

    elif strat == Strategy.ONE_YEAR_FULL:
        first_day = datetime(2023, 1, 1)
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        if plot_type == PlotType.METRICS:
            fig, axs = plt.subplots(1, 5, figsize=(20, 20))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))
        data = DFH.get_initialized_dataframe()
        model = func.perform_linear_regression(data, 'CO2', print_=False)
        for n in range(0, 365):
            get_day = (first_day + timedelta(days=n)).date()
            day = DFH.get_day(get_day)
            actual, pred = func.predict_co2_for_day(
                model, day, day.date[0], print_=False)
            comparison = func.compare_predictions(actual, pred, print_=False)
            new_metrics_row = pd.DataFrame([comparison[1]])
            metrics_df = pd.concat(
                [metrics_df, new_metrics_row], ignore_index=True)
            new_prediction_row = pd.DataFrame(
                [{'Actual': actual.mean(), 'Predicted': pred.mean()}])
            predictions_df = pd.concat(
                [predictions_df, new_prediction_row], ignore_index=True)

        if plot_type == PlotType.METRICS:
            plot_metrics(metrics_df, None, axs, [0, 365])
        else:
            plot_predictions(predictions_df, None, axs, [0, 365])
        plt.show()
        return metrics_df, predictions_df

    elif strat == Strategy.ONE_MONTH_FULL:
        first_day = datetime(2023, 1, 1)
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()

        if plot_type == PlotType.METRICS:
            fig, axs = plt.subplots(12, 5, figsize=(20, 20))
        else:
            fig, axs = plt.subplots(2, 6, figsize=(20, 20))

        for month in range(1, 13):
            data = DFH.limit(2023, month, 2023, month).get_range_df()
            model = func.perform_linear_regression(data, 'CO2', print_=False)
            month_metrics_df = pd.DataFrame()
            month_predictions_df = pd.DataFrame()
            red_line_limit = [500, 0]
            for n in range(0, 365):
                get_day = (first_day + timedelta(days=n)).date()
                if get_day.month == month:
                    red_line_limit[0] = min(red_line_limit[0], n)
                    red_line_limit[1] = max(red_line_limit[1], n)
                day = DFH.get_day(get_day)
                actual, pred = func.predict_co2_for_day(
                    model, day, day.date[0], print_=False)
                comparison = func.compare_predictions(
                    actual, pred, print_=False)
                new_metrics_row = pd.DataFrame([comparison[1]])
                month_metrics_df = pd.concat(
                    [month_metrics_df, new_metrics_row], ignore_index=True)
                new_prediction_row = pd.DataFrame(
                    [{'Actual': actual.mean(), 'Predicted': pred.mean()}])
                month_predictions_df = pd.concat(
                    [month_predictions_df, new_prediction_row], ignore_index=True)

            metrics_df = pd.concat(
                [metrics_df, month_metrics_df], ignore_index=True)
            predictions_df = pd.concat(
                [predictions_df, month_predictions_df], ignore_index=True)

            if plot_type == PlotType.METRICS:
                plot_metrics(month_metrics_df, month-1, axs, red_line_limit)
            else:
                plot_predictions(month_predictions_df,
                                 month-1, axs, red_line_limit)

        plt.show()
        return metrics_df, predictions_df

    elif strat == Strategy.THREE_MONTHS_FULL:
        first_day = datetime(2023, 1, 1)
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()
        if plot_type == PlotType.METRICS:
            fig, axs = plt.subplots(10, 5, figsize=(20, 20))
        else:
            fig, axs = plt.subplots(2, 6, figsize=(80, 20))

        for month in range(1, 11):
            data = DFH.limit(2023, month, 2023, month+2).get_range_df()
            model = func.perform_linear_regression(data, 'CO2', print_=False)
            df = pd.DataFrame()
            red_line_limit = [500, 0]
            for n in range(0, 365):
                get_day = (first_day + timedelta(days=n)).date()
                if get_day.month >= month and get_day.month <= month+2:
                    red_line_limit[0] = min(red_line_limit[0], n)
                    red_line_limit[1] = max(red_line_limit[1], n)
                day = DFH.get_day(get_day)
                actual, pred = func.predict_co2_for_day(
                    model, day, day.date[0], print_=False)
                comparison = func.compare_predictions(
                    actual, pred, print_=False)
                new_metrics_row = pd.DataFrame([comparison[1]])
                metrics_df = pd.concat(
                    [metrics_df, new_metrics_row], ignore_index=True)
                new_prediction_row = pd.DataFrame(
                    [{'Actual': actual.mean(), 'Predicted': pred.mean()}])
                predictions_df = pd.concat(
                    [predictions_df, new_prediction_row], ignore_index=True)

            if plot_type == PlotType.METRICS:
                plot_metrics(metrics_df, month-1, axs, red_line_limit)
            else:
                plot_predictions(predictions_df, month-1, axs, red_line_limit)
        plt.show()
        return metrics_df, predictions_df

    elif strat == Strategy.YEAR_WITH_X_DAYS_PRIOR:
        first_day = datetime(2023, 1, 1)
        data = DFH.get_initialized_dataframe()
        metrics_df = pd.DataFrame()
        predictions_df = pd.DataFrame()

        if plot_type == PlotType.METRICS:
            fig, axs = plt.subplots(1, 5, figsize=(20, 20))
        else:
            fig, axs = plt.subplots(1, 1, figsize=(20, 20))

        for n in range(prior_days, 365):
            end = first_day + timedelta(days=n)
            start = end - timedelta(days=prior_days+1)
            start = start.date()
            end = (end - timedelta(days=1)).date()

            training_data = DFH.limit(start=start, end=end).get_range_df()
            model = func.perform_linear_regression(
                training_data, 'CO2', print_=False)
            get_day = (first_day + timedelta(days=n)).date()
            day = DFH.get_day(get_day)
            actual, pred = func.predict_co2_for_day(
                model, day, day.date[0], print_=False)
            comparison = func.compare_predictions(actual, pred, print_=False)

            new_metrics_row = pd.DataFrame([comparison[1]])
            metrics_df = pd.concat(
                [metrics_df, new_metrics_row], ignore_index=True)

            new_prediction_row = pd.DataFrame(
                [{'Actual': actual.mean(), 'Predicted': pred.mean()}])
            predictions_df = pd.concat(
                [predictions_df, new_prediction_row], ignore_index=True)

        if plot_type == PlotType.METRICS:
            plot_metrics(metrics_df, None, axs, [0, 365])
            plt.show()
        elif plot_type == PlotType.PREDICTION:
            plot_predictions(predictions_df, None, axs, [0, 365])
            plt.show()

        return metrics_df, predictions_df


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


def evaluate_metrics(df, print_=True):

    results = {}
    results['max_spearman'] = df['spearman'].max()
    results['min_spearman'] = df['spearman'].min()
    results['avg_spearman'] = df['spearman'].mean()

    results['below_zero'] = df[df['spearman'] < 0].shape[0]
    results['between_zero_and_half'] = df[(
        df['spearman'] >= 0) & (df['spearman'] <= 0.5)].shape[0]
    results['above_half'] = df[df['spearman'] > 0.5].shape[0]

    # makes no sense to calculate MAPE or R2 from daily values: only 24 data points and
    # hence there are days with very close to zero CO2 and/or variance leading to unreasoable high MAPE and low R2
    # results['max_mape'] = df['mape'].max()
    # results['min_mape'] = df['mape'].min()
    # results['avg_mape'] = df['mape'].mean()

    # results['max_r2'] = df['r2'].max()
    # results['min_r2'] = df['r2'].min()
    # results['avg_r2'] = df['r2'].mean()

    if print_:
        for key, value in results.items():
            print(f"{key.replace('_', ' ').title()}: {value}")

    return results


def calculate_annual_metrics(predictions_df):
    # R2 and MAPE need to be calculated from annaul values
    print(f'shape of predictions_df: {predictions_df.shape}')
    actual = predictions_df['Actual']
    predicted = predictions_df['Predicted']

    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    return {
        'mape': mape,
        'r2': r2
    }


if __name__ == '__main__':
    path = 'data/data_2023.csv'
    DFH = dataframe_handler.DFHandler(path)
    DFH.initialize()
    # Don't use renormalizing on time limited dataframe
    DFH.set_renormalize_limit(False)

    results_df = pd.DataFrame()

    metrics, predictions = run_strategy(
        Strategy.YEAR_WITH_X_DAYS_PRIOR, PlotType.PREDICTION, 50, prior_days=5)

    evaluate_metrics(metrics)

    annual_metrics = calculate_annual_metrics(predictions)
    print(f"Annual MAPE: {annual_metrics['mape']}")
    print(f"Annual R²: {annual_metrics['r2']}")

    # print("one year full:")
    # metrics, _ = run_strategy(Strategy.ONE_YEAR_FULL, PlotType.NONE, 50)
    # results = evaluate_metrics(metrics)
    # results['strategy'] = 'ONE_YEAR_FULL'
    # results['prior_days'] = 0
    # results_df = pd.concat(
    #     [results_df, pd.DataFrame([results])], ignore_index=True)

    # print("one month full:")
    # metrics, _ = run_strategy(Strategy.ONE_MONTH_FULL, PlotType.NONE, 50)
    # results = evaluate_metrics(metrics)
    # results['strategy'] = 'ONE_MONTH_FULL'
    # results['prior_days'] = 0
    # results_df = pd.concat(
    #     [results_df, pd.DataFrame([results])], ignore_index=True)

    # for prior_days in range(1, 30):
    #     print(f"year with {prior_days} days prior:")
    #     metrics, _ = run_strategy(
    #         Strategy.YEAR_WITH_X_DAYS_PRIOR, PlotType.NONE, 0, True, prior_days)
    #     results = evaluate_metrics(metrics)
    #     results['strategy'] = 'YEAR_WITH_X_DAYS_PRIOR'
    #     results['prior_days'] = prior_days
    #     results_df = pd.concat(
    #         [results_df, pd.DataFrame([results])], ignore_index=True)

    # results_df.to_csv('data/evaluation_results.csv', index=False)
