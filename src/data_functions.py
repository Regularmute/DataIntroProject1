import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from patsy import dmatrices
from fmi_config import stations as fmi_stations
from patsy import dmatrices
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from scipy.stats import spearmanr


from data_test_config import configs

alldata2023path = 'data/data_2023.csv'


def read_full_df(path):
    df = pd.read_csv(path)
    return df


def summarize_df(df):
    """Summarize the DataFrame."""
    summary = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'summary_statistics': df.describe(include='all').to_dict()
    }
    return summary


def print_summary(summary):
    """Print the summary in a readable format."""
    print("DataFrame Summary:")
    print(f"Shape: {summary['shape']}")
    print("\nColumns:")
    for column in summary['columns']:
        print(f"  - {column}")
    print("\nData Types:")
    for column, dtype in summary['dtypes'].items():
        print(f"  - {column}: {dtype}")
    print("\nMissing Values:")
    for column, missing in summary['missing_values'].items():
        print(f"  - {column}: {missing}")
    print("\nSummary Statistics:")
    for column, stats in summary['summary_statistics'].items():
        print(f"\n  - {column}:")
        for stat, value in stats.items():
            print(f"    - {stat}: {value}")


def check_data_quality(df):
    """Check for NaN values and values not in the expected format."""
    issues = {}

    for column in df.columns:
        issues[column] = {
            'NaN_count': df[column].isnull().sum(),
            'invalid_format_count': 0
        }

        if df[column].dtype == 'int64' or df[column].dtype == 'float64':
            invalid_format = df[column].apply(
                lambda x: not isinstance(x, (int, float)) and pd.notnull(x))
        elif df[column].dtype == 'object':
            invalid_format = df[column].apply(
                lambda x: not isinstance(x, str) and pd.notnull(x))
        else:
            invalid_format = pd.Series([False] * len(df))

        issues[column]['invalid_format_count'] = invalid_format.sum()

    return issues


def print_data_quality_issues(issues):
    """Print data quality issues in a readable format."""
    print("Data Quality Issues:")
    for column, issue in issues.items():
        print(f"\n  - {column}:")
        print(f"    - NaN count: {issue['NaN_count']}")
        print(f"    - Invalid format count: {issue['invalid_format_count']}")


def plot_pairplot(df):
    """Plot pair plots for all variables in the DataFrame."""
    sns.pairplot(df)
    plt.show()


def perform_linear_regression2(df, dependent_var):
    """Perform linear regression with the specified dependent variable."""
    df = df.dropna(subset=[dependent_var])
    # Define the formula for the linear regression model with interaction terms    df.drop(columns=['electricity_cost'], inplace=True)
    formula = f"{dependent_var} ~ " + \
        " + ".join([f"{col}" for col in df.columns if col != dependent_var])
    formula += " + " + " + ".join([f"{col1}:{col2}" for i, col1 in enumerate(df.columns)
                                  for col2 in df.columns[i+1:] if col1 != dependent_var and col2 != dependent_var])

    # Create the design matrices
    y, X = dmatrices(formula, data=df, return_type='dataframe')

    # Fit the linear regression model
    model = sm.OLS(y, X).fit()
    print(model.summary())


def perform_linear_regression(df, dependent_var, print_=True):
    """Perform linear regression with the specified dependent variable."""
    df = df.dropna(subset=[dependent_var])

    X = df.drop(columns=[dependent_var, 'date', 'hour'])

    X = sm.add_constant(X, has_constant='add')
    y = df[dependent_var]
    if print_:
        print("TRAINING features ==================================")
        print(X.columns.tolist())
        print("=====================================================")
    model = sm.OLS(y, X).fit()
    if print_:
        print(model.summary())
    return model


def clean_df(df,):
    df = df.copy()
    df = df.dropna()
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day'] / 7)

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    stations = fmi_stations
    stations = [station + " temperature" for station in stations]

    for station in stations:
        if station not in df.columns:
            raise ValueError(
                f"Station column '{station}' not found in DataFrame")

    df['temperature'] = df[stations].mean(axis=1)
    df.drop(columns=stations, inplace=True)

    df['export'] = df['production'] - df['consumption']

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['solar_prediction'] = df['solar prediction']
    df.drop(columns=['solar prediction'], inplace=True)

    columns_to_transform = [
        'wind', 'solar_prediction', 'electricity_cost', 'temperature', 'production']

    for column in columns_to_transform:
        mean = df[column].mean()
        sd = df[column].std()
        df[column] = (df[column] - mean) / (2 * sd)

    # for column in columns_to_transform:
    #     min_value = df[column].min()
    #     if min_value <= 0:
    #         df[column] = df[column] - min_value + 1e-6
    #     df[column], _ = stats.boxcox(df[column])

    df.drop(columns=config['drop'], inplace=True)

    return df


def calculate_vif(df):
    """Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame."""
    X = df.drop(columns=['CO2', 'date', 'hour'])
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]

    return vif_data


def cross_validate_linear_regression(df, dependent_var,  print_=True, k=10):
    """Perform k-fold cross-validation for linear regression."""
    df = df.dropna(subset=[dependent_var])

    X = df.drop(columns=[dependent_var, 'date', 'hour'])
    X = sm.add_constant(X)
    y = df[dependent_var]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    mape_scores = []
    r2_scores = []
    y_avgs = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)

        y_avgs.append(y_test.mean())
        mse_scores.append(mean_squared_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    y_avgs_str = [f"{avg:.4f}" for avg in y_avgs]
    mse_scores_str = [f"{score:.4f}" for score in mse_scores]
    rmse_scores_str = [f"{score:.4f}" for score in rmse_scores]
    mape_scores_str = [f"{score:.4f}" for score in mape_scores]
    r2_scores_str = [f"{score:.4f}" for score in r2_scores]
    if print_:
        print(f"Average y: {y_avgs_str}")
        print(f"Avg of avg ys: {np.mean(y_avgs):.4f}")
        print(f"Mean Squared Error (MSE) scores: {mse_scores_str}")
        print(f"Average MSE: {np.mean(mse_scores):.4f}")
        print(f"Root Mean Squared Error (RMSE) scores: {rmse_scores_str}")
        print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
        print(
            f"Mean Absolute Percentage Error (MAPE) scores: {mape_scores_str}")
        print(f"Average MAPE: {np.mean(mape_scores):.4f}")
        print(f"R^2 scores: {r2_scores_str}")
        print(f"Average R^2: {np.mean(r2_scores):.4f}")


def predict_co2_for_day(model, df, date, print_=True):
    """Predict CO2 emissions for one day using the trained model."""
    day_data = df[df['date'] == date].sort_values(by='hour')
    if print_:
        print(f"day_data: {day_data}")
    y_actual = day_data['CO2']
    X_day = day_data.drop(columns=['CO2', 'date', 'hour'])
    X_day = sm.add_constant(X_day)
    X_day = sm.add_constant(X_day, has_constant='add')

    if print_:
        print(f"X_day: {X_day}")

        print(f"peridcting inside predict_co2_for_day function")

        print(f"X_day shape: {X_day.shape}")
        print(f"X_day columns: {X_day.columns}")
        print(f"y_actual shape: {y_actual.shape}")
        print(f"y_actual: {y_actual}")

    model_exog_names = model.model.exog_names
    X_day_columns = X_day.columns.tolist()

    missing_in_X_day = [
        col for col in model_exog_names if col not in X_day_columns]
    extra_in_X_day = [
        col for col in X_day_columns if col not in model_exog_names]

    if missing_in_X_day and print_:
        print(f"Columns missing in X_day: {missing_in_X_day}")
    if extra_in_X_day and print_:
        print(f"Extra columns in X_day: {extra_in_X_day}")
    if print_:
        print(f"Model parameters: {model.params}")
        print(f"Model exog names: {model_exog_names}")

    try:
        y_pred = model.predict(X_day)
    except Exception as e:
        print(f"*** Cannot make prediction *** {e}")
        print(f"Day-columns: {X_day_columns}")
        print(f"Model parameters: {model_exog_names}")

    if print_:
        print("fails before this?")
    return y_actual, y_pred


def compare_predictions(y_actual, y_pred, print_=True):
    """Compare the predicted CO2 emissions to the actual values."""
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    if print_:
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
        print(f"R^2 Score: {r2:.4f}")

    # Assuming y_actual has 24 values for 24 hours
    hours = list(range(len(y_actual)))
    comparison_df = pd.DataFrame({
        'Hour': hours,
        'Actual CO2': y_actual,
        'Predicted CO2': y_pred
    })

    # Calculate the rank for both actual and predicted CO2 emissions
    comparison_df['Actual Rank'] = comparison_df['Actual CO2'].rank()
    comparison_df['Predicted Rank'] = comparison_df['Predicted CO2'].rank()

    # Calculate Spearman's rank correlation coefficient
    spearman_corr, _ = spearmanr(
        comparison_df['Actual Rank'], comparison_df['Predicted Rank'])
    # Print the DataFrame

    comparison_df['spearman'] = spearman_corr

    if print_:
        print("\nComparison of Actual and Predicted CO2 Emissions:")
        print(comparison_df)

    # Print Spearman's rank correlation coefficient
        print(f"\nSpearman's Rank Correlation: {spearman_corr:.4f}")

    results = {"mse": mse, "rmse": rmse, "mape": mape,
               "r2": r2, "spearman": spearman_corr}

    return comparison_df, results


if __name__ == '__main__':
    data2023 = read_full_df(alldata2023path)
    summary2023 = summarize_df(data2023)

    print_summary(summary2023)
    data_quality_issues = check_data_quality(data2023)

    # for config in configs:
    #     print("Config:", config)
    #     df = clean_df(data2023)
    #     vif_data = calculate_vif(df)

    #     print(type(vif_data))
    #     print("Variance Inflation Factor (VIF):")
    #     print(vif_data)
    #     perform_linear_regression(df, 'CO2')
    #     cross_validate_linear_regression(df, 'CO2', k=10)

    for config in configs:
        print("Config:", config)
        df = clean_df(data2023)
        vif_data = calculate_vif(df)

        print(type(vif_data))
        print("Variance Inflation Factor (VIF):")
        print(vif_data)

        # Train the model and get the trained model
        print("\nTraining the linear regression model:")
        model = perform_linear_regression(df, 'CO2')
        print("\nPerforming k-fold cross-validation:")
        cross_validate_linear_regression(df, 'CO2', k=10)

       # Extract unique dates and select a random date
        unique_dates = df['date'].unique()
        random_date = random.choice(unique_dates)

        # Predict CO2 emissions for the selected date
        print(f"\nPredicting CO2 emissions for date {random_date}:")
        y_actual, y_pred = predict_co2_for_day(model, df, random_date)

        print(f"\n comparing predictions for date {random_date}:")
        # Compare the predictions to the actual values
        compare_predictions(y_actual, y_pred)
