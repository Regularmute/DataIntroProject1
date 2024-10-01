import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
from patsy import dmatrices
from fmi_config import stations as fmi_stations
from patsy import dmatrices
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

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


def perform_linear_regression(df, dependent_var):
    """Perform linear regression with the specified dependent variable."""
    df = df.dropna(subset=[dependent_var])

    X = df.drop(columns=[dependent_var])

    X = sm.add_constant(X)
    y = df[dependent_var]

    model = sm.OLS(y, X).fit()
    print(model.summary())


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
        'CO2', 'wind', 'solar_prediction', 'electricity_cost', 'temperature']
    for column in columns_to_transform:
        min_value = df[column].min()
        if min_value <= 0:
            df[column] = df[column] - min_value + 1e-6
        df[column], _ = stats.boxcox(df[column])

    df.drop(columns=config['drop'], inplace=True)

    return df


def calculate_vif(df):
    """Calculate Variance Inflation Factor (VIF) for each feature in the DataFrame."""
    X = df.drop(columns=['CO2'])
    X = sm.add_constant(X)

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(
        X.values, i) for i in range(X.shape[1])]

    return vif_data


def cross_validate_linear_regression(df, dependent_var, k=10):
    """Perform k-fold cross-validation for linear regression."""
    df = df.dropna(subset=[dependent_var])

    X = df.drop(columns=[dependent_var])
    X = sm.add_constant(X)
    y = df[dependent_var]

    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_scores = []
    rmse_scores = []
    mape_scores = []
    r2_scores = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = sm.OLS(y_train, X_train).fit()
        y_pred = model.predict(X_test)

        mse_scores.append(mean_squared_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        mape_scores.append(mean_absolute_percentage_error(y_test, y_pred))
        r2_scores.append(r2_score(y_test, y_pred))

    mse_scores_str = [f"{score:.4f}" for score in mse_scores]
    rmse_scores_str = [f"{score:.4f}" for score in rmse_scores]
    mape_scores_str = [f"{score:.4f}" for score in mape_scores]
    r2_scores_str = [f"{score:.4f}" for score in r2_scores]

    print(f"Mean Squared Error (MSE) scores: {mse_scores_str}")
    print(f"Average MSE: {np.mean(mse_scores):.4f}")
    print(f"Root Mean Squared Error (RMSE) scores: {rmse_scores_str}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Mean Absolute Percentage Error (MAPE) scores: {mape_scores_str}")
    print(f"Average MAPE: {np.mean(mape_scores):.4f}")
    print(f"R^2 scores: {r2_scores_str}")
    print(f"Average R^2: {np.mean(r2_scores):.4f}")


if __name__ == '__main__':
    data2023 = read_full_df(alldata2023path)
    summary2023 = summarize_df(data2023)

    print_summary(summary2023)
    data_quality_issues = check_data_quality(data2023)

    for config in configs:
        print("Config:", config)
        df = clean_df(data2023)
        vif_data = calculate_vif(df)

        print(type(vif_data))
        print("Variance Inflation Factor (VIF):")
        print(vif_data)
        perform_linear_regression(df, 'CO2')

    print("Performing k-fold cross-validation with the last config:")
    cross_validate_linear_regression(df, 'CO2', k=10)

    # print_data_quality_issues(data_quality_issues)cesces

    # data2023 = clean_df(data2023)

    # summary2023 = summarize_df(data2023)
    # print_summary(summary2023)

    # data_quality_issues = check_data_quality(data2023)
    # print_data_quality_issues(data_quality_issues)

    # vif_data = calculate_vif(data2023)
    # print("Variance Inflation Factor (VIF):")
    # print(vif_data)

    # perform_linear_regression(data2023, 'CO2')

   # perform_linear_regression2(data2023, 'CO2')

    # cov_matrix = data2023.cov()    df.drop(columns=['electricity_cost'], inplace=True)
    # print(cov_matrix)

    # plot_pairplot(data2023)
