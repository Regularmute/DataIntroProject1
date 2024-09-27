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

    # Define the formula for the linear regression model with interaction terms
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


def clean_df(df):
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

    # columns_to_transform = [
    #     'CO2', 'wind', 'solar prediction', 'electricity_cost', 'temperature']
    # for column in columns_to_transform:
    #     min_value = df[column].min()
    #     if min_value <= 0:
    #         df[column] = df[column] - min_value + 1e-6
    #     df[column], _ = stats.boxcox(df[column])

    # On purpose line-by-line for easier variable selection
    # Comment out the lines you don't want to drop
    df.drop(columns=['date'], inplace=True)
    df.drop(columns=['day'], inplace=True)
    df.drop(columns=['year'], inplace=True)
    # df.drop(columns=['hour'], inplace=True)
    # df.drop(columns=['month'], inplace=True)
    # df.drop(columns=['day_of_week'], inplace=True)

    # df.drop(columns=['electricity_cost'], inplace=True)
    # df.drop(columns=['production'], inplace=True)
    df.drop(columns=['consumption'], inplace=True)
    df.drop(columns=['hydro'], inplace=True)
    df.drop(columns=['district'], inplace=True)
    # df.drop(columns=['wind'], inplace=True)
    # df.drop(columns=['solar_prediction'], inplace=True)
    # df.drop(columns=['hour_sin'], inplace=True)
    # df.drop(columns=['hour_cos'], inplace=True)
    # df.drop(columns=['day_of_week_sin'], inplace=True)
    # df.drop(columns=['day_of_week_cos'], inplace=True)
    # df.drop(columns=['month_sin'], inplace=True)
    # df.drop(columns=['month_cos'], inplace=True)
    # df.drop(columns=['CO2'], inplace=True)
    df.drop(columns=['export'], inplace=True)
    # df.drop(columns=['temperature'], inplace=True)

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


if __name__ == '__main__':
    data2023 = read_full_df(alldata2023path)
    summary2023 = summarize_df(data2023)

    print_summary(summary2023)
    data_quality_issues = check_data_quality(data2023)
    # print_data_quality_issues(data_quality_issues)

    data2023 = clean_df(data2023)
    summary2023 = summarize_df(data2023)
    print_summary(summary2023)

    data_quality_issues = check_data_quality(data2023)
    print_data_quality_issues(data_quality_issues)

    vif_data = calculate_vif(data2023)
    print("Variance Inflation Factor (VIF):")
    print(vif_data)

    perform_linear_regression(data2023, 'CO2')

    perform_linear_regression2(data2023, 'CO2')

    # cov_matrix = data2023.cov()
    # print(cov_matrix)

    # plot_pairplot(data2023)
