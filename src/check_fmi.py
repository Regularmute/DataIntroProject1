import pandas as pd


def list_and_remove_duplicate_rows(file_path, output_file_path):
    df = pd.read_csv(file_path)

    duplicates = df[df.duplicated(
        subset=['year', 'month', 'day', 'hour'], keep=False)]

    if not duplicates.empty:
        print("Duplicate rows based on 'year', 'month', 'day', and 'hour':")
        print(duplicates)
        df_cleaned = df.drop_duplicates(
            subset=['year', 'month', 'day', 'hour'], keep='first')

        df_cleaned.to_csv(output_file_path, index=False)
        print(f"Cleaned data saved to {output_file_path}")
    else:
        print("No duplicate rows found.")


if __name__ == "__main__":
    file_path = "data/annual_weather_data_2023.csv"
    output_file_path = "data/cleaned_annual_weather_data_2023.csv"
    list_and_remove_duplicate_rows(file_path, output_file_path)
