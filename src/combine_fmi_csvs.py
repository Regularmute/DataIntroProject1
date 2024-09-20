import pandas as pd
import os


def merge_csv_files(input_folder, output_file):
    """Merge multiple CSV files into one, keeping column names only once."""
    all_files = sorted([f for f in os.listdir(
        input_folder) if f.endswith('.csv')])
    combined_df = pd.DataFrame()

    for i, file in enumerate(all_files):
        file_path = os.path.join(input_folder, file)
        df = pd.read_csv(file_path)

        if i == 0:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved to {output_file}")


if __name__ == "__main__":
    input_folder = "data/fmi_parts"
    output_file = "data/annual_weather_data_2023.csv"

    merge_csv_files(input_folder, output_file)
