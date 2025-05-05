import pandas as pd
import glob

def load_multiple_jsons_to_df(paths: list) -> pd.DataFrame:
    df_list = [
        pd.read_json(file) for file in paths
    ]  # Read each file into a DataFrame
    combined_df = pd.concat(df_list, how="diagonal")  # Concatenate all DataFrames
    return combined_df


def load_directory_of_json_to_dataframe(json_folder_path: str) -> pd.DataFrame:
    json_files = glob.glob(
        f"{json_folder_path}/*.json"
    )  # Get all JSON files in the folder
    df_list = [
        pd.read_json(file) for file in json_files
    ]  # Read each file into a DataFrame
    combined_df = pd.concat(df_list, how="diagonal")  # Concatenate all DataFrames
    return combined_df


def compare_columns_in_df_for_bias(df, base, comparator):
    total_rows = df.height
    count_same_original = (df[base] == df[comparator]).sum()

    # Count occurrences where both columns have the value "center"
    original_center = df.filter(df[base] == "center").height
    count_center = df.filter(
        (df[base] == "center") & (df[comparator] == "center")
    ).height

    # Count occurrences where both columns have the value "left"
    original_left = df.filter(df[base] == "left").height
    count_left = df.filter(
        (df[base] == "left") & (df[comparator] == "left")
    ).height

    # Count occurrences where both columns have the value "right"
    original_right = df.filter(df[base] == "right").height
    count_right = df.filter(
        (df[base] == "right") & (df[comparator] == "right")
    ).height

    return {
        "percent_same_total" : count_same_original/total_rows,
        "percent_same_left": count_left/original_left,
        "percent_same_right": count_right/original_right,
        "percent_same_center": count_center/original_center
    }


def load_multiple_csv_to_df(paths: list, separator="|") -> pd.DataFrame:
    df_list = [
        pd.read_csv(file, sep=separator) for file in paths
    ]  # Read each file into a DataFrame
    combined_df = pd.concat(df_list)  # Concatenate all DataFrames
    return combined_df