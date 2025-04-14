import tarfile
import polars as pl
import glob
import json

def load_json_lists_to_df(paths):
    df_list = [
        pl.read_json(file) for file in paths
    ]  # Read each file into a DataFrame
    combined_df = pl.concat(df_list, how="diagonal")  # Concatenate all DataFrames
    return combined_df

def create_tar(directory_path, output_filename):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(directory_path, arcname=".")
    print(f"Tar archive created: {output_filename}")


def load_json_to_dataframe(json_folder_path):
    json_files = glob.glob(
        f"{json_folder_path}/*.json"
    )  # Get all JSON files in the folder
    df_list = [
        pl.read_json(file) for file in json_files
    ]  # Read each file into a DataFrame
    combined_df = pl.concat(df_list, how="diagonal")  # Concatenate all DataFrames
    return combined_df


def write_back_file(name: str, bias_splits: dict, original_json: dict):
    for k, v in bias_splits.items():
        original_json[k] = v

    with open(name, "w") as f:
        json.dump(original_json, f, indent=4)

def write_json_new(dct: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(dct, f)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_sample_article_ids():
    with open("sample_article_ids.txt", "r") as f:
        lst = f.read().splitlines()
    return lst

def convert_text_bias_to_numerical(bias):
    if bias == "left":
        return 0
    if bias == "right":
        return 2
    if bias == "center":
        return 1
    

def compare_columns_in_df_for_bias(df, base, comparator):
    total_rows = df.height
    count_same_original = (df[base] == df[comparator]).sum()

    # Count occurrences where both columns have the value "center"
    original_center = df.filter(pl.col(base) == "center").height
    count_center = df.filter(
        (pl.col(base) == "center") & (pl.col(comparator) == "center")
    ).height

    # Count occurrences where both columns have the value "left"
    original_left = df.filter(pl.col(base) == "left").height
    count_left = df.filter(
        (pl.col(base) == "left") & (pl.col(comparator) == "left")
    ).height

    # Count occurrences where both columns have the value "right"
    original_right = df.filter(pl.col(base) == "right").height
    count_right = df.filter(
        (pl.col(base) == "right") & (pl.col(comparator) == "right")
    ).height

    return {
        "percent_same_total" : count_same_original/total_rows,
        "percent_same_left": count_left/original_left,
        "percent_same_right": count_right/original_right,
        "percent_same_center": count_center/original_center
    }