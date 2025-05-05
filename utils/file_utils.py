import tarfile
import json

def create_tar(directory_path: str, output_filename: str) -> None:
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(directory_path, arcname=".")
    print(f"Tar archive created: {output_filename}")

def update_file(name: str, bias_splits: dict, original_json: dict):
    for k, v in bias_splits.items():
        original_json[k] = v

    with open(name, "w") as f:
        json.dump(original_json, f, indent=4)


def write_json_file(dct: dict, file_path: str):
    with open(file_path, "w") as f:
        json.dump(dct, f)


def load_json_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def get_sample_article_ids():
    with open("sample_article_ids.txt", "r") as f:
        lst = f.read().splitlines()
    return lst