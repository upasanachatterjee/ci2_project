from datasets import Dataset, load_dataset
from huggingface_hub import login
import os
from dotenv import load_dotenv

load_dotenv()  # Load variables from .env into the environment

def login_huggingface(token):
    login(token)
    print("Logged in to Hugging Face Hub")

def load_dataset_from_huggingface(dataset_name="dragonslayer631/ci2_allsides", split="train"):
    dataset = load_dataset(dataset_name, split=split)
    print(f"Loaded dataset: {dataset_name}, split: {split}")
    return dataset

def save_dataset_to_huggingface(dataset, dataset_name="dragonslayer631/ci2_allsides", split="train"):
    dataset.push_to_hub(dataset_name, split=split)
    print(f"Saved dataset: {dataset_name}, split: {split}")


def save_dataframe_to_huggingface(df, dataset_name="dragonslayer631/ci2_allsides", split="train"):
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(dataset_name, split=split)
    print(f"Saved DataFrame to Hugging Face: {dataset_name}, split: {split}")