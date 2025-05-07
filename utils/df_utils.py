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

def keep_columns_related_to(df: pd.DataFrame, type_to_keep: str): 
    always_drop = [
    'topic', 'tags', 'id',
    'summary_100', 'summary_5', 'summary_50', 'text', 
    'summary_100_entity_sentiments', 'summary_100_topic_to_sentiment',
    'summary_50_entity_sentiments', 'summary_50_topic_to_sentiment',
    'text_entity_sentiments', 'text_topic_to_sentiment', 
    ]
    summary_50_related = [
        'summary_50_encoded',
        'summary_50_topic_0', 'summary_50_sentiment_0', 
        'summary_50_topic_1', 'summary_50_sentiment_1', 
        'summary_50_topic_2', 'summary_50_sentiment_2',
        'summary_50_topic_3', 'summary_50_sentiment_3', 
        'summary_50_topic_4', 'summary_50_sentiment_4', 
    ]

    summary_100_related = [
        'summary_100_encoded',
        'summary_100_topic_0', 'summary_100_sentiment_0', 
        'summary_100_topic_1', 'summary_100_sentiment_1', 
        'summary_100_topic_2', 'summary_100_sentiment_2', 
        'summary_100_topic_3', 'summary_100_sentiment_3', 
        'summary_100_topic_4','summary_100_sentiment_4', 
    ]

    text_related = [
        'text_encoded',
        'text_topic_0', 'text_sentiment_0', 
        'text_topic_1', 'text_sentiment_1', 
        'text_topic_2', 'text_sentiment_2', 
        'text_topic_3', 'text_sentiment_3', 
        'text_topic_4','text_sentiment_4',
    ]
    if type_to_keep == 'text':
        return df.drop(summary_50_related + summary_100_related + always_drop, axis=1)
    if type_to_keep == 'summary_100':
        return df.drop(summary_50_related + text_related + always_drop, axis=1)
    if type_to_keep == 'summary_50':
        return df.drop(text_related + summary_100_related + always_drop, axis=1)
    

def expand_columns(df, topic_columns, sentiment_columns, encoded_text_column) -> pd.DataFrame:
    # Stack topics and sentiments into long format
    topic_long = df[topic_columns].copy()
    sentiment_long = df[sentiment_columns].copy()

    topic_long.columns = range(len(topic_columns))  # avoid duplicate column names
    sentiment_long.columns = range(len(sentiment_columns))

    assert len(topic_long) == len(sentiment_long)

    topic_series = topic_long.stack()
    sentiment_series = sentiment_long.stack()

    # Align manually by index
    combined = pd.DataFrame({
        'topic': topic_series.values,
        'sentiment': sentiment_series.values
    }, index=topic_series.index)

    # One-hot encode
    one_hot = pd.get_dummies(combined['topic'])
    weighted = one_hot.mul(combined['sentiment'], axis=0)

    # Aggregate back to row-level (level 0)
    one_hot_topics = one_hot.groupby(level=0).max().add_prefix('topic ')
    one_hot_sentiments = weighted.groupby(level=0).sum().add_prefix('sentiment ')

    # Expanded topic and sentiment
    res = pd.concat([one_hot_topics, one_hot_sentiments], axis=1)
    final = pd.concat([
    df.drop(columns=topic_columns + sentiment_columns).reset_index(drop=True),
    res.reset_index(drop=True)
        ], axis=1)
    

    text_encoded = pd.DataFrame(final[encoded_text_column].tolist())
    final = pd.concat([final.drop(encoded_text_column, axis=1), text_encoded], axis=1)
    return final


def get_df_split(type, train, test):
    joined = pd.concat([train, test], axis=0).reset_index(drop=True)
    print(joined.shape)
    df = keep_columns_related_to(joined, type)
    topic_columns = [f'{type}_topic_0', f'{type}_topic_1', f'{type}_topic_2', f'{type}_topic_3', f'{type}_topic_4']
    sentiment_columns = [f'{type}_sentiment_0', f'{type}_sentiment_1', f'{type}_sentiment_2', f'{type}_sentiment_3', f'{type}_sentiment_4']
    encoded_text_column = f"{type}_encoded"
    df = expand_columns(df, topic_columns, sentiment_columns, encoded_text_column)

    # split back into test and train

    train_idx = train.index
    test_idx = test.index

    X_train = df.loc[train_idx]
    Y_train = X_train[['int_bias']]
    X_train.drop(["int_bias"], axis=1, inplace=True)

    X_test = df.loc[test_idx]
    Y_test = X_test[['int_bias']]
    X_test.drop(["int_bias"], axis=1, inplace=True)

    X_train = X_train.convert_dtypes()
    X_test = X_test.convert_dtypes()
    Y_test = Y_test.convert_dtypes()
    Y_train = Y_train.convert_dtypes()

    return (X_train, Y_train, X_test, Y_test)