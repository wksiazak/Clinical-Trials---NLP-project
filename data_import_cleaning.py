import json
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('punkt_tab')

def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def preprocess_dataframe(data):
    df = pd.json_normalize(data)
    df.columns = df.columns.str.replace(r'protocolSection\.|resultsSection\.|derivedSection\.', '', regex=True)
    return df

def flatten_dataframe(df):
    while True:
        nested_columns = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, (dict, list))).any()]
        if not nested_columns:
            break
        for col in nested_columns:
            expanded_df = pd.json_normalize(df[col].dropna())
            expanded_df.columns = [f"{col}_{subcol}" for subcol in expanded_df.columns]
            df = df.drop(columns=[col]).join(expanded_df, how='left')
    return df

def remove_empty_columns(df, threshold):
    return df.loc[:, df.isna().sum() <= threshold]

def select_columns(df, selected_columns_file):
    selected_columns = pd.read_csv(selected_columns_file)['columns'].tolist()
    return df[selected_columns]

def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return text

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    if isinstance(text, str):
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return ' '.join(filtered_tokens)
    return text

def clean_dataframe(df):
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_text)
        df[col] = df[col].apply(remove_stopwords)
    return df

def save_to_csv(df, output_file):
    df.to_csv(output_file, index=False)

def main():
    download_nltk_resources()
    data = load_json_data('ctg-studies_top_4000.json')
    df = preprocess_dataframe(data)
    df = flatten_dataframe(df)
    df = remove_empty_columns(df, 2000)
    df = select_columns(df, 'selected_columns.csv')
    df = clean_dataframe(df)
    save_to_csv(df, 'cleaned_data.csv')

if __name__ == "__main__":
    main()

