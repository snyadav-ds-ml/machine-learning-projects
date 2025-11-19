import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")

# -------------------------------
# 1. Load dataset
# -------------------------------
def read_data(file_path):
    return pd.read_csv(file_path) # columns: Metaphor_ID, Label, Text

# -------------------------------
# tokenization and cleaning
# -------------------------------
def tokenize_and_clean(df,join_tokens=False, colname='text'):
    document = nlp.pipe(data['text'], disable=["ner", "parser", "textcat"])
    df.loc[:, 'spacy_text'] = list(document)
    def tokenize(doc, join_tokens=False):
        tokens = [word.lemma_.lower() for word in doc if not word.is_stop and not word.is_punct and not word.like_num or (word.lemma_ == 'not')]
        if join_tokens:
            return " ".join(tokens)
        return tokens
    df.loc[:, 'cleaned_tokens'] = df['spacy_text'].apply(tokenize)
    return df



# -------------------------------
# Extract candidate sentence
# -------------------------------

def get_candidate_sentence(row, candidate):
    nlp_sentence = nlp(row.text)
    for sent in nlp_sentence.sents:
        if candidate in [token for token in sent]:
            return sent.text

# Load data
initial_data = read_data("train-train.csv")
    
# Drop duplicates
data = initial_data.drop_duplicates(keep='first').reset_index(drop=True).copy()

# Tokenize and clean text
tokenized_data = tokenize_and_clean(data)
#print(data.head())

# Map metahor id with candidates
metaphor_candidates = {0: 'road', 1: 'candle', 2: 'light', 3: 'spice', 4: 'ride', 5: 'train', 6: 'boat'}
tokenized_data.loc[:, 'metaphor_candidate'] = data['metaphorID'].map(metaphor_candidates)


tokenized_data.loc[:, 'candidate_sentence'] = tokenized_data.apply(
    lambda row: get_candidate_sentence(row['spacy_text'], row.metaphor_candidate),
    axis=1
)

# save data in .csv
tokenized_data.to_csv("processed_data.csv", index=False)

