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
# Extract candidate sentence
# -------------------------------

def get_candidate_sentence(row, candidate):
    nlp_sentence = nlp(row)
    for sent in nlp_sentence.sents:
        for token in sent:
            str_token = str(token)
            if candidate in token.lemma_.lower() or candidate in  str_token.lower() :
                return sent.text

def tokenize(doc, join_tokens=False):
    nlp_sentence = nlp(doc)
    tokens = [word.lemma_.lower() for word in nlp_sentence if not word.is_stop and not word.is_punct and not word.like_num or (word.lemma_ == 'not')]
    if join_tokens:
        return " ".join(tokens)
    return tokens


# Load data
initial_data = read_data("train-train.csv")
    
# Drop duplicates
data = initial_data.drop_duplicates(keep='first').reset_index(drop=True).copy()


# Map metahor id with candidates
metaphor_candidates = {0: 'road', 1: 'candle', 2: 'light', 3: 'spice', 4: 'ride', 5: 'train', 6: 'boat'}
data.loc[:, 'metaphor_candidate'] = data['metaphorID'].map(metaphor_candidates)

data.loc[:, 'candidate_sentence'] = data.apply(
    lambda row: get_candidate_sentence(row['text'], row.metaphor_candidate),
    axis=1
)


data.loc[:, 'tokenized_candidate_sentence'] = data['candidate_sentence'].apply(lambda row: tokenize(row))

print(data['tokenized_candidate_sentence'].head())

# save data in .csv
data.to_csv("processed_data_candidate_sentences.csv", index=False)