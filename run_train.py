import pandas as pd
import numpy as np
import spacy

nlp = spacy.load("en_core_web_sm")
# -------------------------------
# 1. Load dataset
# -------------------------------
def read_data(file_path):
    return pd.read_csv(file_path) # columns: Metaphor_ID, Label, Text

def tokenize(doc, join_tokens=False):
    nlp_sentence = nlp(doc)
    tokens = [word.lemma_.lower() for word in nlp_sentence if not word.is_stop and not word.is_punct and not word.like_num or (word.lemma_ == 'not')]
    if join_tokens:
        return " ".join(tokens)
    return tokens

if __name__ == "__main__":
    # Load data
    data = read_data("processed_data.csv")
    
    print(data['candidate_sentence'].head())

    # Tokenize candidate sentences
    data['tokenized_sentences'] = data['candidate_sentence'].apply(tokenize)

    print(data['tokenized_sentences'].head())