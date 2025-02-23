import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_data(file_path):
    # load the dataset
    df = pd.read_csv(file_path, comment='/')
    df['text'] = df['Overview'].fillna("")
    return df

def build_vectorizer(df):
    #build the vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['text'])
    return vectorizer, tfidf_matrix

def search(query, df, vectorizer, tfidf_matrix, top_n=5):
    #transform query into vector
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df = df.copy()
    df['similarity'] = cosine_similarities
    df_sorted = df.sort_values(by='similarity', ascending=False)
    return df_sorted.head(top_n)[['Series_Title', 'similarity']]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python recommend.py \"Your movie description...\"")
        sys.exit(1)
    query = sys.argv[1]
    dataset_path = "imdb_top_1000.csv"
    
    df = load_data(dataset_path)
    vectorizer, tfidf_matrix = build_vectorizer(df)
    results = search(query, df, vectorizer, tfidf_matrix)
    
    for idx, row in enumerate(results.itertuples(), start=1):
        print(f"{idx}. {row.Series_Title} (similarity: {row.similarity:.3f})")