import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(cleaned_data):
    ids_selected = cleaned_data['identificationModule.nctId']
    docs_selected = cleaned_data.drop(columns=['identificationModule.nctId'])
    docs_selected_combined = docs_selected.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    return ids_selected, docs_selected_combined

def split_data(docs, ids, test_size=0.2, random_state=42):
    return train_test_split(docs, ids, test_size=test_size, random_state=random_state)

def compute_tfidf(train_data, test_data):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train_matrix = tfidf_vectorizer.fit_transform(train_data)
    tfidf_test_matrix = tfidf_vectorizer.transform(test_data)
    return tfidf_vectorizer, tfidf_train_matrix, tfidf_test_matrix

def get_top_words(row, feature_names, top_n=10):
    sorted_indices = row.argsort()[::-1][:top_n]
    return [feature_names[i] for i in sorted_indices]

def generate_top_words_df(tfidf_matrix, ids, tfidf_vectorizer):
    feature_names = tfidf_vectorizer.get_feature_names_out()
    return pd.DataFrame({
    'Top_10_words': [
    get_top_words(row, feature_names) for row in tfidf_matrix.toarray()
    ],
    'nctId': ids.values
    })

def perform_clustering(tfidf_matrix, n_clusters=9, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(tfidf_matrix)
    silhouette = silhouette_score(tfidf_matrix, clusters)
    return kmeans, clusters, silhouette

def reduce_dimensions(tfidf_matrix, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    return pca, pca.fit_transform(tfidf_matrix.toarray())

def save_clusters_with_pca(df, clusters, pca_data, output_file):
    df['Cluster'] = clusters
    df['PCA1'] = pca_data[:, 0]
    df['PCA2'] = pca_data[:, 1]
    df['Combined_Text'] = [" ".join(words) for words in df['Top_10_words']]
    df['Top_10_words_text'] = df['Top_10_words'].apply(lambda x: ", ".join(x))
    df[['nctId', 'Cluster', 'PCA1', 'PCA2', 'Combined_Text', 'Top_10_words_text']].to_csv(output_file, index=False)

def visualize_clusters(train_pca, train_clusters, test_pca, test_clusters):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c=train_clusters, cmap='viridis', s=10)
    plt.title("Klastry na danych treningowych")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.subplot(1, 2, 2)
    plt.scatter(test_pca[:, 0], test_pca[:, 1], c=test_clusters, cmap='viridis', s=10)
    plt.title("Klastry na danych testowych")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")

    plt.tight_layout()
    plt.show()

def main():
    # Load and prepare data
    cleaned_data = load_data("cleaned_data.csv")
    ids_selected, docs_selected_combined = prepare_data(cleaned_data)

    # Split data
    X_train, X_test, ids_train, ids_test = split_data(docs_selected_combined, ids_selected)

    # Compute TF-IDF
    tfidf_vectorizer, tfidf_train_matrix, tfidf_test_matrix = compute_tfidf(X_train, X_test)

    # Generate top words dataframes
    df_train_top_words = generate_top_words_df(tfidf_train_matrix, ids_train, tfidf_vectorizer)
    df_test_top_words = generate_top_words_df(tfidf_test_matrix, ids_test, tfidf_vectorizer)

    # Perform clustering
    kmeans, train_clusters, silhouette_train = perform_clustering(tfidf_train_matrix)
    print(f"Silhouette Score dla danych treningowych: {silhouette_train:.2f}")

    test_clusters = kmeans.predict(tfidf_test_matrix)

    # Reduce dimensions for visualization
    train_pca_model, train_pca = reduce_dimensions(tfidf_train_matrix)
    test_pca = train_pca_model.transform(tfidf_test_matrix.toarray())

    # Save clusters and PCA data
    save_clusters_with_pca(df_train_top_words, train_clusters, train_pca, "train_clusters_with_pca.csv")
    save_clusters_with_pca(df_test_top_words, test_clusters, test_pca, "test_clusters_with_pca.csv")

    # Visualize clusters
    visualize_clusters(train_pca, train_clusters, test_pca, test_clusters)

    print("Dane zostały wyeksportowane do plików 'train_clusters_with_pca.csv' i 'test_clusters_with_pca.csv'.")


if __name__ == "__main__":
    main()