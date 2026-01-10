import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns

import spacy
from spacy.cli import download

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# Function to create cluster plots with a legend
def plot_clusters(data, selected_clusters, title):
    filtered_data = data[data['Cluster'].isin(selected_clusters)]
    fig, ax = plt.subplots(figsize=(6, 6))

    # List of unique clusters for iteration
    unique_clusters = sorted(filtered_data['Cluster'].unique())

    # Creating points with labels for the legend
    for cluster in unique_clusters:
        cluster_data = filtered_data[filtered_data['Cluster'] == cluster]
        ax.scatter(
            cluster_data['PCA1'],
            cluster_data['PCA2'],
            label=f"Cluster {cluster}",
            s=10
        )

    ax.set_title(title)
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend(loc="best", title="Clusters")  # Adding a legend
    return fig


# Function to generate a WordCloud based on the selected cluster (without filtering by tag)
def generate_wordcloud(data, selected_clusters):
    # Filtering data based only on the selected clusters
    filtered_data = data[data['Cluster'].isin(selected_clusters)]

    # Merging text for selected clusters
    combined_text = " ".join(filtered_data['Combined_Text'])

    # Generating WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)

    # Displaying WordCloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("WordCloud for selected clusters")
    return fig


# Function to compute the most frequent words in the data
def calculate_top_words(data, top_n=50):
    combined_text = " ".join(data['Combined_Text'])
    word_counts = Counter(combined_text.split())
    return word_counts.most_common(top_n)

# Function to display related NCTIDs, Combined_Text, and Cluster + bar chart
def display_related_data(data, word):
    filtered_data = data[data['Combined_Text'].str.contains(word, na=False, case=False)]

    st.write(f"**Clinical trials for selected tag '{word}':**")
    st.dataframe(filtered_data[['nctId', 'Combined_Text', 'Cluster']])

    # Counting tag occurrences in different clusters
    cluster_counts = filtered_data['Cluster'].value_counts().sort_index()

    # Creating a bar chart
    if not cluster_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis",
                    ax=ax)
        ax.set_title(f"Occurrences of '{word}' in different clusters")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")

        # Displaying the chart below the table
        st.pyplot(fig)


def main():
    st.title("Cluster Analysis prepared based on TF-IDF and PCA analysis ")

    st.markdown("##### The visualization below presents the results of text data analysis using the TF-IDF method. "
                "Data comes from https://clinicaltrials.gov/ - here, we use 4000 examples of different clinical trials.")

    # Loading data
    train_data = pd.read_csv("working_files/train_clusters_with_pca.csv")
    test_data = pd.read_csv("working_files/test_clusters_with_pca.csv")

    # Merging training and test data
    combined_data = pd.concat([train_data, test_data])

    # Retrieving unique clusters and their counts
    clusters = sorted(combined_data['Cluster'].unique())

    # Sidebar: displaying the number of documents in each cluster
    st.sidebar.header("Cluster statistics")
    cluster_counts = combined_data['Cluster'].value_counts().sort_index()

    # Adding "Select all" and "Deselect all" buttons
    select_all = st.sidebar.button("Select all")
    deselect_all = st.sidebar.button("Deselect all")

    # Sidebar: selecting clusters
    st.sidebar.header("Select cluster")

    # Initializing checkbox states if not present
    if "checkbox_states" not in st.session_state:
        st.session_state.checkbox_states = {cluster: True for cluster in clusters}

    # Logic for buttons
    if select_all:
        st.session_state.checkbox_states = {cluster: True for cluster in clusters}
    if deselect_all:
        st.session_state.checkbox_states = {cluster: False for cluster in clusters}

    # Dynamically creating checkboxes
    selected_clusters = []
    for cluster in clusters:
        count = cluster_counts.get(cluster, 0)  # Getting the document count for the cluster
        state = st.session_state.checkbox_states[cluster]
        state = st.sidebar.checkbox(f"Cluster {cluster} ({count} clinical trials)",
                                    value=state, key=f"checkbox_{cluster}")
        st.session_state.checkbox_states[cluster] = state
        if state:
            selected_clusters.append(cluster)

    # Checking if at least one cluster is selected
    if not selected_clusters:
        st.warning("Select at least one cluster to see results.")
        return

    # Sidebar: selecting a tag with occurrence count
    st.sidebar.header("Select tag")

    # Generating the most frequent words in the dataset
    top_words = calculate_top_words(combined_data)

    # Adding "No tag selected" as the first option in the tag list
    tag_options = ["No tag selected"] + [f"{word} ({count})" for word, count in
                                         top_words]

    # Adding a dropdown list to the sidebar with a default option
    selected_tag = st.sidebar.selectbox("Select tag:", tag_options, index=0)

    # Checking if the user has selected a tag
    selected_word = None
    if selected_tag != "No tag selected":
        selected_word = selected_tag.split(" ")[
            0]  # Extracting only the word from the selected tag

    # Cluster plots
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clusters - training data")
        fig_train = plot_clusters(train_data, selected_clusters, "Clusters (Training data)")
        st.pyplot(fig_train)
    with col2:
        st.subheader("Clusters - testing data")
        fig_test = plot_clusters(test_data, selected_clusters, "Clusters (Testing data)")
        st.pyplot(fig_test)

    # Handling the selected tag
    selected_word = None  # Initializing the variable before use

    if selected_tag:
        selected_word = selected_tag.split(" ")[
            0]  # Extracting only the word from the selected tag

    # WordCloud for selected clusters
    st.subheader("WordCloud for selected clusters")
    fig_wordcloud = generate_wordcloud(combined_data, selected_clusters)
    st.pyplot(fig_wordcloud)

    # Displaying related data below the WordCloud
    if selected_word:
        display_related_data(combined_data, selected_word)

if __name__ == "__main__":
    main()