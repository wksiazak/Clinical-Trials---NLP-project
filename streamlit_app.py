import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import seaborn as sns


# Funkcja do tworzenia wykresów klastrów z legendą
def plot_clusters(data, selected_clusters, title):
    filtered_data = data[data['Cluster'].isin(selected_clusters)]
    fig, ax = plt.subplots(figsize=(6, 6))

    # Lista unikalnych klastrów do iteracji
    unique_clusters = sorted(filtered_data['Cluster'].unique())

    # Tworzenie punktów z etykietą dla legendy
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
    ax.legend(loc="best", title="Clusters")  # Dodanie legendy
    return fig


# Funkcja do generowania WordCloud na podstawie wybranego klastra (bez filtrowania po tagu)
def generate_wordcloud(data, selected_clusters):
    # Filtrowanie danych tylko na podstawie wybranych klastrów
    filtered_data = data[data['Cluster'].isin(selected_clusters)]

    # Łączenie tekstu dla wybranych klastrów
    combined_text = " ".join(filtered_data['Combined_Text'])

    # Generowanie WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate(combined_text)

    # Rysowanie WordCloud
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("WordCloud for selected clusters")
    return fig


# Funkcja do obliczania najczęstszych słów w danych
def calculate_top_words(data, top_n=50):
    combined_text = " ".join(data['Combined_Text'])
    word_counts = Counter(combined_text.split())
    return word_counts.most_common(top_n)

# Funkcja do wyświetlania powiązanych NCTID, Combined_Text oraz Cluster + wykres słupkowy
def display_related_data(data, word):
    filtered_data = data[data['Combined_Text'].str.contains(word, na=False, case=False)]

    st.write(f"**Clinical trials for selected tag '{word}':**")
    st.dataframe(filtered_data[['nctId', 'Combined_Text', 'Cluster']])

    # Liczenie wystąpień tagu w poszczególnych klastrach
    cluster_counts = filtered_data['Cluster'].value_counts().sort_index()

    # Tworzenie wykresu słupkowego
    if not cluster_counts.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette="viridis",
                    ax=ax)
        ax.set_title(f"Occurrences of '{word}' in different clusters")
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Count")

        # Wyświetlenie wykresu pod tabelą
        st.pyplot(fig)


def main():
    st.title("Cluster Analysis prepared based on TF-IDF and PCA analysis ")

    st.markdown("##### Below visualization is presenting results of analysing text data using TF-IDF method. "
                "Data coming from https://clinicaltrials.gov/ - here we are using 4000 examples of different clinical trials")

    # Ładowanie danych
    train_data = pd.read_csv("working_files/train_clusters_with_pca.csv")
    test_data = pd.read_csv("working_files/test_clusters_with_pca.csv")

    # Połączenie danych treningowych i testowych
    combined_data = pd.concat([train_data, test_data])

    # Pobieranie unikalnych klastrów i liczności
    clusters = sorted(combined_data['Cluster'].unique())

    # Panel boczny: informacje o liczbie dokumentów w klastrach
    st.sidebar.header("Cluster statistics")
    cluster_counts = combined_data['Cluster'].value_counts().sort_index()

    # Dodanie przycisków "Zaznacz wszystkie" i "Odznacz wszystkie"
    select_all = st.sidebar.button("Select all")
    deselect_all = st.sidebar.button("Deselect all")

    # Panel boczny: wybór klastrów
    st.sidebar.header("Select cluster")

    # Inicjalizacja stanu checkboxów, jeśli nie istnieje
    if "checkbox_states" not in st.session_state:
        st.session_state.checkbox_states = {cluster: True for cluster in clusters}

    # Logika dla przycisków
    if select_all:
        st.session_state.checkbox_states = {cluster: True for cluster in clusters}
    if deselect_all:
        st.session_state.checkbox_states = {cluster: False for cluster in clusters}

    # Tworzenie checkboxów dynamicznie
    selected_clusters = []
    for cluster in clusters:
        count = cluster_counts.get(cluster, 0)  # Pobranie liczby dokumentów dla klastra
        state = st.session_state.checkbox_states[cluster]
        state = st.sidebar.checkbox(f"Cluster {cluster} ({count} clinical trials)",
                                    value=state, key=f"checkbox_{cluster}")
        st.session_state.checkbox_states[cluster] = state
        if state:
            selected_clusters.append(cluster)

    # Sprawdzenie, czy wybrano jakiekolwiek klastry
    if not selected_clusters:
        st.warning("Select at least one cluster to see results.")
        return

    # Panel boczny: wybór tagu z liczbą wystąpień
    # Panel boczny: wybór tagu z liczbą wystąpień
    st.sidebar.header("Select tag")

    # Generowanie najczęstszych słów w danych
    top_words = calculate_top_words(combined_data)

    # Dodanie opcji "no tag selected" na początek listy tagów
    tag_options = ["No tag selected"] + [f"{word} ({count})" for word, count in
                                         top_words]

    # Dodanie listy rozwijalnej do panelu bocznego z domyślną opcją
    selected_tag = st.sidebar.selectbox("Select tag:", tag_options, index=0)

    # Sprawdzenie, czy użytkownik nie wybrał tagu
    selected_word = None
    if selected_tag != "No tag selected":
        selected_word = selected_tag.split(" ")[
            0]  # Pobranie samego słowa z wybranego tagu

    # Wykresy klastrów
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Clusters - training data")
        fig_train = plot_clusters(train_data, selected_clusters, "Clusters (Training data)")
        st.pyplot(fig_train)
    with col2:
        st.subheader("Clusters - testing data")
        fig_test = plot_clusters(test_data, selected_clusters, "Clusters (Testing data)")
        st.pyplot(fig_test)

    # Obsługa wybranego tagu
    selected_word = None  # Zainicjalizowanie zmiennej przed użyciem

    if selected_tag:
        selected_word = selected_tag.split(" ")[
            0]  # Pobranie samego słowa z wybranego tagu

    # WordCloud dla wybranych klastrów
    # WordCloud dla wybranych klastrów (bez tagów)
    st.subheader("WordCloud for selected clusters")
    fig_wordcloud = generate_wordcloud(combined_data, selected_clusters)
    st.pyplot(fig_wordcloud)

    # Wyświetlanie powiązanych danych poniżej WordCloud
    if selected_word:
        display_related_data(combined_data, selected_word)

if __name__ == "__main__":
    main()