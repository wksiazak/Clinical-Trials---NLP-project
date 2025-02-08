# Clinical-Trials---NLP-project
This code allows to deep dive into details of clinical trials data by using NLP techinques. Source of data: [ClinicalTrials.gov](https://www.clinicaltrials.gov/search?viewType=Table&limit=100&aggFilters=status:com)

## 🚀 Live Demo

You can access the live version of the Streamlit app here:

👉 [Click to Open Streamlit App](https://clinical-trials---nlp-project-vvwtszf2ujmtqwgjayyj2m.streamlit.app/)

## 📖 General info <a name="general-info"></a>

<p> 📌 Main purpose of this project was to analyse what kind of text data are clinical trials and which NLP techniques are the best ones for this task. 
  This project is based on 4000 clinical trials.
    
The project consists of several steps, including:</p>

* **Data preparation(import&cleaning)**:  
  Original file was downloaded as json file with some columns with nested structrures. Therefore there was need to flatten file and select columns with most important information for text analysis. Next major step was to clean data (remove stopwords, blanks etc. )
  After finishing this step we received first insight with what kind of data we will work -> by creating first Word Cloud  
  ![WordCloud](https://github.com/wksiazak/Clinical-Trials---NLP-project/blob/master/working_files/Word_cloud_general_clinical_trials.png)

* **TF-IDF Vectorizer**
  In next step I computed TF-IDFs on train and test data and then performed clustering and reduced dimension. The results we can verify in prepared **streamlit app**.
  Main insights:
  - both train and test data are nicely clustering in main groups - in this case I selected main 9 clusters where each represents different symptoms
  - Word Clouds were prepared for every cluster and viewer can check each of them in dynamic way
  - Tags were counted as more frequent words and viewer can also check in which clinical trials we may find them 
  - the **Kullback-Leibler (KL**) divergence measures how much one probability distribution differs from another - in this example the KL divergence between the training and test distributions is 0.0104 -  it suggests that the training and test data distributions are well-aligned, reducing the risk of distribution shift and ensuring that the model's performance on the test set should be a good indicator of real-world performance.
  
* **Random Forest Classifier**
  After analysing of TF-IDF and clustering  I used Random Forrest Classifier algorithm to check predictions of cluster on test data. Applying **SMOTE** metod and **Gradient Boosting** model I received quite good Overall Accuracy: 83% -> details you may find in in the folder "notebooks" in workbook with TF-IDF
* **NER**
  NER (Named Entity Recognition)  - in the folder "notebooks"  you may find dedicated workbook for this analysis. I tried fine-tuning approach on my data with using **BioBert** model and NER with extrenal model like "**en_core_web_sm**"
* **Word Embeding**
  In the folder "notebooks"  you may find dedicated workbook for this analysis. It uses Word2Vec which is a word embedding technique, clustering and cosine similarity calculated for clusters,  


## Technolgies & Libraries  <a name="technologies/libraries"></a>
<ul>
<li>Python -  scripts are written in Python</li>
<li>pandas</li>
<li>nltk</li>
<li>sklearn</li>
<li>matplotlib</li>
<li>seaborn</li>
</ul>

## Summary
tbd 

### Possible Improvements:
tbd
