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
  In next step I computed TF-IDFs on train and test data and then performed clustering and reduced dimension. The results we can verify in prepared streamlit app.
  Main insights:
  - both train and test data are nicely clustering in main groups - in this case I selected main 9 clusters where each represents different symptoms 
  
* Random Forest Classifier
* NER
* Word Embeding

## Technolgies <a name="technologies/libraries"></a>
<ul>
<li>Python -  scripts are written in Python</li>
<li>sklearn</li>
<li>matplotlib</li>
</ul>

## Summary
tbd 

### Possible Improvements:
tbd
