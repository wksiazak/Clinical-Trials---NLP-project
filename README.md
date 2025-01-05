# Clinical-Trials---NLP-project
This code allows to deep dive into details of clinical trials data by using NLP techinques. Source of data: [ClinicalTrials.gov](https://www.clinicaltrials.gov/search?viewType=Table&limit=100&aggFilters=status:com)

## General info <a name="general-info"></a>

<p>Main purpose of this project was to analyse what kind of text data are clinical trials and which NLP techniques are the best ones for this task. 
  This project is based on 4000 clinical trials downloaded from official website -> [ClinicalTrials.gov](https://www.clinicaltrials.gov/search?viewType=Table&limit=100&aggFilters=status:com)  
    
The project consists of several steps, including:</p>

* Data preparation:  
  Original file was downloaded as json file with some columns with nested structrures. Therefore there was need to flatten file and select columns with most important information for text analysis. Next major step was to clean data (remove stopwords, blanks etc. )
  After finishing this step we had first insight with what kind of data we will work -> by creating first Word Cloud
* TF IDF - used for general analysis  
* TF IDF - used for clustering and unsupervised machine learnign methods (Random Forest Classifier)
* ...


## Technolgies <a name="technologies/libraries"></a>
<ul>
<li>Python -  scripts are written in Python</li>
<li>sklearn</li>
<li>matplotlib</li>
</ul>


## Results 


## Summary
tbd 

### Possible Improvements:
tbd
