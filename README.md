# Aggressive Tweet Analyzer: an NLP app
### Ironhack Data Analytics Bootcamp - Final Project

Júlia Rodríguez Sánchez | *August 2021*



## Contents
   
1. [Project description](#id1)
2. [Business objectives](#id2)<br>
2.1. [Long-term goal](#id21)<br>
2.2. [Short-term goal](#id22)
3. [Datasets](#id3)<br>
3.1 [Training datasets](#id31)<br>
3.2 [Evaluation dataset](#id32)
4. [BERT fine-tuning](#id4)
5. [Evaluation](#id5)<br>
5.1. [Test split](#id51)<br>
5.2. [Evaluation data](#id52)
6. [Web app](#id6)
7. [Next steps](#id7)
8. [Project structure](#id8)

## Project Description <a name="id1"></a>
The Aggressive Tweet Analyzer is a web app that takes a Twitter handle as input and analyzes the last 100 tweets of that user. It runs the tweets through an NLP model (based on BERT) that classifies them as aggressive or not. Then it returns a results page that contains an aggressiveness score for the user and the tweets classified as aggressive. 


## Business objectives<a name="id2"></a>
### Long-term goal<a name="id21"></a>
The ultimate aim of this project is serve as a cyber-bullying detector for FITA Fundación, an organization that works to prevent mental illness in Spain. After working with people affected by eating disorders, conduct disorders, addictions and other mental health problems, they have come to realize that bullying is a major risk factor for developing them. They are currently working on tools to detect bullying cases earlier and support the victims to prevent them from becoming ill. 

The model developed in this project detects comments that are very aggressive, so it serves as a first step to detect open attacks. However, cyber-bullying is ofter subtler, and the model needs to be further improved to detect these cases. 

### Short-term goal<a name="id22"></a>
The most immediate goal of this project is to detect aggressive comments. I have deployed the model as a standalone app, but it could also be used in any website, forum or social media platform that wants to ban aggressive language from its posts or comments section.

## Datasets<a name="id3"></a>

All the data used for this project can be found [here](https://drive.google.com/drive/u/1/folders/1v_0Qsvn43zcv3dMFjdeg-st1C0-qMAgK). See the [Project structure](#id8) to understand how the folders are organized. 

### Training datasets<a name="id31"></a>
The training datasets contain comments that are classified as different types of bullying (aggression, racism, sexism, etc) and that are sourced from different social media platforms. 

### Evaluation dataset<a name="id32"></a>
The long-term application of this model will be to detect cyber-bullying cases on social media. The two most popular platforms among teenagers are Instagram and TikTok, so I decided to evaluate the model with original Instagram data that I scraped myself. Using the Instaloader library, I obtained the comments of Kevin Spacey's last 3 Instagram posts. I chose this user because he receives many comments from haters as well as supporters. The evaluation dataset consists of 22k+ comments, 17k+ after deleting the ones that contained only emojis or words shorter that two characters. 

## BERT fine-tuning<a name="id4"></a>
The model I used as a basis for this project is BERT, a transformer. Transformers use an attention mechanism that learns contextual relations between words (or sub-words) in a text. BERT is pre-trained on a large corpus of English text data.

To fine-tune the model I used the prebuilt <CODE>TFBertForSequenceClassification</CODE> class and trained it. My intention was to use all the datasets to train a binary classifier that would sort comments as "bullying" or "not bullying". However, as I trained the model with more datasets its score decreased, so I decided to train it with the aggression data only. A multi-class classifier would be more appropriate to train with several datasets. 

The weights of the fine-tuned models I tested in the notebooks can be found [here](https://drive.google.com/drive/u/1/folders/1fdrckXTMFYfj9R33LwH2Xi3CsVxReW6e).

## Evaluation<a name="id5"></a>

### Test split<a name="id51"></a>
When predicting the labels of the test split of the aggression dataset, the model achieved the following metrics:  

<img src="https://i.imgur.com/tHKiC64.png" alt="evaluation" align="center"/>

### Evaluation data<a name="id52"></a>
Since the evaluation data was obtained directly from Instagram, it was not pre-labeled. 
* Out of 17.749 comments, 83% were predicted as class 0 ("not aggressive") and 17% as class 1 ("aggressive"). 
* For the comments predicted as class 1, I manually checked if the label was correct and found that in 86% of the cases it was correct. In most of the cases where it was incorrect, the comment contained swear words used in a friendly way. 

## Web app<a name="id6"></a>
I deployed the model into production using Flask and the Twitter API. 
* **Flask** is a micro web framework well suited for building light-weight apps. 
* The **Twitter API** allows developers to programmatically access to public Twitter data.

In the home page of the app, users can enter a Twitter handle to analyze:

<img src="https://i.imgur.com/lvfX5PX.png" alt="homepage" width="100%" height="" align="center"/>

When they hit "Analyze", the app makes a request to the Twitter API to get the last 100 tweets of that username, including replies but excluding retweets. It preprocesses and tokenizes the tweets, then runs them throught the model to classify them. Finally, it renders the results page, which includes an aggressiveness score and the tweets labeled as aggressive:


<img src="https://i.imgur.com/tqzKLnM.png" alt="results-page-top" width="100%" height="" align="center"/>

The user can click on the tweets and navigate to the original tweet URL or hit "Try again" (at the bottom of the page) and go back to the home. 

<img src="https://i.imgur.com/UUPguDy.png" alt="results-page-bottom" width="100%" height="" align="center"/>

## Next steps<a name="id7"></a>

* Train a multi-class classifier to detect more types of bullying such as racism and sexism. 
* Train the model with comments that contain swear words but are not aggressive.
* Build the Keras layers from scratch using <CODE>TFBertModel</CODE> and test more parameters. 
* Create spiders to monitor Instagram and TikTok in order to find victims of cyber-bullying. 

## Project structure<a name="id8"></a>

**GitHub repository**:

* web_app folder: contains the code for the local deployment of the Aggressive Tweet Analyzer. To execute the code you need to download the model weights from the drive and create a json file with your Twitter API credentials.
* 1_EDA_Preprocessing_Tokenization.ipynb
* 2_BERT_Fine_tuning.ipynb
* 3_Evaluation.ipynb
* 4_Web_app.ipynb

**Drive**:

* [data](https://drive.google.com/drive/u/1/folders/1v_0Qsvn43zcv3dMFjdeg-st1C0-qMAgK):
  * [1_raw_data](https://drive.google.com/drive/u/1/folders/1UcjJq4Bjyf_qZZnOJlFJD_q9nNWk652g): training datasets obtained from Mendeley Data. ([original source](https://data.mendeley.com/datasets/jf4pzyvnpj/1)) 
  * [2_clean_data](https://drive.google.com/drive/u/1/folders/1sMGR9NgNA_MzMGuIvzbhFWIN43Iv3xZD): preprocessed training datasets.
  * [3_tokenized_data](https://drive.google.com/drive/u/1/folders/1gtwIK8Mxx7BtrTBz57DaZhHnTFhfSrjW): pickle files containing input ids, attention masks and labels ready to be inputed into the BERT model. Includes training and evaluation data. 
  * [4_evaluation_data](https://drive.google.com/drive/u/1/folders/17G0_j-W7wlmJ9RzY8EhEzIkT8oKbbJkX): 
    * [raw_comments](https://drive.google.com/drive/u/1/folders/1_kZKP7M4ZXM4UZWj97pdgICNRsbWK6iY): 3 json files containing comments from 3 Instagram posts, extracted directly from the platform with Instaloader. 
    * clean_evaluation_data.csv: preprocessed comments.
    * labeled_evaluation_data.csv: comments labeled by the model.
    * labeled_evaluation_data_checked.csv: same as the provious file but with an added column <CODE>wrong_label</CODE>.

* [models](https://drive.google.com/drive/u/1/folders/1fdrckXTMFYfj9R33LwH2Xi3CsVxReW6e): saved weights of the fine-tuned BERT models. 
  * [histories](https://drive.google.com/drive/u/1/folders/1UKlXsHP04y-F_ZqfXjfYuGFez8rR_P7y): saved histories for the fine-tuning of each model. 













