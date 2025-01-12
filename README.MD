# Machine Learning (CS 412) Term Project

Sabanci University, 2024-2025 Fall.  
Machine Learning (CS 412) Course Term Project.

![](Media/washing-machine-trampoline.gif)

## Team Members

- Tasnylu Akhmetova ([@tansylu](https://github.com/tansylu))
- Egi Cekici ([@egi-c](https://github.com/egi-c))
- Tugrul Agrikli ([@tagrikli](https://github.com/Tagrikli))

---


> It was nice to work with you all. It was quite a long and tiring journey, but it is finally over and we can rest now. Thank you very much for all of your contributions, as well as your supportive and friendly attitudes. I might not have come this far without your help. `Tugrul`



## Project Goal

This machine learning project addresses classification and regression tasks in Instagram data analysis, utilizing a dataset of 5,415 Instagram accounts, including detailed account profiles and metrics from 35 posts per account.

Both models will undergo a three-round evaluation process, where performance is tested against different test datasets provided sequentially by the lecturer. This multi-round testing approach ensures the robustness of the models against varying data distributions and helps validate their generalization capabilities.

The project leverages the rich feature set available in the dataset, including account biographical information, follower metrics, and historical post data, to create robust and accurate prediction models. The project allows for exploration and comparison of different machine learning algorithms to determine the most effective approach for each task.


## Regression

The regression task focuses on predicting the like count for individual Instagram posts, using both account characteristics and post-specific information such as captions and media type. The model's performance will be evaluated using Mean Squared Error (MSE) on log-transformed like counts (log10).

## Classification

The classification task aims to develop a model that can accurately categorize Instagram accounts into 10 distinct categories: entertainment, food, travel, health and lifestyle, mom and children, fashion, tech, sports, art, and gaming. This model will be trained on a labeled subset of 2,743 accounts and evaluated using accuracy metrics.

### Dataset cleaning and preparation for classification

`train-classification.csv` file included `username` and `category` pairs to train the models for classification. Categories are processed and cleaned, any row with `Null` item has been dropped and resulting file is saved as a dataframe to use later.


`train-classification.csv` included all of the data required for both classification and regression. Each row included profile and posts information of a single account.  
First we normalized the json values provided in fields and evaluated fields predictive power on categories based on several metrics.  

First we checked the distribution of the data to see if there is any imbalance that would decrease our accuracy, there was an imbalance but it wasnt too much, and since our evaluation metric will be accuracy, we decided not to attempt make dataset more balanced.

Profile data included numerous fields both helful for classification like `biography`, `is_verified`, `fbid` and others. 

Fields which used by instagram for identification are dropped since they are not relevant to our aim in any way.

Reamining labels are examiend closly to see their predictive power. We concucted chi-square test and calculated Cramers V value to see fields independence.

After this evaluation, we decided that the fields `biography` and `category_enum` from accounts profile information would've been helpful for us to classify accounts. 

We also used `caption` field from the posts of these accounts for classification.

At the end of the preperation, 2 datasets are generated for further processing.

1. Dataset for account information, included: `username`, `biography` and, `category_enum` fields for accounts.
2. Dataset for post information, included: `username`, `caption` fields for posts.

### Text Processing

Since we are aimed to do classification based on textual data, we needed a way to turn textual data into numerical representation. After a group discussion, we concluded that in the time that we live, "Bag of Words" or "TFIDF" representation of text would be too simplistic for this project and we needed to use neural network based approach to create Embeddings of the text data. 

We used 3 different models for creating embeddings:

- Our lecturer, Onur Varol ([@onurvarol](https://github.com/onurvarol))  who is the main auther of the article [TurkishBERTweet: Fast and Reliable Large Language Model for Social Media Analysis
](https://arxiv.org/abs/2311.18063) had a Github repository [TurkishBERTweet](https://github.com/ViralLab/TurkishBERTweet) where he published the finetuned model. Since the model is already pretrained on text which was written on social media (Twitter), we thought that it must have a good capability of quality embeddings, since our data was also turkish and consisted of social media text.  
We extracted **CLS Embeddings** from the output of the model.

- We discovered a python package called [SentenceTransformers](https://www.sbert.net/index.html) which go-to Python module for accessing, using, and training state-of-the-art text and image embedding models. The package was providing a model called [distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) which is a multilingual semantic similarity model and we used this model to generate embeddings as well.

- Lastly, we used OpeAI's [text-embedding3-large](https://platform.openai.com/docs/guides/embeddings) model to generate our embeddings. We prepared and sent a batch request for processing and got our results after ~9 hours of processing.


### Embedding Processing

After we acquired the embeddings of both biographies and captions of the posts of the accounts, we needed a way to combine them in a meaningful way before train our models. We tried going 2 ways

- We take the mean of all of the captions of the posts of an account and took a further weighted averaging them with the embedding of the biography of the account to get the final embedding.

- We calculated a centroid from all embeddings values of the posts and took weithed average of them based on their similarity to the average meaning to give less importance to the captions would lie outside of the mean category of the account.

After experimentation we combined the embeddings of the captions and the biography of the accounts with weigths of `0.9` and `0.1`, respectively to give more importance to the biography of the account.


### Training

We trained several models and compared their accuracy. 
Here is the list of models tha we trained.
1. CatBoost
2. k-NN
3. LightGbm
4. RandomForest
5. XGBoost


...