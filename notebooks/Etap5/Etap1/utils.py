import pandas as pd
import numpy as np
import string
import json
from IPython.display import display
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet
from nltk.corpus import words
from nltk.corpus import stopwords
from emoji import UNICODE_EMOJI
import re
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

RE_HTTP = re.compile("http[s]?://[/\.a-zA-Z0-9]+")
def getListOfURLs(text):
    return RE_HTTP.findall(str(text))

def countEmojis(text):
    count = 0
    for em in UNICODE_EMOJI['en'].keys():
        if em in str(text):
            count += 1
    return count

def removePunctuationInString(text):
    new_text = [char for char in text if char not in string.punctuation]
    return ''.join(new_text)

lemmatizer = WordNetLemmatizer()
def lemmatize(sentence):
    # POS_TAGGER_FUNCTION : TYPE 1
    def pos_tagger(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    # tokenize the sentence and find the POS tag for each token
    pos_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  

    # we use our own pos_tagger function to make things simpler to understand.
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])), pos_tagged))

    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            # else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    lemmatized_sentence = " ".join(lemmatized_sentence)

    return lemmatized_sentence

def cleanText(text):
    text = text.lower()
    words_list = text.split(' ')   
    words_list = [removePunctuationInString(word) for word in words_list]
    words_without_stopwords = [word for word in words_list if word not in stopwords.words('english') and word != '']
    words_lemmatized = [lemmatize(word) for word in words_without_stopwords]
    
    return ' '.join(words_lemmatized)

def checkIfFacebookInListOfURLs(list_of_urls):
    isFacebook = False

    for url in list_of_urls:
        url = url.lower()
        if re.search('facebook', url) or re.match('fb.com', url):
            isFacebook = True
    return isFacebook

def checkIfInstagramInListOfURLs(list_of_urls):
    isInstagram = False
    for url in list_of_urls:
        url = url.lower()
        if re.search('instagram', url):
            isInstagram = True 
    return isInstagram

def checkIfTwitterInListOfURLs(list_of_urls):
    isTwitter = False
    for url in list_of_urls:
        url = url.lower()
        if re.search('twitter', url):
            isTwitter = True
    return isTwitter

# formatowanie czasu i rozbicie na atrybuty  
def addColumnsAndSaveCSV(df, pathToSave):
    ###### Adam ######
    for index, row in df.iterrows():    
        pubdate = datetime.strptime(row['publish_time'],'%Y-%m-%dT%H:%M:%S.%fZ')
        df.loc[index,'pub_date'] = pubdate
        df.loc[index,'pub_day_of_the_week'] = pubdate.weekday()
        df.loc[index,'pub_year'] = pubdate.year
        df.loc[index,'pub_month'] = pubdate.month
        df.loc[index,'pub_day'] = pubdate.day
        df.loc[index,'pub_hour'] = pubdate.hour
        trendate = datetime.strptime(row['trending_date'] + ' 23:59:59','%y.%d.%m %H:%M:%S') # Wymuszamy godzinę 23:59:59
        df.loc[index,'tren_date'] = trendate
        df.loc[index,'tren_day_of_the_week'] = trendate.weekday()
        df.loc[index,'tren_year'] = trendate.year
        df.loc[index,'tren_month'] = trendate.month
        df.loc[index,'tren_day'] = trendate.day
        df.loc[index,'time_to_trend_in_days'] = (trendate - pubdate).total_seconds()/(3600*24)

    df.loc[df['dislikes'] == 0, 'dislikes'] = 1
    df.loc[df['comment_count'] == 0, 'comment_count'] = 1
    df.loc[df['views'] == 0, 'views'] = 1
    df.loc[df['likes'] == 0, 'likes'] = 1

    # stosunek likes, dislikes, views
    df['dislikes/likes'] = df['dislikes']/df['likes']
    df['likes/views'] = df['likes']/df['views']
    df['dislikes/views'] = df['dislikes']/df['views']
    df['comment_count/views'] = df['comment_count']/df['views']
    unique = df['video_id'].unique()

    for uid in unique:
        temp = df[df['video_id']==uid]
        times_in_trend = len(temp)
        increase_views = 0
        increase_likes = 0
        increase_dislikes = 0
        increase_comms = 0
        indexes =  temp.index.values.tolist() 
        if times_in_trend > 1:
            increase_views = temp['views'][indexes[1]] - temp['views'][indexes[0]]
            increase_likes = temp['likes'][indexes[1]] - temp['likes'][indexes[0]]
            increase_dislikes = temp['dislikes'][indexes[1]] - temp['dislikes'][indexes[0]]
            increase_comms = temp['comment_count'][indexes[1]] - temp['comment_count'][indexes[0]]

        df.loc[indexes[0],'increase_views'] = increase_views
        df.loc[indexes[0],'increase_likes'] = increase_likes
        df.loc[indexes[0],'increase_dislikes'] = increase_dislikes
        df.loc[indexes[0],'increase_comms'] = increase_comms
        df.loc[indexes[0],'times_in_trend'] = times_in_trend
        df.loc[indexes[0],'avg_views'] = np.mean(temp['views'])
        df.loc[indexes[0],'avg_likes'] = np.mean(temp['likes'])
        df.loc[indexes[0],'avg_dislikes'] = np.mean(temp['dislikes'])
        df.loc[indexes[0],'avg_comms'] = np.mean(temp['comment_count'])
#         df.loc[indexes[0],'avg_views_increase_per_hour'] = temp['views'][0] / temp['time_to_trend_in_days'][0]
    df = df[df['avg_comms'] >= 0] 
    
    df["avg_views_increase_per_hour"] = df.apply(lambda row: row.views/row.time_to_trend_in_days, axis=1)
    ###### Marcin ######
    df['tags'] = df['tags'].apply(lambda tags: json.dumps(tags.replace('"','').split('|')))
    df["n_of_tags"] = df.apply(lambda row: len(json.loads(row.tags)), axis=1)

    df["title_clean"] = df.apply(lambda row: cleanText(row.title), axis=1)
    df["title_length"] = df.apply(lambda row: len(row.title), axis=1)
    df["title_n_of_words"] = df.apply(lambda row: len(row.title.split(' ')), axis=1)
    df["title_capital_letters"] = df.apply(lambda row: sum(1 for l in row.title if l.isupper()), axis=1)
    df["title_capital_letters_percent"] = df.apply(lambda row: row.title_capital_letters/row.title_length, axis=1)
    df["title_small_letters"] = df.apply(lambda row: sum(1 for l in row.title if l.islower()), axis=1)
    df["title_small_letters_percent"] = df.apply(lambda row: row.title_small_letters/row.title_length, axis=1)
    df["title_punctuation"] = df.apply(lambda row: sum(1 for l in row.title if l in string.punctuation), axis=1)
    df["title_punctuation_percent"] = df.apply(lambda row: sum(1 for l in row.title if l in string.punctuation)/len(row.title), axis=1)
    df["title_n_of_emojis"] = df.apply(lambda row: str(countEmojis(row.title)), axis=1)

    df["desc_clean"] = df.apply(lambda row: cleanText(row.description), axis=1)
    df["desc_length"] = df.apply(lambda row: len(row.description), axis=1)
    df["desc_n_of_words"] = df.apply(lambda row: len(row.description.split(' ')), axis=1)
    df["desc_capital_letters"] = df.apply(lambda row: sum(1 for l in row.description if l.isupper()), axis=1)
    df["desc_capital_letters_percent"] = df.apply(lambda row: row.desc_capital_letters/row.desc_length if row.desc_length != 0 else 0, axis=1)
    df["desc_small_letters"] = df.apply(lambda row: sum(1 for l in row.description if l.islower()), axis=1)
    df["desc_small_letters_percent"] = df.apply(lambda row: row.desc_small_letters/row.desc_length if row.desc_length != 0 else 0, axis=1)
    df["desc_punctuation"] = df.apply(lambda row: sum(1 for l in RE_HTTP.sub(' ', row.description.replace(r'\n', ' ')) if l in string.punctuation), axis=1)
    df["desc_punctuation_percent"] = df.apply(lambda row: (sum(1 for l in row.description if l in string.punctuation)/len(row.description)) if len(row.description)!=0 else 0 , axis=1)
    df["desc_list_of_urls"] = df.apply(lambda row: json.dumps(getListOfURLs(row.description)), axis=1)
    df["desc_n_of_urls"] = df.apply(lambda row: len(getListOfURLs(row.description)), axis=1)
    df["desc_n_of_emojis"] = df.apply(lambda row: str(countEmojis(row.description)), axis=1)
    
    df["isFacebook"] = df.apply(lambda row: checkIfFacebookInListOfURLs(json.loads(row.desc_list_of_urls)), axis=1)
    df["isTwitter"] = df.apply(lambda row: checkIfTwitterInListOfURLs(json.loads(row.desc_list_of_urls)), axis=1)
    df["isInstagram"] = df.apply(lambda row: checkIfInstagramInListOfURLs(json.loads(row.desc_list_of_urls)), axis=1)
    
    df.to_csv(pathToSave, index=False)
    return df

