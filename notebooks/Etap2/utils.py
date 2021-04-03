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
from  datetime import datetime

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

cnt = 0
def cleanOcrText(text):
    global cnt
    if (cnt % 100 == 0):
        print("cleanOcrText:", cnt, end='\r')
    cnt += 1
    
    if (str(text) == 'nan'):
        return ''
    text = cleanText(text)
    list_of_words = [word for word in text.split() if len(word) > 2]
    list_of_words = [word for word in list_of_words if word in words.words()]
    
    return ' '.join(list_of_words)

cntV2 = 0
def concatAndCleanOcrTextV2(df):
    global cntV2
    if (cntV2 % 100 == 0):
        print(cntV2, end='\r')
    cntV2 += 1
        
    words_list = df.loc[:,"ocr_text"].values
    bboxes_list = df.loc[:,"bbox"].values
    word_box_dict = dict(zip(words_list, bboxes_list))

    result_words_list = []
    result_bboxes_list = []
    for word, bbox in word_box_dict.items():
        if word not in stopwords.words('english'):
            word = lemmatize(word)
            if (len(word) > 2):# and word in words.words():
                result_words_list.append(word)
                result_bboxes_list.append(bbox)

    data = {
        "thumb_ocr_text_V2": ' '.join(words_list),
        "clean_thumb_ocr_text_V2": ' '.join(result_words_list),
        "bboxes": json.dumps(result_bboxes_list)
    }
    return pd.Series(data=data, index=['thumb_ocr_text_V2', 'clean_thumb_ocr_text_V2', 'bboxes'])