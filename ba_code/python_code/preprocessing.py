import email
import re
from time import perf_counter

import bs4 as bs
import nltk
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from spellchecker import SpellChecker
from autocorrect import Speller


#preprocessing with emails in html format with header etc.
def preprocess_html_format(data):
    nltk.download('wordnet')
    nltk.download('stopwords')
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    text_list = []
    start = perf_counter()
    #extract the body from the html
    for mail in data:
        b = email.message_from_string(mail)
        body = ""
        if b.is_multipart():
            for part in b.walk():
                ctype = part.get_content_type()
                cdispo = str(part.get('Content-Disposition'))
                if ctype == 'text/plain' and 'attachment' not in cdispo:
                    body = part.get_payload(decode=True)  # get body of email
                    break
        else:
            body = b.get_payload(decode=True) # get body of email
        soup = BeautifulSoup(body, "html.parser") #get text from body (HTML/text)
        text = soup.get_text().lower()
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', 
                        ' ', 
                        text, 
                        flags=re.MULTILINE) #remove links
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                        ' ', 
                        text, 
                        flags=re.MULTILINE) #remove email addresses
        text = re.sub(r'[^\w\s]', ' ', text) # remove punctuation
        text = ''.join([i for i in text if not i.isdigit()]) # remove digits
        text = re.sub(r'\r',' ', text)
        text = re.sub(r'\n',' ', text)
        text = re.sub(r'_',' ', text)
        stop_words = stopwords.words('english')
        # remove stop words
        pattern = re.compile(r'\b(' + r'|'
                        .join(stopwords.words('english')) + r')\b\s*')
        text = pattern.sub(' ', text)
        #remove 'subject' and 'Subject' because they are stop words in our context
        text = re.sub('subject', ' ', text) 
        text = re.sub('Subject', ' ', text)
        words_list = [ps.stem(w) for w in nltk.word_tokenize(text)] #Stemming
        words_list = [wnl.lemmatize(w) for w in words_list] #Lemmatization
        text_list.append(' '.join(words_list))
    end = perf_counter()
    print(f'It took {end-start} second(s) to finish preprocessing.')
    return text_list

#preprocessing - text contains only message and subject.
def preprocess_only_text_format(data):
    nltk.download('wordnet')
    nltk.download('stopwords')
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    text_list = []
    start = perf_counter()
    for mail in data:
        text = mail.lower()
        text = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b',
                        ' ', text, 
                        flags=re.MULTILINE) #remove links
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                        ' ', text, 
                        flags=re.MULTILINE) #remove email addresses
        text = re.sub(r'[^\w\s]', ' ', text) # remove punctuation
        text = ''.join([i for i in text if not i.isdigit()]) # remove digits
        text = re.sub(r'\r',' ', text)
        text = re.sub(r'\n',' ', text)
        text = re.sub(r'_',' ', text)
        stop_words = stopwords.words('english')
        # remove stop words
        pattern = re.compile(r'\b(' + r'|'
                        .join(stopwords.words('english')) + r')\b\s*')
        text = pattern.sub(' ', text)
        #remove 'subject' and 'Subject' because they are stop words in our context
        text = re.sub('subject', ' ', text)
        text = re.sub('Subject', ' ', text)
        words_list = [ps.stem(w) for w in words_list] #Stemming
        words_list = [wnl.lemmatize(w) for w in text.split()] #Lemmatization
        text_list.append(' '.join(words_list))
    end = perf_counter()
    print(f'It took {end-start} second(s) to finish preprocessing.')
    text_series = pd.Series(text_list)
    return text_series

#The following methods are examples of spell checkers.
#All spell checkers we test are too time consuming for larger datasets.

#Spelling correction for a given text with pyspellchecker
def spell_correct(text):
    spell = SpellChecker()
    my_list = []
    for word in text:
        try:
            #get corrected word from dictionary
            corrected = ''.join([spell.correction(word)]) 
        except:
            corrected = word
        my_list.append(corrected)   
    return my_list  

#apply spelling correction on a dataset which is a list of texts
def spelling_correction(mails):
    start = perf_counter()
    result = [spell_correct(text.split()) for text in mails]
    end = perf_counter()
    print(f'It took {end-start} second(s) to finish.')
    return result

#Spelling correction for a given text with autocorrect
def spell_autocorrect(text):
    spell = Speller(lang='en')
    my_list = []
    for word in text:
        try:
            #get corrected word from dictionary
            corrected = ''.join([spell(word)])
        except:
            corrected = word
        my_list.append(corrected)   
    return my_list  

#apply spelling correction on a dataset which is a list of texts
def spelling_autocorrection(mails):
    start = perf_counter()
    result = [spell_autocorrect(text.split()) for text in mails]
    end = perf_counter()
    print(f'It took {end-start} second(s) to finish.')
    return result
