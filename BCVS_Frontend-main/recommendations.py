from flask import request
import json
import os
import requests
from keybert import KeyBERT
from collections import Counter
from spacy.lang.fr.stop_words import STOP_WORDS as fr_stop
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
import re
kw_model = KeyBERT("paraphrase-multilingual-MiniLM-L12-v2")


def get_portfolio():
    # portfolio = {'crypto': 70,
    #              'stocks': 20,
    #              'MF': 10}

    portfolio = {
        "hybrid funds (MF)": 267,
        "tax saving funds (MF)": 549,
        "equity funds (MF)": 4760,
        "debt funds (MF)": 172,
        "Bitcoin (CC)": 549,
        "Ethereum (CC)":  654,
        "Dodge (CC)": 31,
        "large CAP (S)": 452,
        "medium CAP (S)": 128,
        "small CAP (S)": 526         
    }
    return portfolio

def recommendations():
    output = {}
    subscription_key = "1efefb33177548b2b8b1c63a283c6bc1"
    endpoint ='https://api.bing.microsoft.com/' + "v7.0/search"
    new=[]
    url=[]
    title_list = []
    portfolio = get_portfolio()
    for key in (portfolio.keys()):
        query=key+'current info'
        mkt = ['en-US']
        sortBy='Date' #Date
        params = { 'q': query, 'mkt': mkt, 'sortBy':sortBy}
        headers = { 'Ocp-Apim-Subscription-Key': subscription_key}
        try:
                response = requests.get(endpoint, headers=headers, params=params)
                response.raise_for_status()
                search_response=response.json()
                pages=search_response['webPages']
                # # pprint(pages)
                # print(pages.keys())
                # pprint(pages['value'])
                value=pages['value']
                t=int(portfolio[key]/10)
                print(t)
                for i in range(0,t):
                    url1=value[i]['url']
                    data=value[i]['snippet']
                    title = value[i]['name']
                    data=data.replace('\n',' ')
                    data=data.replace('\t',' ')
                    # data=regex.sub(' ', data)
                    data=data.strip()
                    if len(data)>0:
                        new.append(data.lower())
                        url.append(url1)
                        title_list.append(title)
        except Exception as es:
            pass
    url, snippet, keywords_list = file_generation(new, url)
    output['URL'] = url
    output['title'] = title_list
    output['Snippet'] = snippet
    output['Keywords'] = keywords_list
    output['portfolio_optoins'] = list(portfolio.keys())
    output['portfolio_values'] = list(portfolio.values())
    direct = os.getcwd()
    path = os.path.join(direct, 'static/output')
    with open(os.path.join(path, 'recommendation.json'), 'w') as outfile:
        fun = json.dump(output, outfile)
    return output


def word_count(str):
    counts = dict()
    words = str.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1

    return counts


def file_generation(snippet, url):
    final_stopwords_list = list(fr_stop) + list(en_stop)
    keywords_list = []
    for i in range(0, len(snippet)):
        # hyperparameter tuning needed
        doc = snippet[i]
        keywords = kw_model.extract_keywords(doc, use_mmr=True, keyphrase_ngram_range=(
            1, 1), diversity=0.5, stop_words=final_stopwords_list)
        temp = []
        for keys in keywords:
            word = keys[0]
            temp.append(word)
        tempstr = ", ".join(temp)
        keywords_list.append([tempstr])
        # keywords_strings=", ".join(keywords_list)

    # output['URL']=url
    # output['Snippet']=snippet
    # output['Keywords']=keywords_list
    return url, snippet, keywords_list

