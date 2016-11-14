# -*- coding: utf-8 -*-

import pandas as pd
import os
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
directory = os.path.dirname(os.path.abspath(__file__))

comb_news = pd.read_csv(directory+"/stocknews/Combined_News_DJIA.csv")

stock_prices = pd.read_csv(directory+"/stocknews/DJIA_table.csv")

combined = {}
sid = SentimentIntensityAnalyzer()

comb_news_10 = comb_news.head(10)
keys = []
preprocessed = []
for row in comb_news.iterrows():

    top1 = row[1][2][:].strip()
    date = row[1][0]


    ss = sid.polarity_scores(top1)
    #print(top1)
    #print(ss)
    converted_stats = [date,top1]
    keys = list(ss.keys())
    for v in sorted(ss):
        converted_stats.append(ss[v])
    #print(sorted(ss))

    series = pd.Series(converted_stats)
    combined[date] = converted_stats
    #preprocessed.append(converted_stats)

for row in stock_prices.iterrows():
    date = row[1][0]

    prices = [x for x in row[1][1:]]
    diff = prices[0] - prices[3]

    if diff >= 0:
        diff = 1
    else:
        diff = 0

    prices.append(diff)
    combined[date].extend(prices)

final_output = [v for k,v in sorted(combined.items())]
#print(final_output[0])
headers = ["date","sentence","compound","neg","neu","pos","open","high","low","close","volume","adj close","diff"]

preprocessed = pd.DataFrame(data=final_output,columns=headers)


preprocessed.to_csv(path_or_buf=directory+"/stocknews/preprocessed.csv",columns=headers)
