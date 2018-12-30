import csv
import pandas as pd
import numpy as np
import quandl as quandl
quandl.ApiConfig.api_key = "y94CFqx58gLCyaY9hsRf"
import tweepy
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM


consumer_key = 'nHv8Cx32VE2rXLhskSmR9JwDC'
consumer_secret = '7pKXbHO1mxIy87qPn85CKkiTziwMsddbo9mu4aW7EBJ3kgUHvV'
access_token = '805917270719557636-vZmznZ7QigDKLWzLAeNB3d6Y9BJ9nBx'
access_secret = 'U2nQL9SYMDkDpKyfJqzDTgyEIsYdzMqkK1Xx2jnD5L682'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

def get_all_tweets(screen_name):
    all_the_tweets = []
    new_tweets = api.user_timeline(screen_name=screen_name, count=200, include_rts=False)
    all_the_tweets.extend(new_tweets)
    oldest_tweet = all_the_tweets[-1].id-1
    while(len(new_tweets)>0):
        # The max_id param will be used subsequently to prevent duplicates
        new_tweets = api.user_timeline(screen_name=screen_name,
                                       count=200, max_id=oldest_tweet, include_rts=False)

        # save most recent tweets
        all_the_tweets.extend(new_tweets)

        # id is updated to oldest tweet - 1 to keep track
        oldest_tweet = all_the_tweets[-1].id - 1

        # print('...%s tweets have been downloaded so far' % len(all_the_tweets))

    # transforming the tweets into a 2D array that will be used to populate the csv
    outtweets = [[tweet.created_at,
                  tweet.text.encode('utf-8')] for tweet in all_the_tweets]

    # writing to the csv file
    with open(screen_name + '_tweets.csv', 'w', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(['created_at', 'text'])
        writer.writerows(outtweets)
    print('All Tweets Downloaded!')

def language_analysis(text):
    client = language.LanguageServiceClient()
    document = types.Document(content=text, type=enums.Document.Type.PLAIN_TEXT)

    sent_analysis = client.analyze_sentiment(document=document)
    entity_sent_analysis = client.analyze_entity_sentiment(document=document)

    return sent_analysis, entity_sent_analysis

def print_result(annotations):
    score = annotations.document_sentiment.score
    magnitude = annotations.document_sentiment.magnitude

    for index, sentence in enumerate(annotations.sentences):
        sentence_sentiment = sentence.sentiment.score
        print('Sentence {} has a sentiment score of {}'.format(
            index, sentence_sentiment))

    print('Overall Sentiment: score of {} with magnitude of {}'.format(
        score, magnitude))
    return 0

## MAIN ##
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/Veraj/PycharmProjects/twitterSideProject/googleAPICredentials.json'
text = "There is one kind of food I cannot stand: sashimi. Sashimi is raw fish. Many people like it. They love it in Japan, but I cannot even look at without feeling sick. First of all, it looks like it has not been cooked and that makes me think it might not be clean."

# screen_name = input("Enter the twitter handle of the person whose tweets you want to download: ")
screen_name = 'realDonaldTrump'
# get_all_tweets(screen_name)
# with open(screen_name + '_tweets.csv') as csv_file:
#
# df = pd.read_csv(screen_name+'_tweets.csv')
# # print(df)
# sentiment, entities = language_analysis(text)
# print_result(sentiment)

#creating the dataframe
# df = quandl.get('WIKI/GOOGL')
# df.to_csv('googleStock.csv')

df = pd.read_csv('googleStock.csv')
df = df[['Date', 'Adj. Close']]
df.dropna(axis=0, inplace=True)
df.index = df.Date
df.drop('Date', axis=1, inplace=True)

train_size = 0.8
train = df[:int(len(df)*train_size)]
test = df[int(len(df)*train_size):]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])

x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train[0])
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))
print(x_train[0])
