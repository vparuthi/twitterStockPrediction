import csv

import numpy as np
import pandas as pd
import quandl as quandl
from keras.engine.saving import load_model

quandl.ApiConfig.api_key = "y94CFqx58gLCyaY9hsRf"
import tweepy
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import os
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

pd.options.mode.chained_assignment = None  # default='warn'
style.use('ggplot')


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
def create_df_csv(ticker):
    df = quandl.get('WIKI/'+ticker)
    df.to_csv(ticker + 'Stock.csv')
    df.dropna(axis=0, inplace=True)
    df['HL_PCT'] = (df["Adj. High"] - df["Adj. Low"]) / df["Adj. Low"] * 100
    df['PCT_Change'] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100
    if 'Date' in df.columns:
        df = df[['Date', 'Adj. Close', 'HL_PCT', 'PCT_Change']]
        df.index = df.Date
        df.drop('Date', axis=1, inplace=True)
    else:
        df = df[['Adj. Close', 'HL_PCT', 'PCT_Change']]
    return df

def read_df_csv(ticker):
    return pd.read_csv(ticker+'Stock.csv')

def create_model(X_train, y_train, number_col):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], number_col), activation='relu'))
    model.add(LSTM(units=50, activation='relu'))
    model.add(Dense(number_col))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=2)
    model.save("LSTM_model")
    return model

## MAIN ##
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/Veraj/PycharmProjects/twitterSideProject/googleAPICredentials.json'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

number_col = 3
train_size = 0.7
ticker = 'AAPL'
df = create_df_csv(ticker)
# df = read_df_csv(ticker)

train = df[:int(len(df)*train_size)]
test = df[int(len(df)*train_size):]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

X_train, y_train = [], []
for i in range(60,len(train)):
    X_train.append(scaled_data[i - 60:i])
    y_train.append(scaled_data[i])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], number_col))

# model = create_model(X_train, y_train, number_col)
model = load_model('LSTM_model')

inputs_df = df[len(df) - len(test) - 60:]
inputs = df[len(df) - len(test) - 60:].values
inputs = inputs.reshape(-1, number_col)
inputs = scaler.transform(inputs)

X_test, y_test = [], []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i])
    y_test.append(inputs[i])

X_test, y_test = np.array(X_test), np.array(y_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],number_col))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
val_loss, val_acc = model.evaluate(X_test,y_test)
print(val_loss, val_acc)

test['Predictions'] = closing_price[:,0]
plt.plot(train['Adj. Close'])
plt.plot(test[['Adj. Close', 'Predictions']])
print("done")
plt.show()
