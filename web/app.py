
from flask import Flask, render_template, url_for, request

import os
import tensorflow as tf
import numpy as np
from transformers import TFBertForSequenceClassification, BertTokenizer
import re
from text_preprocessing import preprocess_sentence
import tweepy
import json

# start Flask
app = Flask(__name__)

# model setup
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,
                                       epsilon=1e-08)
model.compile(loss = loss, optimizer = optimizer, metrics = [metric])
model.load_weights('aggression_model_1epoch.h5')

# tokenizer steup
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#tweepy authentication
with open('twitter_credentials.json') as data_file:
    credentials = json.load(data_file)
auth = tweepy.OAuthHandler(credentials['api_key'], credentials['api_secret_key'])
auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])
api = tweepy.API(auth)


# render default webpage
@app.route("/")
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user = request.form['message']
        if '@' in user:
          user = user.split('@')[1]
        tweets = [tweet._json for tweet in tweepy.Cursor(
          api.user_timeline, 
          screen_name=user, 
          include_rts=False, 
          tweet_mode = "extended").items(100)] 
        tweet_list = [tweet['full_text'] for tweet in tweets]
        tweets_clean = list(map(preprocess_sentence, tweet_list))
        tweets_final = list(map(lambda x: x.split('http')[0] if 'http' in x else x, tweets_clean))
        input_ids = []
        attention_masks = []
        for tweet in tweets_final:
          bert_inp = bert_tokenizer.encode_plus(tweet,
                                          add_special_tokens = True,
                                          max_length = 100,
                                          truncation = True,
                                          padding = 'max_length',
                                          return_attention_mask = True)
    # Append every sentence vector to a list
          input_ids.append(bert_inp['input_ids'])
          attention_masks.append(bert_inp['attention_mask'])

# Convert the lists to arrays so that we can input them into the model
        input_ids = np.asarray(input_ids)
        attention_masks = np.array(attention_masks)
        preds = model.predict([input_ids, attention_masks], batch_size=32)
        pred_labels = [np.argmax(pred) for pred in preds[0]]
        aggressiveness = int(sum(pred_labels))

    return render_template('result.html', prediction = "Out of the last 100 tweets of this account, {} were aggressive".format(aggressiveness))




if __name__ == '__main__':
    app.run(debug=True)

