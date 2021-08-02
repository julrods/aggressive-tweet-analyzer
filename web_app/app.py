### Import libraries
from flask import Flask, render_template, url_for, request
import os
import pandas as pd
import numpy as np
import json
import tweepy
from text_preprocessing import preprocess_sentence
import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer

### start Flask
app = Flask(__name__)

### Model setup
# Load pretrained model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Set loss, metric and optimizer
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,
                                       epsilon=1e-08)

# Compile model
model.compile(loss = loss, optimizer = optimizer, metrics = [metric])

# Load fine-tuned model weights
model.load_weights('aggression_model_1epoch.h5')

### Tokenizer steup
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

### Tweepy authentication
with open('twitter_credentials.json') as data_file:
    credentials = json.load(data_file)
auth = tweepy.OAuthHandler(credentials['api_key'], credentials['api_secret_key'])
auth.set_access_token(credentials['access_token'], credentials['access_token_secret'])
api = tweepy.API(auth)


### Render home webpage
@app.route("/")
def home():
    return render_template('home.html')

### Render the results page
@app.route('/result', methods=['POST'])
def result():
    ### Get the Twitter handle from the user input
    if request.method == 'POST':
        user = request.form['user_handle']
        
        # Remove @ if the handle is inputed with it
        if '@' in user:
          user = user.split('@')[1]
        
        ### Extract tweets
        tweet_dict = [{'tweet': tweet.full_text, # Full text
              'created_at': tweet.created_at, # Date
              'username': user, # Username
              'headshot_url': tweet.user.profile_image_url, # Profile image URL
              'url': f'https://twitter.com/user/status/{tweet.id}' # Tweet URL
               } for tweet in tweepy.Cursor(api.user_timeline,
                                            screen_name = user,
                                            exclude_replies = False, # Include replies
                                            include_rts = False, # Exclude retweets
                                            tweet_mode = "extended" # Include full tweet text
                                            ).items(100)] # Extract 100 items
        ### Save only the tweet texts in a list
        tweet_text_list = [tweet['tweet'] for tweet in tweet_dict]
        
        ### Preprocess tweets
        tweets_clean = list(map(preprocess_sentence, tweet_text_list))
        
        ### Tokenize tweets
        # Create empty lists to append the tweet vectors to
        input_ids = []
        attention_masks = []

        # Loop through the tweets texts to tokenize each tweet
        for tweet in tweets_clean:
          bert_inp = bert_tokenizer.encode_plus(tweet,
                                          add_special_tokens = True,
                                          max_length = 100,
                                          truncation = True,
                                          padding = 'max_length',
                                          return_attention_mask = True)
          # Append every tweet vector to a list
          input_ids.append(bert_inp['input_ids'])
          attention_masks.append(bert_inp['attention_mask'])

        # Convert the lists to arrays so that we can input them into the model
        input_ids = np.asarray(input_ids)
        attention_masks = np.array(attention_masks)

        ### Make predictions 
        # Predict the label (0 = not aggressive / 1 = aggressive)
        preds = model.predict([input_ids, attention_masks], batch_size=32)

        # Find the predicted label
        pred_labels = [np.argmax(pred) for pred in preds[0]]
        
        # Create a new variable 'label' for each tweet in tweet_dict
        for tweet, pred in zip(tweet_dict, pred_labels):
            tweet['label'] = pred

        ### Define output variables
        # Filter to keep the aggressive tweets only
        aggressive_tweets = [tweet for tweet in tweet_dict if tweet['label'] == 1]
        
        # Define the aggressiveness score
        aggressiveness = len(aggressive_tweets)

    return render_template('result.html', aggressiveness = aggressiveness, tweets = aggressive_tweets)

if __name__ == '__main__':
    # For development:
    app.run(debug=True)

    # For production:
    #from waitress import serve
    #serve(app, host="0.0.0.0", port=8080) #open http://localhost:8080/

