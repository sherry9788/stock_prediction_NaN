import tweepy
import json
from tweepy import OAuthHandler

consumer_key = 'O2cRYY5H68dpd416mkAMfyBaH'
consumer_secret = 'AlyMBZtoZyjDluuP5LN464zMIdj3k28Ccmr6G5Nqkvu2p7RvCv'
access_token = '345850017-TjvWKWQEDZNxTkiu8JJ8yCvD4zAY7oAJW2v4biWA'
access_secret = 'w98rJiftZBPcaRbmZx7YKElnd8R2MjMKvgN8MqktXtfWX'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)
results = api.search(q='iphone x',lang='en', count=100)

text = []
for r in results:
    print r.text
    text.append(r.text)
data = {}
data['text'] = text
with open('sample_tweets.json', 'w') as fp:
    json.dump(data, fp, indent=4)
