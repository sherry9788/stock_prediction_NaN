import tweepy
import json
from tweepy import OAuthHandler

consumer_key = 'O2cRYY5H68dpd416mkAMfyBaH'
consumer_secret = 'AlyMBZtoZyjDluuP5LN464zMIdj3k28Ccmr6G5Nqkvu2p7RvCv'
access_token = '345850017-TjvWKWQEDZNxTkiu8JJ8yCvD4zAY7oAJW2v4biWA'
access_secret = 'w98rJiftZBPcaRbmZx7YKElnd8R2MjMKvgN8MqktXtfWX'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
# Crawl metatweet instead of texts
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
results = api.search(q="iphone x", count=100)

# for i in range(100):
#     print results["statuses"][i]

# text = []
# ts = []
# location = []
# favorites = []
# retweets = []
#
# for r in results:
#     print r.text
#     print r.created_at
#     print r.coordinates
#     print r.favorite
#     print r.retweeted
#     break
#     text.append(r.text)
    # ts.append(r.created_at)
    # location.append(r.coordinates)
    # favorites.append(r.favorite)
    # retweets.append(r.retweeted)

# data = {}
# data['text'] = text
# data['time'] = ts
# data['location'] = location
# data['favorites'] = favorites
# data['retweets'] = retweets
#
# with open('sample_tweets.json', 'w') as fp:
#     json.dump(data, fp, indent=4)
#
# with open('sample_tweets.json', 'r') as fp:
#     test = json.load(fp)

# Stores meta tweets into json
with open('sample_tweets.json', 'w') as fp:
    json.dump(results["statuses"], fp, indent=4)

with open('sample_tweets.json', 'r') as fp:
    test = json.load(fp)

print test
