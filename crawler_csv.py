import tweepy
# import json
import csv
from tweepy import OAuthHandler

end_date = "2017-12-06" # crawled tweets will be  ONE DAY BEFORE the end_date

consumer_key = 'O2cRYY5H68dpd416mkAMfyBaH'
consumer_secret = 'AlyMBZtoZyjDluuP5LN464zMIdj3k28Ccmr6G5Nqkvu2p7RvCv'
access_token = '345850017-TjvWKWQEDZNxTkiu8JJ8yCvD4zAY7oAJW2v4biWA'
access_secret = 'w98rJiftZBPcaRbmZx7YKElnd8R2MjMKvgN8MqktXtfWX'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)
results = api.search(q='iphone x', count=100, lang='en', until=end_date)

mydata = []
for r in results:
    content = r.text.encode('utf-8')
    time = r.created_at.strftime("%Y-%m-%d %H:%M:%S")
    # print content
    # print time
    # print r.user.followers_count
    # print r.favorite_count
    mydata.append((content, time, r.user.followers_count, r.favorite_count))
print mydata
with open(end_date+'.csv', 'wb') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',', quotechar='\"', quoting=csv.QUOTE_MINIMAL, escapechar='\\')
    spamwriter.writerows(mydata)