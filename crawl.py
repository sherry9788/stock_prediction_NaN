import tweepy
from tweepy import OAuthHandler
 
consumer_key = 'O2cRYY5H68dpd416mkAMfyBaH'
consumer_secret = '	AlyMBZtoZyjDluuP5LN464zMIdj3k28Ccmr6G5Nqkvu2p7RvCv'
access_token = '345850017-TjvWKWQEDZNxTkiu8JJ8yCvD4zAY7oAJW2v4biWA'
access_secret = 'w98rJiftZBPcaRbmZx7YKElnd8R2MjMKvgN8MqktXtfWX'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)



