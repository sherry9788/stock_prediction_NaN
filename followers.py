
# get the list of followers from a politician's account
# designed to run on the Raspberry Pi to continue collecting data

import twitter
import json
import time
import os

ACCESS_TOKEN = '1087865425-ZUYUWgFvGvDsFeVzosFGO591uIQAx2cQSvwIW3F'
ACCESS_SECRET = 'pYB45mwqTo661tbTMQr9zuiUH6xlNKP042jOzx2Z3hrEH'
CONSUMER_KEY = 'gLny0EEJjicBnTVKQPXhmZ1aT'
CONSUMER_SECRET = 'AsqvFfosQf48sibHg2taP2jFEaoEIGGWr0uMoRo5WaLQcjeHra'

api = twitter.Api(consumer_key=CONSUMER_KEY,
                  consumer_secret=CONSUMER_SECRET,
                  access_token_key=ACCESS_TOKEN,
                  access_token_secret=ACCESS_SECRET)

target_account = 'tim_cook'

os.chdir('followers/' + target_account)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield [int(id_str) for id_str in l[i:i + n]]

finished = os.listdir('users/')
for file_name in os.listdir('ids/'):
    if 'info-' + file_name in finished:
        continue
    print 'Working on ' + file_name + '...'
    with open('ids/' + file_name, 'r') as infile:
        ids = list(chunks(json.load(infile), 100))
    user_info = list()
    for chunk in ids:
        attempt = 0
        while attempt <= 5:
            try:
                result = api.UsersLookup(chunk)
                user_info.append([user._json for user in result])
                break
            except:
                print 'Error at ' + file_name + ' ...'
                time.sleep(5)
                attempt += 1

    with open('users/info-' + file_name, 'w') as outfile:
        json.dump(user_info, outfile)  # note: each file is of ~11 MB
        print ' Done with ' + file_name + '...' 
