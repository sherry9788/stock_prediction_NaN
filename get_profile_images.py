
import json
import os
import urllib
import re
import random
import numpy
import time

account = 'tim_cook'
os.chdir('followers/' + account)

samples = random.sample(range(2768), 20)

start = time.time()
for users_file in numpy.array(os.listdir('users/'))[samples]:

    print 'Working on ' + users_file + '...'
    cursor = re.sub('info-|.txt', '', users_file)
    if os.path.isdir('images/' + 'images-' + cursor):
        continue
    
    with open('users/' + users_file, 'r') as infile:
        users = json.load(infile)
        users = [item for sublist in users for item in sublist]  # flatten chunks of user data

    os.mkdir('images/' + 'images-' + cursor)

    for user in users:
        try:
            image_url = re.sub('_normal', '', user['profile_image_url'])
            urllib.urlretrieve(image_url, 'images/' + 'images-' + cursor + '/' + image_url.split('/')[-1])
        except:
            print 'Error at ' + str(user['id']) + '...'
end = time.time()
print int(end - start)