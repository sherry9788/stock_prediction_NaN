Team Memeber:
Pei Zhou, Xuening Wang, Hao Wang, Shuoyi Wei, Runyu Qian

The Submission zip is structured as following:
18_NaN
 - report.pdf
 - codes/
   - README.txt
   - run.sh
   - crawler.py
   - glove_cnn.py (please use python 3 to run this file)
   - logistic_reg.py (python 3)
   - random_forest.py (python 3)
   - preprocess.py (python 3)
 - data/
   - cnn.h5
   - 11.29.txt
   - 11.30.txt
   - ...
   - 12.10.txt
   - original data/
     - 2017-11-29.csv
     - 2017-11-30.csv
     - ...
     - 2017-12-10.csv
   - training/
     - negative/
       - 1
     - netural/
       - 2
     - positive/
       - 3
   - predict/
     - 1.txt
     - 2.txt
     - 3.txt
   - new_vectors/ (for random_forest and logistic_reg)
     - training/
        - 0.txt
        - 1.txt
        ...
        - 8.txt
     - training/
        - 0.txt
        - 1.txt
        - 2.txt

GloVe word embedding download url: http://nlp.stanford.edu/data/glove.6B.zip
Word2Vec word embedding download url: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit

Please install any of the following dependencies if you don't have them(did not put this into the run.sh because then it will force you to install packages that you might not want to install)
keras, gensim, nltk, preprocessor, numpy, sklearn, scipy
you can do this with simple pip commands like: pip3 install keras

Please restructure the folder format in the following structure in order to properly run the file after downloading the weights. Generally just move every folder one level under the data to just under the folder 18_NaN


18_NaN/
 - report.pdf
 - run.sh
 - glove_cnn.py
 - preprocess.py (will produce 0.csv, 1.csv, ... , 11.csv)
 - 11.29.txt
 - 11.30.txt
 - ...
 - 12.10.txt
 - cnn.h5         # move cnn.h5 to just under the main folder
 - embedding/     # rename the downloaded glove.6B folder (unzipped from glove.6B.zip) to embedding
   - glove.6B.50d.txt
   - glove.6B.100d.txt
   - glove.6B.200d.txt
   - glove.6B.300d.txt
 - training/      # move the training folder to just under 18_NaN
   - negative/
     - 1
   - netural/
     - 2
   - positive/
     - 3
 - predict/       # move the predict folder to just under 18_NaN
   - 1.txt
   - 2.txt
   - 3.txt
 - new_vectors/ (for random_forest and logistic_reg)
   - training/
      - 0.txt
      - 1.txt
      ...
      - 8.txt
   - training/
      - 0.txt
      - 1.txt
      - 2.txt

