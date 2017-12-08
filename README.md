# stock_prediction_NaN
### CS 145

Midterm report shall be found here: https://www.overleaf.com/11956684mvynxdzcyppq
Midterm report feedback from Justin Wood: "The approach looks good. I like that you are using different classifiers. Suggestions: Specify how many tweets are manually labelled. Ideally the more the better but of course we must be practical. The specific aim is not specified directly. Are you targeting specific companies, or are you trying to find a major mover in the market? If it’s the latter then how are you going to do that? I think some preliminary results may help focus the problem better. Overall the report is very general and it would help to get a more focused view on how you are trying to predict stock prices. Please make the run.sh a run.bat and compatible with Python 2.7.13 -- Anaconda 4.3.0 (64-bit). Also make the run.bat install all required libraries so that running run.bat will install necessary files and then run the program without the user needing to do anything."


1.  	Title of your project, and group information (group # and name, group member names)

2.  	Abstract

3.  	Introduction of the overall goal and background

Which problem you want to solve

4.  	Problem definition and formalization

How to break the problem into subtasks and formalize them into data mining problems
- Short term
- Industry: (Techonology)
- Steps:
  * Crawl data with tweepy
  * Manually label the sentiment of tweets
  * Word2vec with labels of sentiments
  * Train with model

5.  	Data preparation description and preprocessing

What’s your strategy in crawling Twitter data and describe what you plan to get 

6.  	Methods description (detailed steps)
- Also crawl the stock price and use it to train the related tweets for a certain time period

7.  	Experiments design and Evaluation

8.  	Schedule: time line of your project 

9.  	Progress discussion

10.  	References
