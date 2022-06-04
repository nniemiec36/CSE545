# Swanie, Nicole, Vasudha
# Topic modeling using BERTopic
# ran locally on python3 via terminal

from bertopic import BERTopic
import re
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import advertools as adv
import numpy as np
tweets = pd.read_csv("./msgs/msgs_bda_urban_lowanx_f.csv") # change csvs here

# need to remove "RT @...:"
# anything with "https://"
# also anything like @

tweets.message = tweets.apply(lambda row: re.sub(r"http\S+", "", row.message).lower(), 1)
tweets.message = tweets.apply(lambda row: re.sub(r"tweet+", "", row.message).lower(), 1)
tweets.message = tweets.apply(lambda row: re.sub(r"retweet+", "", row.message).lower(), 1)
tweets.message = tweets.apply(lambda row: re.sub(r"rt+", "", row.message).lower(), 1)
tweets.message = tweets.apply(lambda row: re.sub(r"RT+", "", row.message).lower(), 1)
tweets.message = tweets.apply(lambda row: " ".join(filter(lambda x:x[0]!="@", row.message.split())), 1)
tweets.message = tweets.apply(lambda row: " ".join(re.sub("[^a-zA-Z]+", " ", row.message).split()), 1)
# high_anxiety_urban = high_anxiety_urban.loc[(high_anxiety_urban.isRetweet == "f") & (high_anxiety_urban.text != ""), :]
# print(high_anxiety_urban_msg)
tweets = tweets.message.to_list()

stopwords = adv.stopwords['english'].union(adv.stopwords['spanish'])

### CREATING THE WORDCLOUD ###
def create_wordcloud(topic_model, topic):
    text = {word: value for word, value in topic}
    wc = WordCloud(background_color="white", stopwords=stopwords, max_words=1000)
    wc.generate_from_frequencies(text)
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

topic_model = BERTopic(language="multilingual")
topics, probs = topic_model.fit_transform(tweets)

topics = []
for i in range(0, 100):
    topics += topic_model.get_topic(i)

create_wordcloud(topic_model, topics)

#### TOPIC MODELING ####
# figure = topic_model.visualize_barchart()
# figure.show()


### PEARSON CORRELATION COEFF ###

# two variables ? anxiety scores and ...
# can use the probability scores from topic models
# but what else

# need two arrays of same size
# can use pandas to get the coeff ?
# x.corr(y) ?
# or we can use numpy
# my_rho = np.corrcoef(x_simple, y_simple)


