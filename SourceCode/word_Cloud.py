from textblob import Word
from wordcloud import WordCloud
import re
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_csv(
    r"C:\Users\user\Downloads\sentiment_analylsis-main\Data\dataset.csv", encoding='latin-1')
dataset.tail()

dataset.Result.value_counts()

own_palette_1 = ["g", "#FFD700", "#FF6347", "#1E90FF"]
pd.Series(dataset["Result"]).value_counts().plot(kind="pie", colors=sns.color_palette(own_palette_1, 10),
                                                 labels=[
                                                     "Positive", "Neutral",  "Negative", "Positive"],
                                                 shadow=True, autopct='%.1f%%', fontsize=15, figsize=(6, 6))
plt.title("Percentage of tweets of each sentiment", fontsize=20)
plt.show()


dataset['tokenized_text'] = dataset['Tweets'].apply(word_tokenize)
dataset['tokenized_text'].head()


def avg_word(sentence):
    words = sentence.split()
    return (sum(len(word) for word in words)/len(words))


dataset['lemmatized_text'] = dataset['Tweets'].apply(
    lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
dataset['lemmatized_text'].tail()

dataset.head()


nltk.download()
stop = stopwords.words('english')

dataset['lemmatized_text'] = dataset['lemmatized_text'].apply(
    lambda x: " ".join(x for x in x.split() if x not in stop))
dataset['lemmatized_text'].head()

# Combining all text in tweets into an array for the WordCloud using .tolist()
text_array = dataset['lemmatized_text'].as_matrix().tolist()

strings = ''.join(text_array)


def clean_tweet(strings):
    tweet = re.sub('http\S+\s*', '', strings)  # remove URLs; http, https,... #
    return tweet


cloud = WordCloud(background_color="white", random_state=42, width=800, height=400, max_words=200,
                  prefer_horizontal=1.0, font_step=1,                        max_font_size=50,).generate(clean_tweet(strings))

# figsize adjusts resolution!! 200x100 is already pretty good!
plt.figure(figsize=(20, 10), facecolor='w')
plt.imshow(cloud)
plt.axis('off')
plt.show()


neutral_tweets = dataset.query("Result == 'Neutral'")
neutral_tweets.head()  # length: 1572

neutral_array = neutral_tweets['lemmatized_text'].as_matrix().tolist()

neutral_strings = ''.join(neutral_array)

cloud = WordCloud(background_color="seashell", random_state=42, width=800, height=400, max_words=150, colormap='Dark2', prefer_horizontal=1.0,
                  max_font_size=50,).generate(clean_tweet(neutral_strings))
# own_palette_1= ["g",
# figsize adjusts resolution!! 200x100 is already pretty good!
plt.figure(figsize=(20, 10), facecolor='w')
plt.imshow(cloud)
plt.axis('off')
plt.show()

positive_tweets = dataset.query("Result == 'Positive'")
positive_array = positive_tweets['lemmatized_text'].as_matrix().tolist()
positive_strings = ''.join(positive_array)


cloud = WordCloud(background_color="cornsilk", random_state=42, width=800, height=500, max_words=200, colormap='Dark2', color_func=None, prefer_horizontal=1.0,
                  max_font_size=50,).generate(clean_tweet(positive_strings))

# figsize adjusts resolution!! 200x100 is already pretty good!
plt.figure(figsize=(20, 10), facecolor='w')
plt.imshow(cloud)
plt.axis('off')
plt.show()

negative_tweets = dataset.query("Result == 'Negative'")
negative_array = negative_tweets['lemmatized_text'].as_matrix().tolist()
negative_strings = ''.join(negative_array)

cloud = WordCloud(background_color="black", random_state=42, width=800, height=500, max_words=200,  prefer_horizontal=1.0, colormap='Pastel1',
                  max_font_size=50,).generate(clean_tweet(negative_strings))

# figsize adjusts resolution!! 200x100 is already pretty good!
plt.figure(figsize=(20, 10), facecolor='w')
plt.imshow(cloud)
plt.axis('off')
plt.show()
