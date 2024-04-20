from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score


# Set Random seed
np.random.seed(500)


# Add the Data using pandas
dataset = pd.read_csv(
    r"C:\Users\user\Downloads\sentiment_analylsis-main\Data\dataset.csv", encoding='latin-1')

dataset['Tweets'].dropna(inplace=True)


# Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
dataset['Tweets'] = [entry.lower() for entry in dataset['Tweets']]

nltk.download('punkt')
dataset['Tweets'] = [word_tokenize(entry) for entry in dataset['Tweets']]


nltk.download('wordnet')
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
for index, entry in enumerate(dataset['Tweets']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    dataset.loc[index, 'text_final'] = str(Final_words)


Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(
    dataset['text_final'], dataset['Result'], test_size=0.3)

Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Test_Y = Encoder.fit_transform(Test_Y)


# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the dataset
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(dataset['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)


# Classifier - Algorithm - Multinomial Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score -> ",
      accuracy_score(predictions_NB, Test_Y)*100)


# Support Vector Machine
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y)*100)

################ Check this Algo if it works#########################
# Randome Forest
RF = RandomForestClassifier(
    n_estimators=10, criterion='entropy', random_state=0)
RF.fit(Train_X_Tfidf, Train_Y)
predictions_RF = RF.predict(Test_X_Tfidf)
print(" RF-> ", accuracy_score(predictions_RF, Test_Y)*100)
#########################################


# K-Neighbors Classifier
knn = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knn.fit(Train_X_Tfidf, Train_Y)
predictions_knn = knn.predict(Test_X_Tfidf)
print("KNN-> ", accuracy_score(predictions_knn, Test_Y)*100)

# -------------------
"""

Naive Bayes Accuracy Score ->  67.56282875511397


SVM (linear) Accuracy Score ->  74.81005260081824



RF->  70.36820572764465

 
 
KNN->  56.925774400935126
"""
