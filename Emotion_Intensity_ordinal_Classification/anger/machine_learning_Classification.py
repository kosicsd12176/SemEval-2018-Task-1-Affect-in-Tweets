from wordcloud import WordCloud
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.get_backend()
import seaborn as sns
import pickle, re
import numpy as np
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


#import dataset as dataframe

data_train = pd.read_csv('/home/konstantinos/PycharmProjects/Affect/DIPLOMATIC/SemEval2018-Task1-all-data/SemEval2018-Task1-all-data/English/EI-oc/training/EI-oc-En-anger-train.txt', sep='\t', error_bad_lines=False)
data_train.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]
data_dev = pd.read_csv('/home/konstantinos/PycharmProjects/Affect/DIPLOMATIC/SemEval2018-Task1-all-data/SemEval2018-Task1-all-data/English/EI-oc/development/2018-EI-oc-En-anger-dev.txt', sep='\t', error_bad_lines=False)
data_dev.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]
data_test= pd.read_csv("/home/konstantinos/PycharmProjects/Affect/DIPLOMATIC/SemEval2018-Task1-all-data/SemEval2018-Task1-all-data/English/EI-oc/test-gold/2018-EI-oc-En-anger-test-gold.txt", sep='\t', error_bad_lines=False)
data_test.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]

frames = [data_train,data_dev]
data_train = pd.concat(frames)


data_train['Intensity_Class'] = pd.Categorical(data_train['Intensity_Class'])
data_train['Intensity_Class'] = data_train.Intensity_Class.cat.codes



data_test['Intensity_Class'] = pd.Categorical(data_test['Intensity_Class'])
data_test['Intensity_Class'] = data_test.Intensity_Class.cat.codes




class_count = data_train['Intensity_Class'].value_counts()

x = np.array(class_count.index)
y = np.array(class_count.values)
plt.figure(figsize=(8,4))
sns.barplot(x,y)
plt.xlabel('Valence')
plt.ylabel('Number of categories')
plt.show()


stop_words = set(stopwords.words('english'))
stop_words.update(["now","let", 'zero', 'one', 'two', 'three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

def show_wordcloud(data, title = None):
    wordcloud = WordCloud(
        background_color='black',
        stopwords=stop_words,
        max_words=200,
        max_font_size=40,
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
).generate(str(data))

    fig = plt.figure(1, figsize=(15, 15))
    plt.axis('off')
    if title:
        fig.suptitle(title, fontsize=20)
        #fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()


show_wordcloud(data_train['Tweet'],'Most Common Words from the whole corpus')
show_wordcloud(data_train[data_train['Intensity_Class'] == 0]['Tweet'],'no E can be inferred')
show_wordcloud(data_train[data_train['Intensity_Class'] == 1]['Tweet'],'low amount of E can be inferred')
show_wordcloud(data_train[data_train['Intensity_Class'] == 2]['Tweet'],'moderate amount of E can be inferred')
show_wordcloud(data_train[data_train['Intensity_Class'] == 3]['Tweet'],' high amount of E can be inferred')


#Removing Stop Words

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data_train['Tweet'] = data_train['Tweet'].apply(removeStopWords)
data_test['Tweet'] = data_test['Tweet'].apply(removeStopWords)



#Stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
from nltk.stem.snowball import SnowballStemmer

stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data_train['Tweet'] = data_train['Tweet'].apply(stemming)
data_test['Tweet'] = data_test['Tweet'].apply(stemming)



from nltk.tokenize import TweetTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
tokenizer = TweetTokenizer()

vectorizer = TfidfVectorizer(ngram_range=(1,1), tokenizer=tokenizer.tokenize)
full_text = list(data_train['Tweet'].values) + list(data_test['Tweet'].values)
vectorizer.fit(full_text)
train_vectorized = vectorizer.transform(data_train['Tweet'])
test_vectorized = vectorizer.transform(data_test['Tweet'])


x_train = train_vectorized
y_train = data_train['Intensity_Class']
x_test = test_vectorized
y_test = data_test["Intensity_Class"]


from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

lr = LogisticRegression()
ovr = OneVsRestClassifier(lr)
ovr.fit(x_train,y_train)

print(classification_report(ovr.predict(x_test) , y_test))
print(accuracy_score( ovr.predict(x_test) , y_test ))

svm = LinearSVC()
svm.fit(x_train,y_train)
print(classification_report( svm.predict(x_test) , y_test))
print(accuracy_score( svm.predict(x_test) , y_test ))

estimators = [ ('svm',svm) , ('ovr' , ovr) ]
clf = VotingClassifier(estimators , voting='hard')
clf.fit(x_train,y_train)

print(classification_report( clf.predict(x_test) , y_test))
print(accuracy_score( clf.predict(x_test) , y_test ))
