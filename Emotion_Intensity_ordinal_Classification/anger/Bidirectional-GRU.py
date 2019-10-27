from nltk.corpus import stopwords
import pandas as pd
import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


#import dataset as dataframe

train = pd.read_csv('/home/konstantinos/PycharmProjects/Affect/DIPLOMATIC/SemEval2018-Task1-all-data/SemEval2018-Task1-all-data/English/EI-oc/training/EI-oc-En-anger-train.txt', sep='\t', error_bad_lines=False)
train.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]
dev = pd.read_csv('/home/konstantinos/PycharmProjects/Affect/DIPLOMATIC/SemEval2018-Task1-all-data/SemEval2018-Task1-all-data/English/EI-oc/development/2018-EI-oc-En-anger-dev.txt', sep='\t', error_bad_lines=False)
dev.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]
test= pd.read_csv("/home/konstantinos/PycharmProjects/Affect/DIPLOMATIC/SemEval2018-Task1-all-data/SemEval2018-Task1-all-data/English/EI-oc/test-gold/2018-EI-oc-En-anger-test-gold.txt", sep='\t', error_bad_lines=False)
test.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]

train['Intensity_Class'] = pd.Categorical(train['Intensity_Class'])
train['Intensity_Class'] =train.Intensity_Class.cat.codes

dev['Intensity_Class'] = pd.Categorical(dev['Intensity_Class'])
dev['Intensity_Class'] = dev.Intensity_Class.cat.codes

test['Intensity_Class'] = pd.Categorical(test['Intensity_Class'])
test['Intensity_Class'] = test.Intensity_Class.cat.codes



stop_words = set(stopwords.words('english'))
stop_words.update(["now","let", 'zero', 'one', 'two', 'three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)



#Removing Stop Words

def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

train['Tweet'] = train['Tweet'].apply(removeStopWords)
test['Tweet'] = test['Tweet'].apply(removeStopWords)
dev['Tweet'] = dev['Tweet'].apply(removeStopWords)



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

train['Tweet'] = train['Tweet'].apply(stemming)
dev['Tweet'] = dev['Tweet'].apply(stemming)
test['Tweet'] = test['Tweet'].apply(stemming)





from keras.utils import to_categorical
target=train.Intensity_Class.values
y=to_categorical(target)

target=dev.Intensity_Class.values
c=to_categorical(target)



max_features = 13000
max_words = 50
batch_size = 128
epochs = 3
num_classes=4


X_train = train["Tweet"]
X_val = dev["Tweet"]
Y_train = y
Y_val = c

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,GRU,Embedding
from keras.layers import SpatialDropout1D,Bidirectional

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)




X_test = tokenizer.texts_to_sequences(test['Tweet'])
X_test =pad_sequences(X_test, maxlen=max_words)



X_train =pad_sequences(X_train, maxlen=max_words)
X_val = pad_sequences(X_val, maxlen=max_words)
X_test =pad_sequences(X_test, maxlen=max_words)


model4_BGRU = Sequential()
model4_BGRU.add(Embedding(max_features, 100, input_length=max_words))
model4_BGRU.add(SpatialDropout1D(0.25))
model4_BGRU.add(Bidirectional(GRU(64,dropout=0.4,return_sequences = True)))
model4_BGRU.add(Bidirectional(GRU(32,dropout=0.5,return_sequences = False)))
model4_BGRU.add(Dense(4, activation='sigmoid'))
model4_BGRU.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model4_BGRU.summary()

history4=model4_BGRU.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)