from nltk.corpus import stopwords
import pandas as pd
import re
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


#import dataset as dataframe

train = pd.read_csv('~/2018-Valence-oc-En-train.txt', sep='\t', error_bad_lines=False)
train.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]
dev = pd.read_csv('~/2018-Valence-oc-En-dev .txt', sep='\t', error_bad_lines=False)
dev.columns = ["ID", "Tweet", "Affect_Dimension", "Intensity_Class"]
test= pd.read_csv("~/2018-Valence-oc-En-test-gold.txt", sep='\t', error_bad_lines=False)
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
num_classes=7


X_train = train["Tweet"]
X_val = dev["Tweet"]
Y_train = y
Y_val = c

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM,Embedding
from keras.optimizers import Adam
from keras.layers import SpatialDropout1D,Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard, Callback, EarlyStopping

tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train))
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)




X_test = tokenizer.texts_to_sequences(test['Tweet'])
X_test =pad_sequences(X_test, maxlen=max_words)



X_train =pad_sequences(X_train, maxlen=max_words)
X_val = pad_sequences(X_val, maxlen=max_words)
X_test =pad_sequences(X_test, maxlen=max_words)



model3_LSTM=Sequential()
model3_LSTM.add(Embedding(max_features,100,mask_zero=True))
model3_LSTM.add(LSTM(64,dropout=0.4,return_sequences=True))
model3_LSTM.add(LSTM(32,dropout=0.5,return_sequences=False))
model3_LSTM.add(Dense(num_classes,activation='sigmoid'))
model3_LSTM.compile(loss='binary_crossentropy',optimizer=Adam(lr = 0.001),metrics=['accuracy'])
model3_LSTM.summary()

history3=model3_LSTM.fit(X_train, Y_train, validation_data=(X_val, Y_val),epochs=epochs, batch_size=batch_size, verbose=1)

y_pred3=model3_LSTM.predict_classes(X_test, verbose=1)

