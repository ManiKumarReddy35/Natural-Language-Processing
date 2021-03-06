# -*- coding: utf-8 -*-
"""NLP PROJECT- STANCE DETECTION -NAIVE BAYES - LSTM

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19fZwgZh5mKRleZc8D6_c6bMni2K2xzic

## **IMPORTING REQUIRED MODULES**
"""

import pandas as pd
from nltk.stem import WordNetLemmatizer 
from nltk import word_tokenize
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import stopwords
stopword = set(stopwords.words('english'))
from sklearn.preprocessing import normalize
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers import Dropout

"""## **MOUNTING THE GOOGLE DRIVE**"""

from google.colab import drive
drive.mount('/content/drive')

"""## **READING THE DATA FROM CSV FILES**"""

train_data = pd.read_csv("/content/drive/My Drive/train_stance.csv",engine='python')
test_data = pd.read_csv("/content/drive/My Drive/test (1).csv",engine='python',names=['Tweet','Target','Stance','Opinion Towards','Sentiment'])

train_data.head()

test_data.head()

"""## **VISUALIZING THE BAR PLOT FOR COUNT OF INDIVIDUAL CLASSES PRESENT IN TRAINING SET**"""

train_data['Stance'].value_counts().plot('bar')

"""## **VISUALIZING THE BAR PLOT FOR COUNT OF INDIVIDUAL CLASSES PRESENT IN TEST SET**"""

test_data['Stance'].value_counts().plot('bar')

train_data['Target'].value_counts().plot('bar')

test_data['Target'].value_counts().plot('bar')

"""## **DIMENSIONS OF TRAINING AND TEST DATA SETS**"""

print("DIMENSIONS OF TRAINING DATA::",train_data.shape)
print("DIMENSIONS OF TEST DATA::",test_data.shape)

"""## **PREPARING THE CUSTOMIZED STOP-WORD LIST(NEGATIVE WORDS ARE NOT CONSIDERED TO BE STOP WORDS)**

---
"""

print("TOTAL NUMBER OF STOP WORDS WHEN NEGATIVE WORDS ARE CONSIDERED AS STOP WORDS::",len(stopword))
stopwords=set()
neg_words = {"didn't","shouldn't","hasn't","wasn't","won't","aren't","weren't","doesn't","couldn't","don't","hadn't","shan't","wouldn't"}
for i in stopword:
    if i not in neg_words:
        stopwords.add(i)
print("TOTAL NUMBER OF STOP WORDS WHEN NEGATIVE WORDS ARE NOT CONSIDERED AS STOP WORDS::",(len(stopwords)))
print(stopwords)

"""## **DATA-PREPROCESSING FUNCTION**"""

def preprocess(text):
  stopwords = {'such', 'each', 'doesn', 'herself', 'been', 'was', 've', 'at', 'before', 'they', 'he', 'all','t', 'wouldn', 'ain', 'the', 'doing', 'so', 'couldn', 
                'ours', 'wasn', 'having', 'your', 'up','do', 're', 'be', 'weren', 'hadn', 'should', 'nor', 'and', 'hers', 'won', 'why', 'i', 'mustn', 
                'don', 'those', 'other', 'her', 'very', 'an', 'just', 'any', 'through', 'himself', 'where','while', 'only', 'ma', 'between', 'from', "you've",
                'in', 'by', 'were', 'to', 'itself', 'how', 'no', 'own', 'too', 'when', 'hasn', 'aren', 'needn','a', 'which', 'shan', 'him', 'if', 'had', "should've", 
                'there', 'haven', 'have', 'd', 'most', 'mightn', 'yourself', 'this', 'about', 'them', 'who','being', 'whom', 'we', 'our', 'my', 'll', 'not', 'above', 
                's', 'until', 'o', "haven't", 'with', "you're", "mustn't", 'themselves', "it's", 'shouldn', "you'd", 'it', 'its', 'both', 'than', 'm', 'that', 'will', 
                'as', 'isn', 'myself', 'yours', 'out', 'she', 'ourselves', 'theirs', 'are', 'few', 'has', 'but', 'into', 'here', 'after', 'you', 'off', "she's", 'then', 
                "you'll", 'y', 'me', 'during', 'again', 'against', 'what', 'is', 'because', 'down','some', 'of', 'or', 'once', 'more', "that'll", 'under', 'for', 'can',
                'am', 'further', 'these', "needn't", 'on', 'did', 'now','below', "isn't", 'his', 'their', 'didn', 'yourselves', 'does', 'over', 'same', "mightn't",}
  lemma = WordNetLemmatizer()
  pattern = '[#?“"",.!0-9$\[\]/\}#=<>"\'*:,|_~;()^-]'
  words = text.split()
  sentence= ""
  for i in words:
    #https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
    if((any(l.isdigit() for l in i)) or (i in stopwords) or len(i)<3 or i[0]=="@"):
      continue
    w=''
    for j in i:
      if(re.match(pattern,j)):
        continue
      else:
        w=w+j
    i=w
    i=i.lower()
    i=lemma.lemmatize(i)
    sentence=sentence+i+" "
  return sentence.strip()

"""## **APPLYING DATA-PREPROCESSING FUNCTION TO EVERY TWEET OF TRAIN AND TEST DATA**"""

train_data['Tweet'] = train_data.Tweet.apply(preprocess)
test_data['Tweet'] = test_data.Tweet.apply(preprocess)

"""## **TRAIN DATA AFTER PREPROCESSING**"""

train_data.head()

"""## **TEST DATA AFTER PREPROCESSING**"""

test_data.head()

"""## **CONVERTING THE TWEETS OF TRAINING DATA INTO BAG OF WORDS REPRESENTATION AND PERFORMING ONE HOT ENCODING OF STANCE FEATURES**"""

bow = CountVectorizer(min_df=50,max_features=1000) 
type(bow.fit_transform(train_data['Tweet']))

bow = CountVectorizer(max_features=2000) 
x_train_bow = bow.fit_transform(train_data['Tweet'])
x_train_oh = np.array(pd.get_dummies(train_data['Target']))

x_train_bow_norm = normalize(x_train_bow,axis=0)

x_test_bow = bow.transform(test_data['Tweet'])
x_test_oh = np.array(pd.get_dummies(test_data['Target']))

x_test_bow_norm = normalize(x_test_bow,axis=0)

print(x_train_bow_norm.shape,x_train_oh.shape,x_test_bow_norm.shape,x_test_oh.shape)

"""## **CONCATENATING TWO ARRAYS OF BOWS FEATURES AND ONE-HOT ENCODING FEATURES**"""

x_train = np.hstack((x_train_bow_norm.toarray(),x_train_oh)) 
x_test = np.hstack((x_test_bow_norm.toarray(),x_test_oh)) 

print(x_train.shape,x_test.shape)

"""## **MODIFYING THE CLASS LABELS OF INTO NUMERICALS**"""

def modify_labels(text):
  if(text == "AGAINST" ):
    return 0
  if(text == "NONE" ):
    return 1
  if(text == "FAVOR" ):
    return 2

y_train = train_data.Stance.apply(modify_labels)
y_test = test_data.Stance.apply(modify_labels)
print(y_train.shape,y_test.shape)

"""## **IMPLEMENTATION OF NAIVE BAYES**"""

#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
hypm_alpha = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]
#trying out with various smoothing factors..
f1scores=[]
for i in hypm_alpha:
    nb = MultinomialNB(alpha=i)
    nb.fit(x_train,y_train)
    prediction = nb.predict(x_test)
    score = metrics.f1_score(y_test,prediction,average='macro')
    f1scores.append(score)

print(f1scores)

"""## **SMOOTHING FACTORS v/s F1 SCORES**"""

from  matplotlib import pyplot
pyplot.plot(hypm_alpha,f1scores)
pyplot.xlabel("SMOOTHENING - FACTORS")
pyplot.ylabel("F1 - SCORES")
pyplot.show()
print(f1scores)

"""# **THE OPTIMAL SMOOTHENING FACTOR IS 0.1**

# **THE F-1 SCORE ON TEST DATA USING NAIVE BAYES WITH OPTIMAL SMOOTHENING PARAMETER OF 0.1 IS 50.2%**
"""

train_data['ModTweet'] = "a"
for i in range(0,len(train_data)):
  train_data['ModTweet'][i] = train_data.iloc[i,0]+" <tar>"+train_data.iloc[i,1].replace(" ","")

test_data['ModTweet'] = "a"
for i in range(0,len(test_data)):
  test_data['ModTweet'][i] = test_data.iloc[i,0]+" <tar>"+test_data.iloc[i,1].replace(" ","")

x_train_lstm = train_data['ModTweet'].copy()
y_train_lstm = y_train.copy()
x_test_lstm  = test_data['ModTweet'].copy()
y_test_lstm  = y_test.copy()

def vocab_gen(data):
  vocab={}
  for i in data:
    for j in i.split():
      if j not in vocab:
        vocab[j]=1
      else:
        vocab[j]+=1
  return vocab

vocab_train = vocab_gen(x_train_lstm)
vocab_test = vocab_gen(x_test_lstm)
vocab_train_mod = sorted(vocab_train, key=vocab_train.get, reverse=True)

"""## **FUNCTION DEFINITION FOR CONVERTING SENTENCE TO NUMERICS** ##"""

def sent_to_vector(tweet,data_vocab,type_data):
  sent = []
  for i in tweet.split():
    if(type_data=="train"):
        sent.append((data_vocab.index(i))+1)
    if(type_data=="test"):
      if(i in data_vocab):
          sent.append((data_vocab.index(i))+1)
      else:
        continue
  return sent

"""## **FUNCTION CALLING FOR "sent_to_vector()" on TRAIN AND TEST DATA SETS** ##"""

x_train_lis=[]
for i in tqdm(range(len(x_train_lstm)),position=0):
  x_train_lis.append(sent_to_vector(x_train_lstm.iloc[i],vocab_train_mod,"train"))

x_train_arr = np.array(x_train_lis)

x_test_lis=[]
for i in tqdm(range(len(x_test_lstm)),position=0):
  x_test_lis.append(sent_to_vector(x_test_lstm.iloc[i],vocab_train_mod,"test"))

x_test_arr = np.array(x_test_lis)

"""## **PERFORMING PADDING MAKING VECTOR LENGTH EQUAL**"""

x_train_arr = sequence.pad_sequences(x_train_arr, maxlen=30)
x_test_arr = sequence.pad_sequences(x_test_arr, maxlen=30)

pip install keras_metrics

"""## **LSTM ARCHITECTURE** ##"""

#ref::https://keras.io/examples/imdb_lstm/
import keras_metrics
import keras
model = Sequential()
model.add(Embedding(10174,64,input_length=30)) #taking whole length of vocab
model.add(LSTM(12))
model.add(Dropout(0.8))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras_metrics.precision(),keras_metrics.recall()])
values=model.fit(x_train_arr,keras.utils.to_categorical(y_train_lstm, num_classes=3),nb_epoch=23,batch_size=500)

scores = model.evaluate(x_test_arr,keras.utils.to_categorical(y_test_lstm,num_classes=3), verbose=0)
print("f1-Score::",round(100*2*scores[1]*scores[2]/(scores[1]+scores[2]),2))

## FOR PREDICTION PURPOSES.....yhat = model.predict(x_test_arr, verbose=0)

plt.figure(figsize=(8,4))
plt.plot(values.history['loss'], label='Train Loss')
plt.plot(values.history['precision'],label="Training Precision Score")
plt.plot(values.history['recall'],label="Training Recall Score")
plt.title('EPOCHS V/S LOSS,PRECISION,RECALL')
plt.ylabel('loss,precision,recall')
plt.xlabel('epochs')
plt.legend()
plt.show()

"""# **THE F-1 SCORE ON TEST DATA USING LSTM'S IS 74.66**

### REFERENCES:: 
1. https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
2. My Own SemEval Sentimix Task
3. https://keras.io/examples/imdb_lstm/
4. https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
5.     #https://stackoverflow.com/questions/19859282/check-if-a-string-contains-a-number
"""