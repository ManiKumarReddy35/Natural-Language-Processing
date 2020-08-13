import warnings
import re
import numpy as np
import pandas as pd

from nltk import word_tokenize, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import *

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import scikitplot as skplt

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, classification_report
from string import punctuation as punct

stp_wrds = stopwords.words('english')
stemmer = PorterStemmer()

warnings.filterwarnings("ignore", category=DeprecationWarning)

def normalize(text):
    text = re.sub('[' + punct + ']', '', text).lower()
    words = word_tokenize(text)

    text = ' '.join([stemmer.stem(word) for word in words if word not in stp_wrds])
    return text


train = pd.read_csv('./stancedataset/train.csv', engine='python')
test = pd.read_csv('./stancedataset/test.csv', engine='python')
test = test[test['Target'] != 'Donald Trump']

train['Target'] = train['Target'].apply(lambda t: t.lower())
test['Target'] = test['Target'].apply(lambda t: t.lower())

# print(train['Tweet'].iloc[0])

d_train = [normalize(tweet) for tweet in train['Tweet'] + ' ' + train['Target']]
d_test = [normalize(tweet) for tweet in test['Tweet'] + ' ' + test['Target']]

# d_train = [normalize(tweet) for tweet in train['Tweet']]
# print(d_train[0])

vectorizer = TfidfVectorizer(ngram_range=(1, 3))

X_tr = vectorizer.fit_transform(d_train)
X_te = vectorizer.transform(d_test)

y_tr = np.asarray([st for st in train['Stance']])
y_te = np.asarray([st for st in test['Stance']])
# print(y_tr)

# params = {
#     'alpha': [10 ** a for a in range(-12, -0)]
# }

# sgd = SGDClassifier(penalty='elasticnet', l1_ratio=0.6)

# clf = GridSearchCV(sgd, params, cv=10)
# clf.fit(X_tr, y_tr)

# print(clf.best_params_)

clf = SGDClassifier(penalty='elasticnet', alpha = 1e-10, l1_ratio=0.6, loss='log')
clf.fit(X_tr, y_tr)

y_pred= clf.predict(X_te)
f_score = f1_score(y_te, y_pred, average='macro')

# print(f_score)

print(classification_report(y_te, y_pred))

#skplt.metrics.plot_roc(y_te, clf.predict_proba(
#    X_te), title='ROC Curve', plot_micro=False, plot_macro=False)
#
#skplt.metrics.plot_confusion_matrix(y_te, clf.predict(
#    X_te), normalize=True)



