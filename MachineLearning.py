__author__ = 'patrickjameswhite'
import pandas as pd
import os
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import MinMaxScaler
from csciML import read_mnist, compute_metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import time


use_dt = False
use_knn = True
use_lgr = True
use_svc = True

directory = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(directory+"/stocknews/preprocessed.csv")
#print(df.head(10))
#["date","sentence","compound","neg","neu","pos","open","high","low","close","volume","adj close"]
# ["compound","neg","neu","pos"]
features = ["compound","neg","neu","pos"]
scaler = MinMaxScaler()

#df[['diff']] = scaler.fit_transform(df[['diff']])


X = df.loc[:, features].as_matrix()


y = df.loc[:,"diff"].as_matrix()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.preprocessing import MinMaxScaler
print('Scaling the data...')
sc = MinMaxScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std  = sc.transform(X_test)

# print('Summary of scaled features')
# df = pd.DataFrame(X_train_std)
# print(df.describe())

# For pickling the results.
from sklearn.externals import joblib


if (use_knn):
    print('kNN ----------------------------------------------')
    print('Building the classifier...')
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier()
    knn.fit(X_train_std, y_train)

    print('Evaluating the classifier on the test set...')
    compute_metrics(knn, X_test_std, y_test)

    print('Saving the classifier...')
    joblib.dump(knn, directory+'/Kernels/knn.pkl')


if (use_dt):

    clf = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                max_features=1, max_leaf_nodes=64, min_samples_leaf=1,
                min_samples_split=1, min_weight_fraction_leaf=0.0,
                presort=False, random_state=42, splitter='best')


    clf.fit(X_train, y_train)
    #print(clf)

    y_pred = clf.predict(X_test)
    #print('\nMisclassified samples for decision tree: %d' % (y_test != y_pred).sum())

    accuracy = float(accuracy_score(y_test, y_pred))

    print(accuracy)


if (use_svc):

    from sklearn.svm import SVC

    svc = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape=None, degree=3, gamma=0.01, kernel='linear',
max_iter=-1, probability=False, random_state=None, shrinking=True,
tol=0.001, verbose=False)
    svc.fit(X_train_std, y_train)

    print('Evaluating the classifier on the test set...')
    compute_metrics(svc, X_test_std, y_test)

    print('Saving the classifier...')
    print("Classification report for classifier %s:\n"
      % (svc))
    joblib.dump(svc, directory+'/Kernels/svc.pkl')



if (use_lgr):
    l1 = time.time()
    print('Logistic regression ------------------------------')
    print('Building the classifier...')
    from sklearn.linear_model import LogisticRegression
    lgr = LogisticRegression()
    lgr.fit(X_train_std, y_train)

    print('Evaluating the classifier on the test set...')
    compute_metrics(lgr, X_test_std, y_test)


    print('Saving the classifier...')
    print("Classification report for classifier %s:\n"
      % (lgr))
    joblib.dump(lgr, directory+'/Kernels/lgr.pkl')

    e3 = time.time() - l1
    print("svc took %d secs"%e3)
