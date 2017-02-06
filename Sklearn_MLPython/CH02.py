
# coding: utf-8

# ### Ch02. Supervised Learning

# In[20]:

# import
from sklearn.datasets import load_iris, fetch_olivetti_faces
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split, cross_val_score, KFold
#from sklearn.metrics import classification_report, confusion_matrix
from sklearn import metrics
from scipy.stats import sem


import sklearn as sk
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().magic('matplotlib inline')
sns.set()


# In[2]:

fo = fetch_olivetti_faces()
print(fo.DESCR)


# In[3]:

# metadata details abt faces
print(fo.keys())


# In[4]:

# checking out shapes
print(fo.images.shape, fo.data.shape, fo.target.shape)


# In[5]:

def print_faces(images, target, top_n):
    # set up the figure size in inches
    fig = plt.figure(figsize=(12, 12))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, 
    hspace=0.05, wspace=0.05)
    for i in range(top_n):
        # plot the images in a matrix of 20x20
        p = fig.add_subplot(20, 20, i + 1, xticks=[], 
        yticks=[])
        p.imshow(images[i], cmap=plt.cm.bone)
        
        # label the image with the target value
        p.text(0, 14, str(target[i]))
        p.text(0, 60, str(i))


# In[6]:

print_faces(fo.images, fo.target, 30)


# In[10]:

# instantiate
svm_linear = sk.svm.SVC(kernel='linear')

# splitting the data into train n test
X_train, X_test, y_train, y_test = train_test_split(fo.data, fo.target, test_size=0.25, random_state=33)
print(len(X_train), len(y_train))


# In[11]:

def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold croos validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score 
    # method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print(scores)
    print(("Mean score: {0:.3f} (+/-{1:.3f})").format(np.mean(scores), sem(scores)))


# In[12]:

evaluate_cross_validation(svm_linear, X_train, y_train, 5)


# In[18]:

def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    
    clf.fit(X_train, y_train)
    
    print( "Accuracy on training set:")
    print( clf.score(X_train, y_train))
    print( "Accuracy on testing set:")
    print( clf.score(X_test, y_test))
    
    y_pred = clf.predict(X_test)
    
    print( "Classification Report:")
    print( metrics.classification_report(y_test, y_pred))
    print( "Confusion Matrix:")
    print( metrics.confusion_matrix(y_test, y_pred))


# In[21]:

train_and_evaluate(svm_linear, X_train, X_test, y_train, y_test)


# In[ ]:



