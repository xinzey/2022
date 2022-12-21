#!/usr/bin/env python
# coding: utf-8

# In[1]:


import folium
import warnings

get_ipython().run_line_magic('matplotlib', 'inline')
import datetime as dt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn import svm
from numpy import mean
from numpy import std

from sklearn import datasets
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error,plot_roc_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc,precision_recall_curve,plot_confusion_matrix


from random import seed
from random import randrange

from mlxtend.feature_selection import SequentialFeatureSelector

warnings.filterwarnings("ignore")


# In[2]:


data = pd.read_csv('airbnb_newyork_cleaned.csv')


# In[3]:


data.describe()


# In[ ]:


data


# In[4]:


# get the first 200 crimes in the cdata
limit = 700
data_map = data.iloc[0:limit, :]

# Instantiate a feature group for the incidents in the dataframe
locations = folium.map.FeatureGroup()

# Loop through the 200 crimes and add each to the incidents feature group
for lat, lng, in zip(data_map.latitude, data_map.longitude):
    locations.add_child(
        folium.CircleMarker(
            [lat, lng],
            radius=7, # define how big you want the circle markers to be
            color='white',
            fill=True,
            fill_color='blue',
            fill_opacity=0.4
        )
    )

# Add incidents to map
san_map = folium.Map(location=[40.7128, -74.0060], zoom_start=10)
san_map.add_child(locations)


# In[5]:


print(data.columns)


# In[6]:


# Create columns of percentage

data['availability_30_Percent'] = data['availability_30'].div(30)
data['availability_60_Percent'] = data['availability_60'].div(60)
data['availability_90_Percent'] = data['availability_90'].div(90)
data['availability_365_Percent'] = data['availability_365'].div(365)

# data['availability_30_D'] = \
#             pd.cut(data['availability_30_Percent'], [-0.01, 0.33, 0.66, 1.01], labels = ['Low', 'Medium', 'High']) 
# data['availability_60_D'] = \
#             pd.cut(data['availability_60_Percent'], [-0.01, 0.33, 0.66, 1.01], labels = ['Low', 'Medium', 'High']) 
# data['availability_90_D'] = \
#             pd.cut(data['availability_90_Percent'], [-0.01, 0.33, 0.66, 1.01], labels = ['Low', 'Medium', 'High']) 
# data['availability_365_D'] = \
#             pd.cut(data['availability_365_Percent'], [-0.01, 0.33, 0.66, 1.01], labels = ['Low', 'Medium', 'High']) 


# In[7]:


data['availability_30_Percent'].describe(percentiles = [0.33, 0.66])


# In[8]:


plt.hist(data['availability_30_Percent'])
plt.title('Availability 30 Days')
plt.xlabel('Availability Percentage')
plt.ylabel('Frequency')


# In[9]:


data['availability_60_Percent'].describe(percentiles = [0.33, 0.66])


# In[10]:


plt.hist(data['availability_60_Percent'])
plt.title('Availability 60 Days')
plt.xlabel('Availability Percentage')
plt.ylabel('Frequency')


# In[11]:


data['availability_90_Percent'].describe(percentiles = [0.33, 0.66])


# In[12]:


plt.hist(data['availability_90_Percent'])
plt.title('Availability 90 Days')
plt.xlabel('Availability Percentage')
plt.ylabel('Frequency')


# In[13]:


data['availability_365_Percent'].describe(percentiles = [0.33, 0.66])


# In[14]:


plt.hist(data['availability_365_Percent'])
plt.title('Availability 365 Days')
plt.xlabel('Availability Percentage')
plt.ylabel('Frequency')


# In[15]:


#Classify the y into Low and High 

data['availability_30_D'] =             pd.cut(data['availability_30_Percent'], [-0.01, 0.31, 1.01], labels = ['Low', 'High']) 
data['availability_60_D'] =             pd.cut(data['availability_60_Percent'], [-0.01, 0.53, 1.01], labels = ['Low', 'High']) 
data['availability_90_D'] =             pd.cut(data['availability_90_Percent'], [-0.01, 0.8, 1.01], labels = ['Low', 'High']) 
data['availability_365_D'] =             pd.cut(data['availability_365_Percent'], [-0.01, 0.63, 1.01], labels = ['Low', 'High']) 


# In[16]:


# Choose the independent variables

X_columns = [
        
        'number_of_reviews', 'review_scores_rating', 'reviews_per_month', \
        'instant_bookable', 'calculated_host_listings_count',  'price',\
        '24-hour check-in', 'Air conditioning', 'Breakfast', 'Cable TV', \
        'Carbon monoxide detector', 'Cat', 'Dog', 'Doorman', 'Doorman Entry', \
        'Dryer', 'Elevator', 'Essentials', 'Eco friendly', 'Fire extinguisher', \
        'First aid kit', 'Free parking on premises', \
        'Paid street parking off premises', 'Gym', 'Hair dryer', 'Hangers', \
        'Heating', 'Hot tub', 'Indoor fireplace', 'Internet', 'Iron', 'Keypad', \
        'Kitchen', 'Laptop friendly workspace', 'Lock on bedroom door', \
        'Lockbox', 'Other pet(s)', 'Paid parking off premises', 'Pool', \
        'Private entrance', 'Shampoo', 'Smart lock', 'Smoke alarm', \
        'Smoking allowed', 'TV', 'Washer', 'Washer / Dryer'
]

y_columns = [
        'availability_30_D', 'availability_60_D', 
        'availability_90_D','availability_365_D'
]


# In[39]:


X_columns


# In[17]:


# Train test split
X = data[X_columns]
y30 = data[y_columns[0]]
y60 = data[y_columns[1]]
y90 = data[y_columns[2]]
y365 = data[y_columns[3]]
X30_train, X30_test, y30_train, y30_test = train_test_split(X, y30, train_size=0.7, random_state=2)
X60_train, X60_test, y60_train, y60_test = train_test_split(X, y60, train_size=0.7, random_state=2)
X90_train, X90_test, y90_train, y90_test = train_test_split(X, y90, train_size=0.7, random_state=2)
X365_train, X365_test, y365_train, y365_test = train_test_split(X, y365, train_size=0.7, random_state=2)


# In[38]:


X


# In[18]:


lr = LogisticRegression(multi_class='multinomial',solver='lbfgs',class_weight='balanced',max_iter=10000)
#For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs' handle multinomial loss
result = lr.fit(X90_train,y90_train) ## fit the model
y90_pred = lr.predict(X90_test)

count = 0
y90_test = list(y90_test)
for index in range(len(y90_pred)):
    if y90_pred[index] == y90_test[index]:
        count += 1
        
accuracy = count / len(y90_pred)
print("The accuracy of Logistic Regression with all features is:", accuracy)   
print(classification_report(y90_test, y90_pred, digits=3))


# In[19]:


lrs = LogisticRegression(multi_class='multinomial',solver='lbfgs',class_weight='balanced',max_iter=10000)

ffs = SequentialFeatureSelector(lrs, k_features = 10, forward = True, n_jobs = -1)
# ffs = SequentialFeatureSelector(lr)
ffs.fit(X90_train,y90_train)

featureslr = list(ffs.k_feature_names_)
print(featureslr)

#features = list(map(int, features))
lrs.fit(X90_train[featureslr], y90_train)
y90_pred = lrs.predict(X90_test[featureslr])

count = 0
y90_test = list(y90_test)
for index in range(len(y90_pred)):
    if y90_pred[index] == y90_test[index]:
        count += 1
        
accuracy = count / len(y90_pred)
print("The accuracy of Logistic Regression with 10 features is:", accuracy)  


# In[20]:


lda = LDA()
model_lda = lda.fit(X90_train, y90_train)
pred=model_lda.predict(X90_test)

# print(np.unique(pred, return_counts=True))
print(confusion_matrix(pred, y90_test))
print(classification_report(y90_test, pred, digits=3))


# In[21]:


qda = QDA()
model_qda = qda.fit(X90_train, y90_train)
pred2=model_qda.predict(X90_test)
print(confusion_matrix(pred2, y90_test))
print(classification_report(y90_test, pred2, digits=3))


# In[22]:


rfc=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)
model_rfc = rfc.fit(X90_train, y90_train)
y90_pred = model_rfc.predict(X90_test)
print(confusion_matrix(y90_pred, y90_test))
print(classification_report(y90_test, y90_pred, digits=3))
# RF_mse = mean_squared_error(y90_test, model_rfc.predict(X90_test))
# print(RF_mse)


# In[23]:


plot_confusion_matrix(model_rfc, X90_test, y90_test, cmap='Blues')


# In[24]:


print(model_rfc.feature_importances_)


# In[25]:


rfcs=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)
model_rfcs = rfcs.fit(X90_train, y90_train)

ffs = SequentialFeatureSelector(model_rfcs, k_features = 10, n_jobs = -1)
ffs.fit(X90_train,y90_train)

featuresrf = list(ffs.k_feature_names_)
print(featuresrf)


# In[26]:


model_90 = rfcs.fit(X90_train[featuresrf], y90_train)
y90_pred = model_90.predict(X90_test[featuresrf])
print(confusion_matrix(y90_pred, y90_test))
print(classification_report(y90_test, y90_pred, digits=3))


# In[27]:


from sklearn.neural_network import MLPClassifier
from sklearn import metrics

mlp = MLPClassifier(
    hidden_layer_sizes=(100,15,5),
    max_iter=10000,
    alpha=1e-4,
    solver="adam",
    verbose=False,
    random_state=1,
    learning_rate_init=1e-3)

mlp_ = mlp.fit(X90_train, y90_train)
pred = mlp_.predict(X90_test)

print("Accuracy:",metrics.accuracy_score(y90_test, pred))


# In[28]:


X90_train.shape


# In[29]:


def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X.iloc[:, 0].min()-pad, X.iloc[:, 0].max()+pad
    y_min, y_max = X.iloc[:, 1].min()-pad, X.iloc[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plt.scatter(X.iloc[:,0], X.iloc[:,1], s=70, c=y, cmap=mpl.cm.Paired)
    # Support vectors indicated in plot by vertical lines
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='x', s=100, linewidths=1)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)


# In[30]:


support = svm.LinearSVC(C=1, random_state = 0).fit(X90_train, y90_train)
predicted = support.predict(X90_test)
labels=sorted(y90_train.unique())

print(classification_report  (y90_test, predicted, target_names=labels))

#y90_pred = clf.predict_proba(X90_test)

print("Accuracy:",metrics.accuracy_score(y90_test,predicted))

#plot_svc(clf, X90_train, y90_train)


# In[31]:


fig, ax = plt.subplots(figsize = (18,10))
plt.plot(figsize = (20,10))
plot_roc_curve(lr, X90_test, y90_test, ax=ax)
plot_roc_curve(lrs, X90_test[featureslr], y90_test, ax=ax)
plot_roc_curve(lda, X90_test, y90_test, ax=ax)
plot_roc_curve(qda, X90_test, y90_test, ax=ax)
plot_roc_curve(model_rfc, X90_test, y90_test, ax=ax)
plot_roc_curve(model_rfcs, X90_test[featuresrf], y90_test, ax=ax)
plot_roc_curve(mlp, X90_test, y90_test, ax=ax)
plot_roc_curve(support, X90_test, y90_test, ax=ax)
ax.plot([0, 1], [0, 1], linestyle='--', color='k')


# In[32]:


import easygui as eg
import tkinter as tk 

X_columns = [
        
        'number_of_reviews', 'review_scores_rating', 'reviews_per_month', \
        'instant_bookable', 'calculated_host_listings_count',  'price',\
        '24-hour check-in', 'Air conditioning', 'Breakfast', 'Cable TV', \
        'Carbon monoxide detector', 'Cat', 'Dog', 'Doorman', 'Doorman Entry', \
        'Dryer', 'Elevator', 'Essentials', 'Eco friendly', 'Fire extinguisher', \
        'First aid kit', 'Free parking on premises', \
        'Paid street parking off premises', 'Gym', 'Hair dryer', 'Hangers', \
        'Heating', 'Hot tub', 'Indoor fireplace', 'Internet', 'Iron', 'Keypad', \
        'Kitchen', 'Laptop friendly workspace', 'Lock on bedroom door', \
        'Lockbox', 'Other pet(s)', 'Paid parking off premises', 'Pool', \
        'Private entrance', 'Shampoo', 'Smart lock', 'Smoke alarm', \
        'Smoking allowed', 'TV', 'Washer', 'Washer / Dryer'
]


# In[35]:


# Code for demostration!!!


# In[33]:


window = tk.Tk()
window.title('Enter Features')
window.geometry('2000x1000')
list_var = [0 for i in range(47)]
list_c = [0 for i in range(47)]
list_e = [0 for i in range(47)]
list_l = [0 for i in range(47)]

for idx, text in enumerate(X_columns):
    if idx <= 5:
        #list_var[idx] = tk.IntVar()
        list_l[idx] = tk.Label(window, text=text, font=('Arial', 14)).place(x=10, y=20*(idx+1))
        list_var[idx] = tk.Entry(window, font=('Arial', 14), textvariable= tk.IntVar())
        list_var[idx].place(x=210, y=20*(idx+1))
    elif idx >= 6 and idx < 12:
        list_var[idx] = tk.IntVar()
        list_c[idx] = tk.Checkbutton(window, text=text,variable=list_var[idx], onvalue=1, offvalue=0).place(x=10+200*(idx-6),y=150)
        
    elif idx >= 12 and idx < 18:
        list_var[idx] = tk.IntVar()
        list_c[idx] = tk.Checkbutton(window, text=text,variable=list_var[idx], onvalue=1, offvalue=0).place(x=10+200*(idx-12),y=180)
        
    elif idx >= 18 and idx < 24:
        list_var[idx] = tk.IntVar()
        list_c[idx] = tk.Checkbutton(window, text=text,variable=list_var[idx], onvalue=1, offvalue=0).place(x=10+200*(idx-18),y=210)

    elif idx >= 24 and idx < 30:
        list_var[idx] = tk.IntVar()
        list_c[idx] = tk.Checkbutton(window, text=text,variable=list_var[idx], onvalue=1, offvalue=0).place(x=10+200*(idx-24),y=240)
        
    elif idx >= 30 and idx < 36:
        list_var[idx] = tk.IntVar()
        list_c[idx] = tk.Checkbutton(window, text=text,variable=list_var[idx], onvalue=1, offvalue=0).place(x=10+200*(idx-30),y=270)  
        
    elif idx >= 36 and idx < 42:
        list_var[idx] = tk.IntVar()
        list_c[idx] = tk.Checkbutton(window, text=text,variable=list_var[idx], onvalue=1, offvalue=0).place(x=10+200*(idx-36),y=300)        
  
    elif idx >= 42:
        list_var[idx] = tk.IntVar()
        list_c[idx] = tk.Checkbutton(window, text=text,variable=list_var[idx], onvalue=1, offvalue=0).place(x=10+200*(idx-42),y=330)

label = tk.StringVar()
l = tk.Label(window, textvariable=label, bg='blue', fg='white', font=('Arial', 12), width=20, height=2)
l.place(x=510,y=390)
label_t = tk.Label(window, text='The predicted answer is: ', font=('Arial', 14)).place(x=310, y=390)
        
def hit():
    list_data = [0 for i in range(47)]
    on_hit = False
    if on_hit == False:
        on_hit = True
        for idx, text in enumerate(X_columns):
            list_data[idx] = float(list_var[idx].get())
        pred = model_rfc.predict(pd.DataFrame(list_data).T)
        label.set(pred)


        
b1 = tk.Button(window, text='Submit', font=('Arial', 14), width=10, height=1, command=hit)
b1.place(x=510,y=360)

bexit = tk.Button(window, text = "Exit", font=('Arial', 14), command = window.quit)
bexit.place(x=540,y=430)


# In[34]:


window.mainloop()


# In[ ]:




