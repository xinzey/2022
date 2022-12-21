#!/usr/bin/env python
# coding: utf-8

# In[1]:


import folium
import warnings
import tkinter as tk

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

data['availability_90_Percent'] = data['availability_90'].div(90)
data['availability_90_D'] =             pd.cut(data['availability_90_Percent'], [-0.01, 0.8, 1.01], labels = ['Normal', 'Good']) 


# In[4]:


X = data[X_columns]
y90 = data['availability_90_D']
X90_train, X90_test, y90_train, y90_test = train_test_split(X, y90, train_size=0.7, random_state=2)

rfc=RandomForestClassifier(n_estimators=100,n_jobs = -1,random_state =50, min_samples_leaf = 10)
model_rfc = rfc.fit(X90_train, y90_train)
#y90_pred = model_rfc.predict(X90_test)
#print(confusion_matrix(y90_pred, y90_test))
#print(classification_report(y90_test, y90_pred, digits=3))


# In[5]:


# Code for demostration!!!


# In[6]:


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


# In[7]:


window.mainloop()


# In[ ]:




