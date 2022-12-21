#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")


# In[7]:


data = pd.read_csv('656 raw data.csv', delimiter=",", dtype="unicode")


# In[8]:


data


# In[3]:


drops = [
    "scrape_id",
    "last_scraped",
    "name",
    "neighborhood_overview",
    "picture_url",
    "host_url",
    "host_name",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_listings_count",
    "neighbourhood",
    "neighbourhood_cleansed",
    "minimum_minimum_nights",
    "maximum_minimum_nights",
    "minimum_maximum_nights",
    "maximum_maximum_nights",
    "minimum_nights_avg_ntm",
    "maximum_nights_avg_ntm",
    "number_of_reviews_ltm",
    "number_of_reviews_l30d",
]
data.drop(columns=drops, inplace=True)
data.to_csv("airbnb_newyork_listing.csv", index=False)


# In[4]:


# opening dataset
df = pd.read_csv("airbnb_newyork_listing.csv", delimiter=",")


# In[5]:


# drop broken lines - where id is not a character of numbers
df.id = pd.to_numeric(df.id, errors="coerce")
df = df[df.id.notna()]


# In[6]:


# display the class and type of each columns
df.dtypes


# In[7]:


#####################
# formatting columns
for perc in ["host_response_rate", "host_acceptance_rate"]:
    df[perc] = pd.to_numeric(df[perc], errors="coerce")


# In[8]:


# format binary variables
for binary in [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
#    "is_location_exact",
#    "requires_license",
    "instant_bookable",
#    "require_guest_profile_picture",
#    "require_guest_phone_verification",
]:
    df[binary] = df[binary].map({"t": 1, "f": 0})


# In[9]:


# Filter
df = df[(df.availability_365>60) & (df.last_review>'2021-05-21')]


# In[10]:


amenity = '24-hour check-in	Air conditioning	Breakfast	Cable TV	Carbon monoxide detector	Cat	Dog	Doorman	Doorman Entry	Dryer	Elevator	Essentials	Eco friendly	Fire extinguisher	First aid kit	Free parking on premises	Paid street parking off premises	Gym	Hair dryer	Hangers	Heating	Hot tub	Indoor fireplace	Internet	Iron	Keypad	Kitchen	Laptop friendly workspace	Lock on bedroom door	Lockbox	Other pet(s)	Paid parking off premises	Pool	Private entrance	Shampoo	Smart lock	Smoke alarm	Smoking allowed	TV	Washer	Washer / Dryer'
list_amenity = amenity.split('\t')
list_amenity_lower = [a.lower() for a in list_amenity]
for a in list_amenity:
    df[a] = 0


# In[11]:


# amenities
for idx1, amenities in enumerate(df["amenities"]):
    for idx2, amenity in enumerate(list_amenity_lower):
        if amenity in amenities.lower():
            df[list_amenity[idx2]].iloc[idx1] = 1
        else:
            df[list_amenity[idx2]].iloc[idx1] = 0

            
        


# In[12]:


df.dropna(how = 'all', inplace = True)
df.dropna(axis = 1, inplace = True)


# In[13]:


df.reset_index()


# In[14]:


# write csv
df.to_csv("airbnb_newyork_cleaned.csv", index=False)


# In[ ]:




