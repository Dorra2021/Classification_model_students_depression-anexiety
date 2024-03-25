#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import json
import seaborn as sns
import nltk
from nltk.stem import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import re
import warnings
warnings.filterwarnings('ignore')

from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[13]:


#drive.mount('/content/drive')
#os.chdir('/content/drive/My Drive/supervisedMLKaggle')


# In[14]:


import pandas as pd

# Spécifiez le chemin complet du fichier Excel
chemin_fichier_excel = 'C:/Users/debbi/Desktop/random_forest_model/dataset.xlsx'

# Lisez le fichier Excel en utilisant Pandas

df = pd.read_excel(chemin_fichier_excel)


# In[ ]:





# In[15]:


#df = pd.read_excel ('dataset.xlsx')


# In[16]:


df.head()


# In[17]:


df.columns


# In[18]:


df


# In[19]:


df['label'].value_counts()


# In[20]:


df.isnull().sum()


# In[21]:


df=df.dropna(how='any')


# In[22]:


df.isnull().sum()


# FEATURE ENGINEERING:

# In[23]:


df['Total_word']= df['text'].apply(lambda x : len(x.split()))


# In[24]:


df['Total_word']


# In[25]:


df


# In[26]:


def count_total_char(text): #compter le nombre de caractère
  char = 0
  for word in text.split():
    char+=len(word)
  return char


# In[27]:


df


# In[28]:


df['num_char']=df['text'].apply(count_total_char)


# In[29]:


df


# In[30]:


#plt.figure(figsize=(10,6))
#sns.kdeplot(x = df['num_char'], hue = df['label'], palette = 'winter', shade = 'True')


# Text Preprocessing

# In[31]:


def lower_text(texte):
  texte= texte.lower()
  return texte


# In[32]:


df['text']=df['text'].apply(lower_text)


# In[33]:


df


# In[34]:


def remove_url(text):
    re_url = re.compile('https?://\S+|www\.\S+')
    return re_url.sub('', text)

df['text'] = df['text'].apply(remove_url)


# In[35]:


df['text']=df['text'].apply(remove_url)


# In[36]:


df['text'] = df['text'].map(lambda x : re.sub('[,\.!?()"]', '', x))


# In[37]:


df['text']


# In[38]:


def tokenization(text):
  text=nltk.tokenize.word_tokenize(text)
  return text


# In[39]:


nltk.download('punkt')


# In[40]:


def perform_stemming(text):
  stemmer = PorterStemmer()
  new_list=[]
  words=tokenization(text)
  for word in words:
    new_list.append(stemmer.stem(word))
  return " ".join(new_list) + " "


# In[41]:


df['text_2']=df['text'].apply(perform_stemming)


# In[42]:


df['text_2']


# In[43]:


df['text']


# In[44]:


df['text_2'].head()


# In[45]:


#df['text']=df['text'].apply(str)


# In[46]:


stopwords = set(STOPWORDS)


# In[47]:


import nltk
nltk.download('stopwords')


# In[48]:


import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))


# In[49]:


df['text_2']


# In[50]:


type(df['text_2'][0])


# In[51]:


text_data = df['text_2'].str.cat(sep=' ')


# In[52]:


#wordcloud = WordCloud(width = 800, height = 800,
#                background_color ='white',
#                stopwords = stopwords,
#                min_font_size = 10).generate(text_data)


# In[53]:


#plt.figure(figsize = (8, 8), facecolor = None)
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.tight_layout(pad = 0)

#plt.show()


# In[54]:


df['text'].isnull().sum()


# In[55]:


df


# In[56]:


df[df['label']==1]['text']


# In[57]:


df['label'].value_counts()


# In[58]:


import collections


# In[59]:


df.columns


# In[60]:


no_dep_word=[]
for sentence in df[df['label']==0]['text'].to_list():
  for word in sentence.split():
    no_dep_word.append(word)

#dataframe qui contient les mots les plus fréquents
df_1 = pd.DataFrame(Counter(no_dep_word).most_common(25), columns=['Word', 'Frequency'])


# In[61]:

"""
sns.set_context('notebook', font_scale= 1.3)
plt.figure(figsize=(18,8))
sns.barplot(y = df_1['Word'], x= df_1['Frequency'], palette= 'summer')
plt.title("Most Commonly Used Words When not Depressed")
plt.xlabel("Frequnecy")
plt.ylabel("Words")
plt.show()
"""

# VECTORIZATION

# In[62]:


df[df['label']==1]['text'][8]


# In[63]:


df


# In[64]:


X=df.iloc[:, 0].values
y=df['label'].values


# In[65]:


X


# In[66]:


type(X)


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42, stratify= y)


# In[68]:


tfidf = TfidfVectorizer(max_features= 2500) #min_df=1, max_df=1)
X_train = tfidf.fit_transform(X_train).toarray()
X_test = tfidf.transform(X_test).toarray()


# In[69]:


tfidf


# Modeling

# In[70]:


model = RandomForestClassifier()
model.fit(X_train, y_train)


# In[71]:


type(X_test)


# In[72]:


# Probabilités prédites
y_prob = model.predict_proba(X_test)


# In[73]:


y_pred = model.predict(X_test)


# In[74]:


y_prob


# In[75]:


# Calcul de l'accuracy
accuracy = round(accuracy_score(y_test, y_pred), 3)

# Calcul de la précision
precision = round(precision_score(y_test, y_pred), 3)

# Calcul du rappel
recall = round(recall_score(y_test, y_pred), 3)


# In[76]:


accuracy


# In[77]:


precision


# In[78]:


recall


# In[79]:


import joblib
from pickle import *


# In[80]:


#get_ipython().system('pip install joblib')


# In[81]:


#loaded_model= joblib.dump(model, 'saved_model.joblib')


# In[87]:


#drive.mount('/content/drive')
#chemin_du_modele='/content/drive/My Drive/supervisedMLKaggle/saved_model.joblib'


# In[83]:


#chemin_du_modele='./saved_model.joblib'


# In[84]:


#loaded_model = joblib.load(chemin_du_modele)


# In[85]:


#df.info()


# Test du modèle:

# In[86]:


# Supposons que text_test contienne une seule entrée que vous voulez tester
#text_test = "i haven't slept well for 2 days it's like i'm restless why huh"
"""
# Prétraitement du texte de test
text_test = lower_text(text_test)
text_test = remove_url(text_test)
text_test = re.sub('[,\.!?()"]', '', text_test)
text_test = perform_stemming(text_test)

# Transformer le texte de test en vecteur de caractéristiques TF-IDF en utilisant le TfidfVectorizer entraîné
text_vec = tfidf.transform([text_test])

# Faire la prédiction avec le modèle chargé
single_prediction = loaded_model.predict(text_vec)

# Afficher la prédiction
print("Prédiction:", single_prediction)
"""
