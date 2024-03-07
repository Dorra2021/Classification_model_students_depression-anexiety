#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

import joblib
from joblib import load
from typing import Optional
from fastapi import FastAPI 
from pydantic import BaseModel 

tfidf = TfidfVectorizer(max_features= 2500, min_df= 2)

# Définir un objet (classe) pour réaliser des requetes
class InputText(BaseModel):
    text: str

# Création d'une nouvelle instance FastAPI
app = FastAPI()

# Définir la fonction de prédiction
@app.post("/predict")
async def predict_sentiment(input_text: InputText):
    # Récupérer le texte d'entrée
    text = input_text.text
    # Vectorisation du texte
    text_vectorized = tfidf.transform([text]).toarray()
    # Prédiction
    prediction = loaded_model.predict(text_vectorized)
    # La prédiction doit être soit 0 soit 1
    return {"prediction": int(prediction[0])}
"""
#charger le modèle
tfidf = TfidfVectorizer(max_features= 2500, min_df= 2)

loaded_model=joblib.load('saved_model.joblib')

class InputText(BaseModel):
    text: str
#création d'une nouvelle instance FastAPI
app=FastAPI()

#Définir un objet (classe) pour réaliser des requetes
#dot notation (.)
#notre fct de prediction prends des données et elle retourne des predictions
@app.post("/predict")
async def predict_sentiment(input_text: InputText):
    #les données sur lesquelles je fais la prédiction
    text = input_text.text
    # Vectorisation du texte
    text_vectorized = tfidf.transform([text]).toarray()
    # Prédiction
    prediction = loaded_model.predict(text_vectorized)
    return {"prediction": prediction[0]}




class request_body(BaseModel):
    text : object
    label : float
    
@app.get('/')
def get_root():
    return {" message": "welcome to the student anexiety and depression prediction"}
#definition du point de terminaison (une URL à partir de la quelle les utilisateurs peuvent faire des requetes  pour ibtenir une prédiction)
#defiition du chemin du point de terminaison (API)
@app.post("/predict") #local : http://127.0.0.1:8000/predict

#definition de la fonction de prediction
def predict(data : request_body):
    #nouvelles données sur lesquelles on fait la prédiction
    new_data= [[
        data.text,
        data.label
    ]]
    
    #prediction
    #class_idx= loaded_model.predict(new_data)[0]
    
    #definition d'un resultat qui parle à l'utilisateur.
    #return {'label': }

#pour executer notre api: unicorn main:app --reload
    
"""