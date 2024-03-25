#from sklearn.externals import joblib
#import sklearn.external.joblib as extjoblib
import joblib
from joblib import load
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import import_ipynb
import re 
#from ipynb.fs.full.ipynb.fs.full.random_forest import *
#from ipynb.fs.full.random_forest import *
#from ipynb.fs.defs.random_forest.ipynb import *
#import ipynb.fs.defs.random_forest
from random_forest import lower_text, remove_url, perform_stemming


print('hihi')
import joblib
from joblib import load
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
chemin_du_modele='./saved_model.joblib'

loaded_model = joblib.load(chemin_du_modele)
tfidf = joblib.load('./tfidf_vectorizer.joblib')
#tfidf = TfidfVectorizer(max_features= 2500)

# Définir un objet (classe) pour réaliser des requetes
class InputText(BaseModel):
    text: str

#tfidf = TfidfVectorizer(max_features= 2500)
# Création d'une nouvelle instance FastAPI
app = FastAPI()

# Définir la fonction de prédiction
@app.post("/predict")
async def predict_sentiment(input_text: InputText):
    # Récupérer le texte d'entrée
    text = input_text.text
    text = lower_text(text)
    text = remove_url(text)
    text = re.sub('[,\.!?()"]', '', text)
    text = perform_stemming(text)
  
    text_vectorized= tfidf.transform([text])
    prediction_is=loaded_model.predict(text_vectorized)
    #return {"prediction": int(prediction[0])}
    return prediction_is
# Exécutez l'application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)