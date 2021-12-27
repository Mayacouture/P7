#imports Flask
from flask import Flask
from flask.globals import request
from flask.wrappers import Response
#imports bibliothèques manipulation données
import joblib
import pandas as pd

#chargement modèle
model = joblib.load('best_model.sav')

#chargement dataset
clients = pd.read_csv('data_sample.csv')
clients.drop(columns ='Unnamed: 0', inplace =True)

#on initialise l'API
app = Flask(__name__)

#on définit une route, url, avec l'ID du client à prédire
#on définit une route, url, avec l'ID du client à prédire
@app.route('/predict/<id_client>')
def predict(id_client:int):
    
    #récupérer l'id_client
    id_client = int(id_client)

    #on récupère les features du client
    features = clients.loc[clients['SK_ID_CURR']==id_client]
    features = features.squeeze(axis =0)
    #on créer un disctionnaire pour la prédiction
    response =  {}
    response['predictions'] = model.predict_proba([features])[0,1].tolist()

    #on retourne le dictionnaire avec la prédiction
    return response

if __name__ == "__main__":
    app.run(debug=True)