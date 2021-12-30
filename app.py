#manipulation des données
import pandas as pd
import numpy as np

#requêtes à l'API
import requests

#plots
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import seaborn as sns

#récupérer modèles & explainer
import joblib

#interprétabilité
import shap
shap.initjs()

#dashboard
import streamlit as st
import streamlit.components.v1 as components
st.set_option('deprecation.showPyplotGlobalUse', False)

#définition de fonctions
#fonction requête API
def get_data(url):
    resp = requests.get(url)
    return resp.json()

#fonction pour tracer des distplots
#fonction pour tracer des distplots
def distplots(data,var,height=600):
    x1 = data.loc[data['PRED'] == 0, var]
    x2 = data.loc[data['PRED'] == 1, var]
    x=data.loc[idx][var]
    plot = ff.create_distplot([x1,x2], [0,1], show_hist=False, colors=['green','red'])
    plot.add_vline(x,line_width=2,line_dash="dash",line_color="orange",annotation_text="Client",annotation_font_color='orange',
                       annotation_font_size=18)
    plot.update_layout(height=height)
    titre = "Distribution de la variable: "+var+" & Positionnement du client"
    plot.update_layout(title_text=titre)
    return plot

#chargement des données
#chargement de l'explainer SHAP
explainer = joblib.load('explainer.sav')
#chargement des fichiers de travail
#chargement de l'échantillon
clients = pd.read_csv('xtrain_model.csv')
#chargement des résultats de la prédiction (pour les graphs)
clients_pred = pd.read_csv('sample_pred.csv')
# Nettoyage des colonnes
clients.drop(columns ='Unnamed: 0', inplace =True)
clients_pred.drop(columns ='Unnamed: 0', inplace =True)


#Titre
st.title("Implémenter un modèle de scoring")

#liste pour sélectionner un client
id_client = st.selectbox('Merci de séléctionner un Client ID :',clients.SK_ID_CURR.tolist() )
id_client = int(id_client)

#url de requetage en fonction de l'ID client
url = "http://127.0.0.1:5000/predict/"
identif = str(id_client) 
url_req = url + identif

#résultat de la requête
predict = get_data(url_req)
proba_pred = predict['predictions']

#Affichage Crédit accepté/refusé
texte = "Résultat pour le client_id: "+identif
if proba_pred < 0.41:
    texte = texte + "  ---> <span style='color:green;font-size:20px;'> ACCCORD </span>"
    st.write(texte,unsafe_allow_html=True)
else:
    texte = texte + "  ---> <span style='color:red;font-size:20px;'> REFUS </span>"
    st.write(texte,unsafe_allow_html=True)

#jauge de score de risque
fig = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = proba_pred,
    mode = "gauge+number+delta",
    title = {'text': "Risque de défaut de paiement"},
    delta = {'reference': 0.41, 
             'increasing':{'color':'red'},
             'decreasing':{'color':'green'}},
    gauge = {'axis': {'range': [None, 1]},
             'bar':{'color': "black"},
             'steps' : [
                 {'range': [0, 0.41], 'color': "green"},
                 {'range': [0.41, 1], 'color': "red"}],
             'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': 0.41}}))

st.plotly_chart(fig, use_container_width=True)



#affichage des informations détaillée du client sélectionné
with st.expander("Detail pour le client :"):
    st.write("Les informations détaillées pour le client :")
    st.write(clients.loc[clients['SK_ID_CURR']==id_client])

#récupération des shap_values de notre échantillon
shap_values = explainer(clients)
shap_base = shap_values.base_values.mean()

#index de l'ID client renseigné
idx = clients.loc[clients['SK_ID_CURR']==id_client].index[0]

#feature importance locale
shap_values.values=shap_values.values[:,:,1]
shap_values.base_values=shap_values.base_values[:,1]
waterfall = shap.plots._waterfall.waterfall_legacy(shap_values.base_values[idx], shap_values.values[idx], feature_names = clients.columns)
with st.expander("Details pour la décision"):
    st.write("Ce graphe montre les critères qui influencent la décision de l'algorythme")
    st.pyplot(waterfall)
    st.write("<span style='color:Crimson;'>Critères qui augmentent le risque de défaut de paiement </span>", unsafe_allow_html=True)
    st.write("<span style='color:DodgerBlue;'>Critères qui augmentent la probabilité que le client rembourse son prêt </span>", unsafe_allow_html=True)
    
 

#feature importance globale
summary_plot = shap.summary_plot(shap_values, max_display=10)

with st.expander("Critères de décision de l'algorythme"):

    st.pyplot(summary_plot)
    st.write("Ce graphe montre les 10 variables qui ont le plus de poids sur la décision de l'algorythme")
    st.write("L'axe horizontal montre l'impact sur la décision de l'algorythme (à droite: influence positive, à gauche: influence négative).")
    st.write("La couleur montre la valeur de la variable.")
    st.write("Par exemple: quand la variable prend des valeurs hautes (rouge), l'impact sur le modèle est négatif.")


#On récupère le 10 features les plus importantes 
feature_names = shap_values.feature_names
shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
vals = np.abs(shap_df.iloc[idx].values)
shap_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name', 'feature_importance_vals'])
shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
top_ten = shap_importance['col_name'].head(10).reset_index(drop=True)
top_ten = pd.DataFrame(top_ten)

#Plus d'informations
with st.expander("Plus d'informations"):
    #liste pour séléctionner la 1ere feature 
    st.write('Vous pouvez séléctionner deux variables afin de voir comment le client se situe par rapport aux autres:')
    var_1 = st.selectbox('variable 1 :',top_ten)
    list_2=top_ten.drop(top_ten[top_ten['col_name']==var_1].index)
    var_2 = st.selectbox('variable 2 :',list_2)

    st.write("Analys univariée et positionnement du client:")
    #var 1
    st.plotly_chart(distplots(clients_pred,var_1), use_container_width=True)
    #var 2
    st.plotly_chart(distplots(clients_pred,var_2), use_container_width=True)

    st.write("Analyse bivariée:")
    titre="Croisement des variables : "+var_1+" & "+var_2
    scat_plot = px.scatter(clients_pred, x=var_1, y=var_2, color="SCORE",
    title=titre, color_continuous_scale='rdylgn_r')
    st.plotly_chart(scat_plot, use_container_width=True)
