
import streamlit as st

import requests
import json
import pickle

import numpy as np
import pandas as pd
import sklearn
import matplotlib
import matplotlib.pyplot as plt

import shap
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

# set page config
st.set_page_config(page_title="Loan_scoring_dashboard", # or None
                          page_icon="U+1F3E6", # or None
                          layout='wide', # or 'centered' for wide margins
                          initial_sidebar_state='auto')


# title
st.title("Dashboard interactif : décision d'octroi de crédit")
st.markdown("<i>La prédiction est ensuite analysée en décomposant les variables les plus influentes</i>", unsafe_allow_html=True)



col1, col2 = st.columns([5, 10]) # crée 3 colonnes
with col1:
    st.write("#### Merci d'entrer un identifiant client :")
    identifiant = st.number_input(' ', min_value=100001, max_value=112188)
    
    
with st.spinner('Import des données'):
    df = pd.read_csv("application_train.csv")

interpretable_important_data = ['SK_ID_CURR',
                                'PAYMENT_RATE',
                                'AMT_ANNUITY',
                                'DAYS_BIRTH',
                                'DAYS_EMPLOYED',
                                'ANNUITY_INCOME_PERC']

interpretable_important_data_target = ['SK_ID_CURR',
                                       'PAYMENT_RATE',
                                       'AMT_ANNUITY',
                                       'DAYS_BIRTH',
                                       'DAYS_EMPLOYED',
                                       'ANNUITY_INCOME_PERC',
                                       'TARGET']

with st.spinner('Import du modèle'):
    # import du modèle lgbm entrainé
    infile = open('best_model.joblib', 'rb')
    lgbm = pickle.load(infile)
    infile.close()
    
