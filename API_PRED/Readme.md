#API Pred
This is the source code for the API "API Pred", hosted on Heroku.
This API is used in this Dashboard.
The goal of this API is to returns the probability that a customer of the bank does not repay his loan.

Files :

api.py : the main file of the Flask API  
best_model.sav : the model trained, a SGD Classifier  
xtrain_model.csv : the CSV file containing data sample from bank customers  
Procfile and requirement.txt : files needed to init and configure the server hosting the API  

How it works ?  
The API take in parameters the ID of a bank customer from xtrain_model.csv.  
Then the API finds the bank customers informations and the model returns the probability that a client will default on a loan.  
You can try the API by specifying a customer ID in the url : /predict/id_client  
