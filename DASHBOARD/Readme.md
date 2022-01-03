#Dashboard Streamlit
This is the source code used to create this Dashboard, hosted by Streamlit.
I made this Dashboard for my Data Scientist training.
The aim of the dashboard is to be used by bank advisors to help them to know if they can approve or reject loan applications of their customers.

Files :

app.py : the main file, displaying the dashboard thanks to the streamlit library
explainer.sav : a SHAP explainer used to make SHAP plots
sample.csv : a sample of the bank customer data
sample_pred.csv : the same sample of the bank customer but with the target predict by my model
How it works ?
First the user select a ID for a customer

The dashboard will request the API Pred and get the probability that a client will default on a loan :

if P < 0.41 --> Approve the loan
Else --> Reject the loan application
The Bank advisor can clearly see the risk score on a gauge.

In order to maintain a good relationship with its customers the advisor has access to more informations. He will be able to explain more specifically the decision of the algorithm.

Those informations are displayed thanks to SHAP plots : A plot for the global feature importance and a plot for the local feature importance.

The local feature importance will help the advisor to understand the decision for his customer.

Then, the advisor can see the position of his client in relation to other individuals on different variables, 2 of the 10 most important in total.
