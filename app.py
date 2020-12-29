import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

import warnings
warnings.filterwarnings("ignore")

from labels import labels_dict

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def prepare_model_input(feats):
    loanptrate = np.log(feats[0] + 1)
    loan_type = labels_dict['loantype'][feats[1]]
    me_lack_req = labels_dict['me_lack_req'][feats[2]]
    loan_amount = np.log(feats[3] + 1)
    me_svc_stat = labels_dict['me_svc_stat'][feats[4]]
    gross_pay = np.log(feats[5] + 1)
    pnp_bill_mode = labels_dict['pnpbillmode'][feats[6]]

    return np.array(
        [
            loanptrate,
            loan_type,
            me_lack_req,
            loan_amount,
            me_svc_stat,
            gross_pay,
            pnp_bill_mode
        ]
    ).reshape(-1,7)

def model_predict(model, X):
        return model.predict_proba(X)

def plot_metrics(model, x_test, y_test):
        
    st.subheader('Precision-Recall Curve')
    plot_precision_recall_curve(model, x_test, y_test)
    st.pyplot()

@st.cache(persist=True)
def load_data():
    x_test = pd.read_csv(
        os.path.join(os.getcwd(), 'datasets/x_test.csv')
    )
    y_test = pd.read_csv(
        os.path.join(os.getcwd(), 'datasets/y_test.csv')
    )

    return x_test, y_test.iloc[:,1]

def main():

    x_test, y_test = load_data()

    st.title("PSSLAI New Application Credit Scorecard")
    st.sidebar.title("Client Information Form")
    st.sidebar.markdown("Please fill up the all the required information below to calculate the credit score of the client.")
    
    model = joblib.load(os.path.join(os.getcwd(), 'decision_tree.joblib'))
    # x_train, x_test, y_train, y_test = split(df)

    #Drop down for Loan Type
    st.sidebar.subheader("What loan type is the member applying for?")
    loan_type = st.sidebar.selectbox("Loan_Type", [i for i in labels_dict['loantype'].keys()])

    #Loan Term input bar
    st.sidebar.subheader("How long in months is the client applying for?")
    loanptrate = st.sidebar.slider("Loan Term", 1, 60, step=1, key='term')

    #Drop down for PNP Bill Mode
    st.sidebar.subheader("Choose the appropriate PNP bill mode")
    pnp_bill_mode = st.sidebar.selectbox("PNP Bill Mode", [i for i in labels_dict['pnpbillmode'].keys()])

    #Drop down for Member Service Status
    st.sidebar.subheader("What's the member's current service status?")
    me_svc_stat = st.sidebar.selectbox("Service Status", [i for i in labels_dict['me_svc_stat'].keys()])

    #Drop down for Lack of Requeirements
    st.sidebar.subheader("Does the member still have pending requirements in his/her application?")
    me_lack_req = st.sidebar.radio("Lacking Requirements", [i for i in labels_dict['me_lack_req'].keys()], key='lack_req')

    #Loan Amount input bar
    st.sidebar.subheader("How much is the client applying for (in PHP)?")
    loan_amount = st.sidebar.number_input("Loan Amount", 5000.0, 10000000.0, step=1000.0, key='loanamt')

    #Gross Pay input bar
    st.sidebar.subheader("How much is the client's gross pay (in PHP)?")
    gross_pay = st.sidebar.number_input("Gross Pay", 0.0, 10000000.0, step=1000.0, key='gross_pay')

    X = prepare_model_input([
        loanptrate,
        loan_type,
        me_lack_req,
        loan_amount,
        me_svc_stat,
        gross_pay,
        pnp_bill_mode
    ])

    if st.sidebar.button("Calculate Credit Score", key='credit_score_button'):
        output_text = "The clien'ts probabilty to default the loan is "
        prob_default = model_predict(model, X)[:,1][0]
        st.write(output_text + '{:.2%}'.format(prob_default))
        # st.write(y_test)
        st.subheader('Precision-Recall Curve')
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()

    if st.sidebar.checkbox("Show raw data", False):
        st.write(x_test)

if __name__ == '__main__':
    main()