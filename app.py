import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from labels import labels_dict

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title("PSSLAI Credit Risk Scorecard for New Clients")
    st.sidebar.title("Client Information Form")
    # st.markdown("Please fill up the all the required information below to calculate the credit score of the client.")
    st.sidebar.markdown("Please fill up the all the required information below to calculate the credit score of the client.")

    # @st.cache(persist=True)
    # def load_data():
    #     data = pd.read_csv("mushrooms.csv")
    #     labelencoder=LabelEncoder()
    #     for col in data.columns:
    #         data[col] = labelencoder.fit_transform(data[col])
    #     return data
    
    # @st.cache(persist=True)
    # def split(df):
    #     y = df.type
    #     x = df.drop(columns=['type'])
    #     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    #     return x_train, x_test, y_train, y_test
    
    # def plot_metrics(metrics_list):
    #     if 'Confusion Matrix' in metrics_list:
    #         st.subheader("Confusion Matrix")
    #         plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
    #         st.pyplot()

    #     if 'ROC Curve' in metrics_list:
    #         st.subheader("ROC Curve")
    #         plot_roc_curve(model, x_test, y_test)
    #         st.pyplot()
        
    #     if 'Precision-Recall Curve' in metrics_list:
    #         st.subheader('Precision-Recall Curve')
    #         plot_precision_recall_curve(model, x_test, y_test)
    #         st.pyplot()

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
    
    def model_predict(X):
        return model.predict_proba(X)

    model = joblib.load(os.path.join(os.getcwd(), 'decision_tree.joblib'))
    # x_train, x_test, y_train, y_test = split(df)

    #Drop down for Loan Type
    st.sidebar.subheader("What loan type is the member applying for?")
    loan_type = st.sidebar.selectbox("Loan_Type", [i for i in labels_dict['loantype'].keys()])

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

    #Loan Term input bar
    st.sidebar.subheader("How long in months is the client applying for?")
    loanptrate = st.sidebar.number_input("Loan Term", 1, 60, step=1, key='term')

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

    # X= np.array([ 4.11087386,  2.,  0., 13.12236538,  0.,10.52779364,  8.]).reshape(-1,7)

    if st.sidebar.button("Calculate Credit Score", key='credit_score_button'):
        st.write(model_predict(X))

    # if classifier == 'Logistic Regression':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
    #     max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

    #     metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    #     if st.sidebar.button("Classify", key='classify'):
    #         st.subheader("Logistic Regression Results")
    #         model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy: ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)
    
    # if classifier == 'Random Forest':
    #     st.sidebar.subheader("Model Hyperparameters")
    #     n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
    #     max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='n_estimators')
    #     bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
    #     metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

    #     if st.sidebar.button("Classify", key='classify'):
    #         st.subheader("Random Forest Results")
    #         model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
    #         model.fit(x_train, y_train)
    #         accuracy = model.score(x_test, y_test)
    #         y_pred = model.predict(x_test)
    #         st.write("Accuracy: ", accuracy.round(2))
    #         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
    #         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
    #         plot_metrics(metrics)

    # if st.sidebar.checkbox("Show raw data", False):
    #     st.subheader("Mushroom Data Set (Classification)")
    #     st.write(df)
    #     st.markdown("This [data set](https://archive.ics.uci.edu/ml/datasets/Mushroom) includes descriptions of hypothetical samples corresponding to 23 species of gilled mushrooms "
    #     "in the Agaricus and Lepiota Family (pp. 500-525). Each species is identified as definitely edible, definitely poisonous, "
    #     "or of unknown edibility and not recommended. This latter class was combined with the poisonous one.")

if __name__ == '__main__':
    main()