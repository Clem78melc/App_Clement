import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import numpy as np


# Chargement des données

col1, col2, col3 = st.columns([1,4,1])

with col1:
    st.write("")

with col2:
    st.image("LOGO_PAD.png")
    

with col3:
    st.write("")



features = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
       'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
       'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE',
       'WEEKDAY_APPR_PROCESS_START', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
       'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
       'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
       'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'FLAG_MOBIL', 'FLAG_EMP_PHONE',
       'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',
       'CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'HOUR_APPR_PROCESS_START',
       'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION',
       'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY',
       'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'EXT_SOURCE_2',
       'EXT_SOURCE_3', 'OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE',
       'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE',
       'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',
       'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',
       'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',
       'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',
       'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
       'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',
       'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',
       'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
       'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
       'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

# URL de l'API
predict_url = 'http://127.0.0.1:5000/predict'


train = pd.DataFrame('new_tr.csv')
test = pd.DataFrame('application_test.csv')


client_id = st.number_input('Sélectionnez l\'ID du client', min_value=100001, max_value=100001+train.shape[0])

num_col = test.select_dtypes(exclude='object').columns
cat_col = test.select_dtypes(include='object').columns

def get_prediction(row_number):
    data = {'data': row_number}
    response = requests.post(predict_url, json=data)
    return response.json(),response.status_code

def get_csv():
    response = requests.post(_url, json=data)
    return response.json(),response.status_code


if client_id in list(test['SK_ID_CURR']):
    if st.button("Get Prediction"):
        prediction,status = get_prediction(client_id)
        if status == 200:
            st.write(test.loc[test['SK_ID_CURR']==client_id])
            feature_importance = prediction['feature_importance']
            prediction_probabilities = prediction['prob']
            pred = prediction['prediction']
            gauge = prediction['gauge']
            # Display prediction result
            threshold = 0.497
            gauge_color = 'green' if prediction_probabilities[0] <= threshold else 'red'
            st.markdown(f'<p style="color: {gauge_color}; font-size: 24px;"> {gauge}</p>', unsafe_allow_html=True)

            st.progress(1 - prediction_probabilities[0])
            feature_importance = dict(sorted(feature_importance.items(), key=lambda item: np.abs(item[1]),reverse=True))
            top_importances = [-i for i in feature_importance.values()]
            colors = ['red' if importance < 0 else 'blue' for importance in top_importances]
            pd.DataFrame()
            # Create the horizontal bar plot
            fig, ax = plt.subplots()

            ax.barh(range(len(top_importances)), top_importances, color=colors)
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels([i for i in feature_importance.keys()])
            ax.invert_yaxis()  # Invert the y-axis to show the highest importance at the top

            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.title('Top 10 LIME Feature Importances')
            st.pyplot(fig)
        else:
            st.error('Erreur lors de la récupération des prédictions')
else:
    st.write('Enter a valid client ID')



feature_list = list(test.columns)
selected_features = st.multiselect('Sélectionnez les features', features)

# Graphique de distribution
if selected_features:
    for feature in selected_features:
        fig, ax = plt.subplots()
        sns.histplot(data=train, x=feature, hue='TARGET', kde=True)
        st.pyplot(fig)
        try:
            client_value = float(test.loc[test['SK_ID_CURR']==client_id, feature])
        except:
            client_value = str(test.loc[test['SK_ID_CURR']==client_id, feature].values[0])
        st.write('Valeur du client pour {} : {}'.format(feature, client_value))
        fig, ax = plt.subplots()
        if feature in num_col:
            sns.boxplot(data=train, x=feature)
            sns.scatterplot(x=[client_value], y=[0], color='red', marker='X', s=100)
            st.pyplot(fig)


# Graphique bi-varié
if len(selected_features) == 2:
    try:
        client_value1 = float(test.loc[test['SK_ID_CURR']==client_id, selected_features[0]])
    except:
        client_value1 = str(test.loc[test['SK_ID_CURR']==client_id, selected_features[0]].values[0])
    try:
        client_value2 = float(test.loc[test['SK_ID_CURR']==client_id, selected_features[1]])
    except:
        client_value2 = str(test.loc[test['SK_ID_CURR']==client_id, selected_features[1]].values[0])
    fig, ax = plt.subplots()
    sns.scatterplot(data=train, x=selected_features[0], y=selected_features[1], hue='TARGET',s=10)
    sns.scatterplot(x=[client_value1], y=[client_value2], color='red', marker='X', s=100)
    st.pyplot(fig)

# Feature importance globale
st.image('Featur_exp.png')