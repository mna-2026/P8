import copy
import streamlit as st
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import plotly.graph_objects as go
import seaborn as sns
import shap
import json
import numpy as np
import pandas as pd
from io import StringIO, BytesIO

import pickle

state = st.session_state
catCols = ['NAME_CONTRACT_TYPE',
           'CODE_GENDER',
           'FLAG_OWN_CAR',
           'FLAG_OWN_REALTY',
           'NAME_TYPE_SUITE',
           'NAME_INCOME_TYPE',
           'NAME_EDUCATION_TYPE',
           'NAME_FAMILY_STATUS',
           'NAME_HOUSING_TYPE',
           'OCCUPATION_TYPE',
           'WEEKDAY_APPR_PROCESS_START',
           'ORGANIZATION_TYPE',
           'FONDKAPREMONT_MODE',
           'HOUSETYPE_MODE',
           'WALLSMATERIAL_MODE',
           'EMERGENCYSTATE_MODE',
           'CNT_CHILDREN',
           'FLAG_MOBIL',
           'FLAG_EMP_PHONE',
           'FLAG_WORK_PHONE',
           'FLAG_CONT_MOBILE',
           'FLAG_PHONE',
           'FLAG_EMAIL',
           'CNT_FAM_MEMBERS',
           'REGION_RATING_CLIENT',
           'REGION_RATING_CLIENT_W_CITY',
           'HOUR_APPR_PROCESS_START',
           'REG_REGION_NOT_LIVE_REGION',
           'REG_REGION_NOT_WORK_REGION',
           'LIVE_REGION_NOT_WORK_REGION',
           'REG_CITY_NOT_LIVE_CITY',
           'REG_CITY_NOT_WORK_CITY',
           'LIVE_CITY_NOT_WORK_CITY',
           'FLAG_DOCUMENT_2',
           'FLAG_DOCUMENT_3',
           'FLAG_DOCUMENT_4',
           'FLAG_DOCUMENT_5',
           'FLAG_DOCUMENT_6',
           'FLAG_DOCUMENT_7',
           'FLAG_DOCUMENT_8',
           'FLAG_DOCUMENT_9',
           'FLAG_DOCUMENT_10',
           'FLAG_DOCUMENT_11',
           'FLAG_DOCUMENT_12',
           'FLAG_DOCUMENT_13',
           'FLAG_DOCUMENT_14',
           'FLAG_DOCUMENT_15',
           'FLAG_DOCUMENT_16',
           'FLAG_DOCUMENT_17',
           'FLAG_DOCUMENT_18',
           'FLAG_DOCUMENT_19',
           'FLAG_DOCUMENT_20',
           'FLAG_DOCUMENT_21']
numCols = ['AMT_INCOME_TOTAL',
           'AMT_CREDIT',
           'AMT_ANNUITY',
           'AMT_GOODS_PRICE',
           'REGION_POPULATION_RELATIVE',
           'DAYS_BIRTH',
           'DAYS_EMPLOYED',
           'DAYS_REGISTRATION',
           'DAYS_ID_PUBLISH',
           'OWN_CAR_AGE',
           'EXT_SOURCE_1',
           'EXT_SOURCE_2',
           'EXT_SOURCE_3',
           'APARTMENTS_AVG',
           'BASEMENTAREA_AVG',
           'YEARS_BEGINEXPLUATATION_AVG',
           'YEARS_BUILD_AVG',
           'COMMONAREA_AVG',
           'ELEVATORS_AVG',
           'ENTRANCES_AVG',
           'FLOORSMAX_AVG',
           'FLOORSMIN_AVG',
           'LANDAREA_AVG',
           'LIVINGAPARTMENTS_AVG',
           'LIVINGAREA_AVG',
           'NONLIVINGAPARTMENTS_AVG',
           'NONLIVINGAREA_AVG',
           'APARTMENTS_MODE',
           'BASEMENTAREA_MODE',
           'YEARS_BEGINEXPLUATATION_MODE',
           'YEARS_BUILD_MODE',
           'COMMONAREA_MODE',
           'ELEVATORS_MODE',
           'ENTRANCES_MODE',
           'FLOORSMAX_MODE',
           'FLOORSMIN_MODE',
           'LANDAREA_MODE',
           'LIVINGAPARTMENTS_MODE',
           'LIVINGAREA_MODE',
           'NONLIVINGAPARTMENTS_MODE',
           'NONLIVINGAREA_MODE',
           'APARTMENTS_MEDI',
           'BASEMENTAREA_MEDI',
           'YEARS_BEGINEXPLUATATION_MEDI',
           'YEARS_BUILD_MEDI',
           'COMMONAREA_MEDI',
           'ELEVATORS_MEDI',
           'ENTRANCES_MEDI',
           'FLOORSMAX_MEDI',
           'FLOORSMIN_MEDI',
           'LANDAREA_MEDI',
           'LIVINGAPARTMENTS_MEDI',
           'LIVINGAREA_MEDI',
           'NONLIVINGAPARTMENTS_MEDI',
           'NONLIVINGAREA_MEDI',
           'TOTALAREA_MODE',
           'OBS_30_CNT_SOCIAL_CIRCLE',
           'DEF_30_CNT_SOCIAL_CIRCLE',
           'OBS_60_CNT_SOCIAL_CIRCLE',
           'DEF_60_CNT_SOCIAL_CIRCLE',
           'DAYS_LAST_PHONE_CHANGE',
           'AMT_REQ_CREDIT_BUREAU_HOUR',
           'AMT_REQ_CREDIT_BUREAU_DAY',
           'AMT_REQ_CREDIT_BUREAU_WEEK',
           'AMT_REQ_CREDIT_BUREAU_MON',
           'AMT_REQ_CREDIT_BUREAU_QRT',
           'AMT_REQ_CREDIT_BUREAU_YEAR',
           'DAYS_EMPLOYED_PERC',
           'INCOME_CREDIT_PERC',
           'INCOME_PER_PERSON',
           'ANNUITY_INCOME_PERC',
           'PAYMENT_RATE']
colDescriptions = {'NAME_CONTRACT_TYPE': 'Identification if loan is cash or revolving',
                   'CODE_GENDER': 'Gender of the client',
                   'FLAG_OWN_CAR': 'Flag if the client owns a car',
                   'FLAG_OWN_REALTY': 'Flag if client owns a house or flat',
                   'CNT_CHILDREN': 'Number of children the client has',
                   'AMT_INCOME_TOTAL': 'Income of the client',
                   'AMT_CREDIT': 'Credit amount of the loan',
                   'AMT_ANNUITY': 'Loan annuity',
                   'AMT_GOODS_PRICE': 'For consumer loans it is the price of the goods for which the loan is given',
                   'NAME_TYPE_SUITE': 'Who was accompanying client when he was applying for the loan',
                   'NAME_INCOME_TYPE': 'Clients income type (businessman, working, maternity leave,…)',
                   'NAME_EDUCATION_TYPE': 'Level of highest education the client achieved',
                   'NAME_FAMILY_STATUS': 'Family status of the client',
                   'NAME_HOUSING_TYPE': 'What is the housing situation of the client (renting, living with parents, ...)',
                   'REGION_POPULATION_RELATIVE': 'Normalized population of region where client lives (higher number means the client lives in more populated region)',
                   'DAYS_BIRTH': 'Client\'s age in days at the time of application',
                   'DAYS_EMPLOYED': 'How many days before the application the person started current employment',
                   'DAYS_REGISTRATION': 'How many days before the application did client change his registration',
                   'DAYS_ID_PUBLISH': 'How many days before the application did client change the identity document with which he applied for the loan',
                   'OWN_CAR_AGE': 'Age of client\'s car',
                   'FLAG_MOBIL': 'Did client provide mobile phone (1=YES, 0=NO)',
                   'FLAG_EMP_PHONE': 'Did client provide work phone (1=YES, 0=NO)',
                   'FLAG_WORK_PHONE': 'Did client provide home phone (1=YES, 0=NO)',
                   'FLAG_CONT_MOBILE': 'Was mobile phone reachable (1=YES, 0=NO)',
                   'FLAG_PHONE': 'Did client provide home phone (1=YES, 0=NO)',
                   'FLAG_EMAIL': 'Did client provide email (1=YES, 0=NO)',
                   'OCCUPATION_TYPE': 'What kind of occupation does the client have',
                   'CNT_FAM_MEMBERS': 'How many family members does client have',
                   'REGION_RATING_CLIENT': 'Our rating of the region where client lives (1,2,3)',
                   'REGION_RATING_CLIENT_W_CITY': 'Our rating of the region where client lives with taking city into account (1,2,3)',
                   'WEEKDAY_APPR_PROCESS_START': 'On which day of the week did the client apply for the loan',
                   'HOUR_APPR_PROCESS_START': 'Approximately at what hour did the client apply for the loan',
                   'REG_REGION_NOT_LIVE_REGION': 'Flag if client\'s permanent address does not match contact address (1=different, 0=same, at region level)',
                   'REG_REGION_NOT_WORK_REGION': 'Flag if client\'s permanent address does not match work address (1=different, 0=same, at region level)',
                   'LIVE_REGION_NOT_WORK_REGION': 'Flag if client\'s contact address does not match work address (1=different, 0=same, at region level)',
                   'REG_CITY_NOT_LIVE_CITY': 'Flag if client\'s permanent address does not match contact address (1=different, 0=same, at city level)',
                   'REG_CITY_NOT_WORK_CITY': 'Flag if client\'s permanent address does not match work address (1=different, 0=same, at city level)',
                   'LIVE_CITY_NOT_WORK_CITY': 'Flag if client\'s contact address does not match work address (1=different, 0=same, at city level)',
                   'ORGANIZATION_TYPE': 'Type of organization where client works',
                   'EXT_SOURCE_1': 'Normalized score from external data source',
                   'EXT_SOURCE_2': 'Normalized score from external data source',
                   'EXT_SOURCE_3': 'Normalized score from external data source',
                   'APARTMENTS_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'BASEMENTAREA_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'YEARS_BEGINEXPLUATATION_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'YEARS_BUILD_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'COMMONAREA_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'ELEVATORS_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'ENTRANCES_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'FLOORSMAX_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'FLOORSMIN_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LANDAREA_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LIVINGAPARTMENTS_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LIVINGAREA_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'NONLIVINGAPARTMENTS_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'NONLIVINGAREA_AVG': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'APARTMENTS_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'BASEMENTAREA_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'YEARS_BEGINEXPLUATATION_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'YEARS_BUILD_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'COMMONAREA_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'ELEVATORS_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'ENTRANCES_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'FLOORSMAX_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'FLOORSMIN_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LANDAREA_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LIVINGAPARTMENTS_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LIVINGAREA_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'NONLIVINGAPARTMENTS_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'NONLIVINGAREA_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'APARTMENTS_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'BASEMENTAREA_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'YEARS_BEGINEXPLUATATION_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'YEARS_BUILD_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'COMMONAREA_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'ELEVATORS_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'ENTRANCES_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'FLOORSMAX_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'FLOORSMIN_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LANDAREA_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LIVINGAPARTMENTS_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'LIVINGAREA_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'NONLIVINGAPARTMENTS_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'NONLIVINGAREA_MEDI': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'FONDKAPREMONT_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'HOUSETYPE_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'TOTALAREA_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'WALLSMATERIAL_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'EMERGENCYSTATE_MODE': 'Normalized information about building where the client lives, What is average (_AVG suffix), modus (_MODE suffix), median (_MEDI suffix) apartment size, common area, living area, age of building, number of elevators, number of entrances, state of the building, number of floor',
                   'OBS_30_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings with observable 30 DPD (days past due) default',
                   'DEF_30_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings defaulted on 30 DPD (days past due) ',
                   'OBS_60_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings with observable 60 DPD (days past due) default',
                   'DEF_60_CNT_SOCIAL_CIRCLE': 'How many observation of client\'s social surroundings defaulted on 60 (days past due) DPD',
                   'DAYS_LAST_PHONE_CHANGE': 'How many days before application did client change phone',
                   'FLAG_DOCUMENT_2': 'Did client provide document 2',
                   'FLAG_DOCUMENT_3': 'Did client provide document 3',
                   'FLAG_DOCUMENT_4': 'Did client provide document 4',
                   'FLAG_DOCUMENT_5': 'Did client provide document 5',
                   'FLAG_DOCUMENT_6': 'Did client provide document 6',
                   'FLAG_DOCUMENT_7': 'Did client provide document 7',
                   'FLAG_DOCUMENT_8': 'Did client provide document 8',
                   'FLAG_DOCUMENT_9': 'Did client provide document 9',
                   'FLAG_DOCUMENT_10': 'Did client provide document 10',
                   'FLAG_DOCUMENT_11': 'Did client provide document 11',
                   'FLAG_DOCUMENT_12': 'Did client provide document 12',
                   'FLAG_DOCUMENT_13': 'Did client provide document 13',
                   'FLAG_DOCUMENT_14': 'Did client provide document 14',
                   'FLAG_DOCUMENT_15': 'Did client provide document 15',
                   'FLAG_DOCUMENT_16': 'Did client provide document 16',
                   'FLAG_DOCUMENT_17': 'Did client provide document 17',
                   'FLAG_DOCUMENT_18': 'Did client provide document 18',
                   'FLAG_DOCUMENT_19': 'Did client provide document 19',
                   'FLAG_DOCUMENT_20': 'Did client provide document 20',
                   'FLAG_DOCUMENT_21': 'Did client provide document 21',
                   'AMT_REQ_CREDIT_BUREAU_HOUR': 'Number of enquiries to Credit Bureau about the client one hour before application',
                   'AMT_REQ_CREDIT_BUREAU_DAY': 'Number of enquiries to Credit Bureau about the client one day before application (excluding one hour before application)',
                   'AMT_REQ_CREDIT_BUREAU_WEEK': 'Number of enquiries to Credit Bureau about the client one week before application (excluding one day before application)',
                   'AMT_REQ_CREDIT_BUREAU_MON': 'Number of enquiries to Credit Bureau about the client one month before application (excluding one week before application)',
                   'AMT_REQ_CREDIT_BUREAU_QRT': 'Number of enquiries to Credit Bureau about the client 3 month before application (excluding one month before application)',
                   'AMT_REQ_CREDIT_BUREAU_YEAR': 'Number of enquiries to Credit Bureau about the client one day year (excluding last 3 months before application)',
                   'DAYS_EMPLOYED_PERC': 'Number of employment days in percentage',
                   'INCOME_CREDIT_PERC': 'Income amount / credit amount ratio in percentage',
                   'INCOME_PER_PERSON': 'Income amount per family members in income unit',
                   'ANNUITY_INCOME_PERC': 'Loan annuity amount / income amount ratio in percentage',
                   'PAYMENT_RATE': 'Loan annuity amount / credit amount ratio in percentage'}
numOfNeighborsOptions = {'All': 0,
                         '100 most similar customers': 100,
                         '200 most similar customers': 200,
                         '500 most similar customers': 500,
                         '1000 most similar customers': 1000,
                         '2000 most similar customers': 2000,
                         '5000 most similar customers': 5000}

def formatNum(f):
    return str(f).rstrip('0').rstrip('.')


def updateExplanation():
    """
    Requests the shapley values for the current application reference by
    calling the API, builds the shap explanation and stores it in the session
    state
    """
    if len(state['apiUrl']) == 0:
        return
    if state['curAppId'] is None or state['explanation'] is not None:
        return
    url = state['apiUrl'] + '/explanations/' + state['curAppId']
    response = requests.get(url)
    if response.status_code != 200:
        return
    payload = json.loads(response.content)
    state['explanation'] = shap.Explanation(values=payload['values'],
                                            base_values=payload['base_values'],
                                            data=payload['data'],
                                            feature_names=payload['feature_names'])


def showExplanation():
    if state['explanation'] is not None:
        exp = state['explanation']
        ix = np.argsort(np.abs(exp.values))[::-1]
        expTable = {
            ':color[**Features**]{foreground="black"}': np.array(exp.feature_names)[ix[:10]],
            ':color[**Values**]{foreground="black"}': np.array(exp.data)[ix[:10]],
            ':color[**Contributions**]{foreground="black"}': np.array(exp.values)[ix[:10]]
        }
        numOfFeatures = 10
        fig, ax = plt.subplots()
        shap.plots.bar(exp,
                       max_display=numOfFeatures + 1,
                       ax=ax)
        # Adjust the x axis to get more room at the boundaries
        spread = ax.get_xbound()[1] - ax.get_xbound()[0]
        ax.set_xbound(ax.get_xbound()[0] - 0.05 * spread,
                      ax.get_xbound()[1] + 0.05 * spread)
        # Adjust the color for the values (seen in the source code)
        for l in ax.yaxis.get_majorticklabels()[:numOfFeatures]:
            l.set_color('#555555')
        st.subheader('TOP 10 contributions to Default Score',
                     text_alignment='center')
        st.pyplot(fig, width=800)
        st.table(expTable, width='content', hide_header=False)


def updateScoreAndThreshold():
    """
    Requests the default score and the threshold for the current application
    reference by calling the API and stores the results in the session state
    """
    if len(state['apiUrl']) == 0:
        return
    if state['curAppId'] is None or state['score'] is not None:
        return
    url = state['apiUrl'] + '/scores/' + state['curAppId']
    response = requests.get(url)
    if response.status_code != 200:
        return   
    payload = json.loads(response.content)
    state['threshold'] = payload['threshold']
    state['score'] = payload['score']


def showScore():
    if state['score'] is not None and state['threshold'] is not None:
        colors = ['#2ca02c',
                  '#8ca232',
                  '#dec12c',
                  '#f19623',
                  '#e65518',
                  '#d62728']
        value = state['score'] * 100
        threshold = state['threshold'] * 100
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=value,
            number={'font': {'color': 'black'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100],
                            'tickwidth': 2,
                            'tickcolor': '#2A3F5F'},
                   'bar': {'color': '#2A3F5F', 'thickness' : 0.4},
                   'steps': [
                       {'range': [0, threshold - 10],
                        'color': colors[0]},
                       {'range': [threshold - 10, threshold - 10/3],
                        'color': colors[1]},
                       {'range': [threshold - 10/3, threshold],
                        'color': colors[2]},
                       {'range': [threshold, threshold + 10/3],
                        'color': colors[3]},
                       {'range': [threshold + 10/3, threshold + 10],
                        'color': colors[4]},
                       {'range': [threshold + 10, 100],
                        'color': colors[5]}
                   ],
                   'threshold': {'line': {'color': 'red', 'width': 3},
                                 'thickness': 1.0,
                                 'value': threshold}}
        ))
        # The center is roughly (0.5, 0.05)
        # x = r*cos(theta), y = r*sin(theta)
        thresholdRad = np.pi * threshold / 100
        r = 0.5
        dx = r * np.cos(thresholdRad)
        dy = r * np.sin(thresholdRad)
        fig.add_annotation(x=0.5 + dx,
                           y=0.05 + dy,
                           xref='paper',
                           yref='paper',
                           text='Threshold:\n' + str(round(threshold, 1)),
                           font={'size': 20, 'color': 'black'},
                           showarrow=False)
        st.subheader('Default Score (lower is better)',
                     text_alignment='center')
        st.plotly_chart(fig)


def addLogo():
    st.markdown(
        '''
        <style>
            [data-testid="stSidebar"] {
                background-image: url(https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png);
                background-repeat: no-repeat;
                padding-top: 30px;
                background-position: 20px 8px;
                background-size: 228px 90px
            }
        </style>
        ''',
        unsafe_allow_html=True,
    )


def initState():
    if 'apiUrl' not in state:
        state['apiUrl'] = ''
    if 'appIdInput' not in state:
        state['appIdInput'] = ''
    if 'curAppId' not in state:
        state['curAppId'] = None
    if 'appData' not in state:
        state['appData'] = None
    if 'simuAppData' not in state:
        state['simuAppData'] = None
    if 'score' not in state:
        state['score'] = None
    if 'threshold' not in state:
        state['threshold'] = None
    if 'explanation' not in state:
        state['explanation'] = None
    if 'feature1' not in state:
        state['feature1'] = None
    if 'feature2' not in state:
        state['feature2'] = None
    if 'logScaleFeature1' not in state:
        state['logScaleFeature1'] = False
    if 'logScaleFeature2' not in state:
        state['logScaleFeature2'] = False
    if 'nn' not in state:
        state['nn'] = 'All'
    if 'dataAnalysisButtonDisabled' not in state:
        updateDataAnalysisButtonState()
    if 'dataAnalysisInput' not in state:
        state['dataAnalysisInput'] = {}
    if 'dataAnalysisFig' not in state:
        state['dataAnalysisFig'] = None
    if 'simuScore' not in state:
        state['simuScore'] = None
    if 'simuThreshold' not in state:
        state['simuThreshold'] = None
    if 'simuExplanation' not in state:
        state['simuExplanation'] = None


def renderHomePage():
    st.header('Information')
    st.markdown('''
        This website assists you in evaluating the application of a customer
        who requests a loan.
        In the backend, a machine learning model calculates a default score
        on the basis of your input.  
        This score is an indication to help you to decide to accept or reject
        the application. The final decision is up to you and may be different
        from what the model suggests.''')
    st.header('Instructions')
    st.markdown('''
        1. Configure the address of the model API in the menu **Settings**''')
    st.markdown('''
        2. In menu **Calculate Credit Score**, input the Application Reference
        of the customer''')
    st.markdown('''
        3. Check the information of the application in tab **Customer
        data**''')
    st.markdown('''
        4. Click on the button **Calculate Score** and check the score in tab
        **Score**. You can additionaly view the explanation of the score in
        tab **Score Interpretation**''')


def renderSettingsPage():
    apiUrl = st.text_input(
        label='Please, input the API URL of the scoring model',
        key='apiUrl',
        type='url',
        persist_state='session'
    )


def updateAppData():
    """
    Updates the current application reference, requests information by calling
    the API and display information.
    """
    if len(state['appIdInput']) > 0:
        if state['appIdInput'] != state['curAppId']:
            state['score'] = None
            state['threshold'] = None
            state['explanation'] = None
            state['appData'] = None
            state['simuAppData'] = None
            state['curAppId'] = None
            state['feature1'] = None
            state['feature2'] = None
            state['logScaleFeature1'] = False
            state['logScaleFeature2'] = False
            state['nn'] = 'All'
            updateDataAnalysisButtonState()
            state['dataAnalysisInput'] = {}
            state['dataAnalysisFig'] = None
            state['simuScore'] = None
            state['simuThreshold'] = None
            state['simuExplanation'] = None
            if len(state['apiUrl']) == 0:
                return
            state['curAppId'] = state['appIdInput']
            url = state['apiUrl'] + '/customer-info/' + state['curAppId']
            response = requests.get(url)
            if response.status_code != 200:
                return   
            payload = json.loads(response.content)
            state['appData'] = payload
            state['appData']['data'] = pd.read_json(StringIO(state['appData']['data']),
                                                    typ='series',
                                                    dtype=False)
            state['simuAppData'] = {
                k: copy.deepcopy(v) for k, v in state['appData'].items() if k != 'data'
            }
    else:
        state['score'] = None
        state['threshold'] = None
        state['explanation'] = None
        state['appData'] = None
        state['simuAppData'] = None
        state['curAppId'] = None
        state['feature1'] = None
        state['feature2'] = None
        state['logScaleFeature1'] = False
        state['logScaleFeature2'] = False
        state['nn'] = 'All'
        updateDataAnalysisButtonState()
        state['dataAnalysisInput'] = {}
        state['dataAnalysisFig'] = None
        state['simuScore'] = None
        state['simuThreshold'] = None
        state['simuExplanation'] = None


def showAppData():
    if state['appData'] is not None:
        info = state['appData']
        customerTable = {
            ':color[' + info['CODE_GENDER']['name'] + ']{foreground="black"} / :color[*CODE_GENDER*]{foreground="#555555"}': info['CODE_GENDER']['value'],
            ':color[' + info['DAYS_BIRTH']['name'] + ']{foreground="black"} / :color[*DAYS_BIRTH*]{foreground="#555555"}': f"{info['DAYS_BIRTH']['value']} years",
            ':color[' + info['NAME_FAMILY_STATUS']['name'] + ']{foreground="black"} / :color[*NAME_FAMILY_STATUS*]{foreground="#555555"}': info['NAME_FAMILY_STATUS']['value'],
            ':color[' + info['CNT_CHILDREN']['name'] + ']{foreground="black"} / :color[*CNT_CHILDREN*]{foreground="#555555"}': info['CNT_CHILDREN']['value'],
            ':color[' + info['NAME_EDUCATION_TYPE']['name'] + ']{foreground="black"} / :color[*NAME_EDUCATION_TYPE*]{foreground="#555555"}': info['NAME_EDUCATION_TYPE']['value'],
            ':color[' + info['NAME_HOUSING_TYPE']['name'] + ']{foreground="black"} / :color[*NAME_HOUSING_TYPE*]{foreground="#555555"}': info['NAME_HOUSING_TYPE']['value'],
            ':color[' + info['FLAG_OWN_REALTY']['name'] + ']{foreground="black"} / :color[*FLAG_OWN_REALTY*]{foreground="#555555"}': info['FLAG_OWN_REALTY']['value'],
        }
        finTable = {
            ':color[' + info['NAME_INCOME_TYPE']['name'] + ']{foreground="black"} / :color[*NAME_INCOME_TYPE*]{foreground="#555555"}': info['NAME_INCOME_TYPE']['value'],
            ':color[' + info['DAYS_EMPLOYED']['name'] + ']{foreground="black"} / :color[*DAYS_EMPLOYED*]{foreground="#555555"}': f"{info['DAYS_EMPLOYED']['value']} quarter(s)",
            ':color[' + info['AMT_INCOME_TOTAL']['name'] + ']{foreground="black"} / :color[*AMT_INCOME_TOTAL*]{foreground="#555555"}': f"{info['AMT_INCOME_TOTAL']['value']:,.2f} $",
            ':color[' + info['EXT_SOURCE_1']['name'] + ']{foreground="black"} / :color[*EXT_SOURCE_1*]{foreground="#555555"}': f"{info['EXT_SOURCE_1']['value']:,.2f}",
            ':color[' + info['EXT_SOURCE_2']['name'] + ']{foreground="black"} / :color[*EXT_SOURCE_2*]{foreground="#555555"}': f"{info['EXT_SOURCE_2']['value']:,.2f}",
            ':color[' + info['EXT_SOURCE_3']['name'] + ']{foreground="black"} / :color[*EXT_SOURCE_3*]{foreground="#555555"}': f"{info['EXT_SOURCE_3']['value']:,.2f}",
        }
        applicationTable = {
            ':color[' + info['NAME_CONTRACT_TYPE']['name'] + ']{foreground="black"} / :color[*NAME_CONTRACT_TYPE*]{foreground="#555555"}': info['NAME_CONTRACT_TYPE']['value'],
            ':color[' + info['AMT_CREDIT']['name'] + ']{foreground="black"} / :color[*AMT_CREDIT*]{foreground="#555555"}': f"{info['AMT_CREDIT']['value']:,.2f} $",
            ':color[' + info['AMT_ANNUITY']['name'] + ']{foreground="black"} / :color[*AMT_ANNUITY*]{foreground="#555555"}': f"{info['AMT_ANNUITY']['value']:,.2f} $",
        }
        custCol, incCol, loanCol = st.columns(3)
        with custCol:
            st.subheader('General Information')
            st.table(customerTable)
        with incCol:
            st.subheader('Financial Information')
            st.table(finTable)
        with loanCol:
            st.subheader('Loan Characteristics')
            st.table(applicationTable)


def calculateScore():
    updateScoreAndThreshold()
    updateExplanation()


def showDataAnalysis():
    if state['appData'] is not None:
        st.subheader('Univariate / Bivariate Data Analysis',
                     text_alignment='center')
        choices = state['appData']['data'].dropna().index
        with st.container():
            lblCol1, sbCol1, cbCol1 = st.columns([0.7, 5, 1], vertical_alignment='center')
            lblCol1.write('1st Feature')
            feat1 = sbCol1.selectbox(label='Feature 1',
                                     key='feature1',
                                     options=choices,
                                     index=None,
                                     label_visibility='collapsed',
                                     on_change=updateDataAnalysisButtonState)
            logScaleFeat1 = cbCol1.checkbox(label='Log scale',
                                            key='logScaleFeature1',
                                            value=False,
                                            on_change=updateDataAnalysisButtonState)
            if state['feature1'] is not None:
                _, desCol1, _ = st.columns([0.7, 5, 1], vertical_alignment='top')
                desCol1.write(colDescriptions[state['feature1']])
            lblCol2, sbCol2, cbCol2 = st.columns([0.7, 5, 1], vertical_alignment='center')
            lblCol2.write('2nd Feature')
            feat2 = sbCol2.selectbox(label='Feature 2',
                                     key='feature2',
                                     options=choices,
                                     index=None,
                                     label_visibility='collapsed',
                                     on_change=updateDataAnalysisButtonState)
            logScaleFeat2 = cbCol2.checkbox(label='Log scale',
                                            key='logScaleFeature2',
                                            value=False,
                                            on_change=updateDataAnalysisButtonState)
            if state['feature2'] is not None:
                _, desCol2, _ = st.columns([0.7, 5, 1], vertical_alignment='top')
                desCol2.write(colDescriptions[state['feature2']])
            lblCol3, sbCol3, cbCol3 = st.columns([0.7, 5, 1], vertical_alignment='center')
            lblCol3.write('Other applications')
            neighbors = sbCol3.selectbox(label='Neighbors',
                                         key='nn',
                                         options=numOfNeighborsOptions,
                                         index=0,
                                         label_visibility='collapsed')
            st.button(label='Show distribution',
                      on_click=runDataAnalysis,
                      disabled=state['dataAnalysisButtonDisabled'])
        with st.container(horizontal_alignment='center'):
            if state['dataAnalysisFig'] is not None:
                feature1 = state['dataAnalysisInput']['feature1']
                feature2 = state['dataAnalysisInput']['feature2']
                if feature1 is not None:
                    if feature2 is not None:
                        st.subheader('Bivariate Distribution: ' + feature1 + ' vs ' + feature2,
                                     text_alignment='center')
                    else:
                        st.subheader('Univariate Distribution: ' + feature1,
                                     text_alignment='center')
                st.image(state['dataAnalysisFig'], width=900)


def updateDataAnalysisButtonState():
    enabled = state['curAppId'] is not None
    enabled = enabled and (state['feature1'] is not None)
    enabled = enabled and state['feature1'] != state['feature2']
    enabled = enabled and (not state['logScaleFeature1'] or state['feature1'] in numCols)
    enabled = enabled and (not state['logScaleFeature2'] or state['feature2'] in numCols)
    state['dataAnalysisButtonDisabled'] = not enabled


def drawHistPlot(data, featureName, x, xlabel, appIdStr):
    fig, ax = plt.subplots(2, 1,
                           figsize=(7.5, 7.5),
                           sharex=True,
                           layout='constrained')
    sns.histplot(data=data,
                 x=featureName,
                 bins='fd',
                 color='#555555',
                 facecolor='orange',
                 kde=True,
                 ax=ax[0])
    sns.histplot(data=data,
                 x=featureName,
                 hue='TARGET',
                 bins='fd',
                 alpha=0.3,
                 palette=['#2ca02c', '#d62728'],                                 
                 kde=True,
                 ax=ax[1])
    legend = ax[1].get_legend()
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]
    line0 = ax[0].axvline(x, color='blue', linestyle='--', label=appIdStr)
    line1 = ax[1].axvline(x, color='blue', linestyle='--', label=appIdStr)
    ax[1].set_xlabel(xlabel)
    handles.append(line1)
    labels.append(appIdStr)
    mapping = {
        '0': 'TARGET == 0',
        '1': 'TARGET == 1',
        appIdStr: appIdStr
    }
    ax[0].legend([line0],
                 [appIdStr],
                 title='Legend',
                 alignment='left',
                 title_fontsize='large',
                 loc='best',
                 bbox_to_anchor=(1, 1))
    ax[1].legend(handles,
                 [mapping[l] for l in labels],
                 title='Legend',
                 alignment='left',
                 title_fontsize='large',
                 loc='best',
                 bbox_to_anchor=(1, 1))
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    return buffer


def drawCountPlot(data, featureName, x, appIdStr):
    fig, ax = plt.subplots(2, 1,
                           figsize=(7.5, 7.5),
                           sharex=True,
                           layout='constrained')
    sns.countplot(data=data,
                  x=featureName,
                  color='orange',
                  stat='count',
                  saturation=1.0,
                  ax=ax[0])
    sns.countplot(data=data,
                  x=featureName,
                  hue='TARGET',
                  palette=['#2ca02c', '#d62728'],
                  stat='count',
                  saturation=1.0,
                  ax=ax[1])
    legend = ax[1].get_legend()
    handles = legend.legend_handles
    labels = [text.get_text() for text in legend.get_texts()]
    line0 = ax[0].axvline(x, color='blue', linestyle='--', label=appIdStr)
    line1 = ax[1].axvline(x, color='blue', linestyle='--', label=appIdStr)
    ax[1].tick_params(axis='x', labelrotation=90)
    handles.append(line1)
    labels.append(appIdStr)
    mapping = {
        '0': 'TARGET == 0',
        '1': 'TARGET == 1',
        appIdStr: appIdStr
    }
    ax[0].legend([line0],
                 [appIdStr],
                 title='Legend',
                 alignment='left',
                 title_fontsize='large',
                 loc='best',
                 bbox_to_anchor=(1, 1))
    ax[1].legend(handles,
                 [mapping[l] for l in labels],
                 title='Legend',
                 alignment='left',
                 title_fontsize='large',
                 loc='best',
                 bbox_to_anchor=(1, 1))
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    return buffer


def drawContingencyTable(data, featureName1, featureName2, x, y, appIdStr):
    dataDf = pd.crosstab(data[featureName1],
                         data[featureName2])
    row = dataDf.index.get_loc(x)
    col = dataDf.columns.get_loc(y)
    if dataDf.shape[0] < dataDf.shape[1]:
        dataDf = dataDf.transpose()
        row = dataDf.index.get_loc(y)
        col = dataDf.columns.get_loc(x)
    fig, ax = plt.subplots(figsize=(7.5, max(6, 0.39 * dataDf.shape[0])))
    figWidth = fig.get_figwidth()
    cbarWidth = 0.20 / figWidth
    cbarPad = 0.1 / figWidth
    axPos = ax.get_position()
    # Axe for the colorbar
    cax = fig.add_axes([axPos.x1 + cbarPad,
                        axPos.y0,
                        cbarWidth,
                        axPos.height])
    # Size of the annotations adjusted to the number of columns
    fontSize = min(10, max(5, 70 / dataDf.shape[1]))
    sns.heatmap(dataDf,
                annot=True,
                cmap='Oranges',
                annot_kws={'fontsize': fontSize},
                fmt='d',
                ax=ax,
                cbar_ax=cax)
    # Compute figure layout
    fig.canvas.draw()
    # Compute cell ratio
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = ax.transData.transform((1, 1))
    cellRatio = abs(x1 - x0) / abs(y1 - y0)
    ax.add_patch(Polygon([(col, row),
                          (col + 0.33 / cellRatio, row),
                          (col, row + 0.33)],
                         closed=True,
                         facecolor='blue',
                         zorder=10))
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    return buffer


def drawBoxPlot(data, featureName1, featureName2, x, xlabel, y, ylabel, appIdStr):
    # if the 1st figure is the categorical one, we swap x with y for a better
    # display
    if featureName1 in catCols: 
        tmp = featureName1
        featureName1 = featureName2
        featureName2 = tmp
        tmp = x
        x = y
        y = tmp
        tmp = xlabel
        xlabel = ylabel
        ylabel = tmp
    fig, ax = plt.subplots(figsize=(7.5,
                                    max(2.5, 0.375 * data[featureName2].nunique())),
                           layout='constrained')
    ax = sns.boxplot(data=data,
                     x=featureName1,
                     y=featureName2,
                     color='orange',
                     order=data.groupby(featureName2).median().sort_values(featureName1).index,
                     saturation=1.0,
                     ax=ax)
    ax.set_xlabel(xlabel)
    ylim = ax.get_ylim()
    ax.scatter(x,
               y,
               label=appIdStr,
               color='blue',
               marker='^',
               s=50,
               zorder=10)
    ax.set_ylim(ylim)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles,
              labels,
              title='Legend',
              alignment='left',
              title_fontsize='large',
              loc='upper right',
              bbox_to_anchor=(1, 1))
    # compute figure layout
    fig.canvas.draw()
    # adjust the x-axis at right to avoid overlap with legend
    legendWidth = ax.get_legend().get_window_extent().width
    dataDisplayWidth = ax.spines['bottom'].get_window_extent().width
    resizefactor = dataDisplayWidth / (dataDisplayWidth - legendWidth * 1.1)
    xlim = ax.get_xlim()
    ax.set_xlim(right=xlim[0]+(xlim[1]-xlim[0])*resizefactor)
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    return buffer


def drawJoinGrid(data, featureName1, featureName2, x, xlabel, y, ylabel, appIdStr):
    g = sns.JointGrid(data=data,
                      x=featureName1,
                      y=featureName2,
                      height=7.5)
    sns.scatterplot(data=data,
                    x=featureName1,
                    y=featureName2,
                    hue='TARGET',
                    style='TARGET',
                    alpha=1.0,
                    palette=['#2ca02c', '#d62728'],
                    s=50,
                    ax=g.ax_joint)
    g.ax_joint.scatter(x,
                       y,
                       label=appIdStr,
                       color='blue',
                       marker='^',
                       zorder=10)
    sns.kdeplot(data=data,
                x=featureName1,
                hue='TARGET',
                legend=False,
                palette=['#2ca02c', '#d62728'],
                ax=g.ax_marg_x)
    sns.kdeplot(data=data,
                y=featureName2,
                hue='TARGET',
                legend=False,
                palette=['#2ca02c', '#d62728'],
                ax=g.ax_marg_y)
    g.ax_marg_x.axvline(x, color='blue', linestyle='--')
    g.ax_marg_y.axhline(y, color='blue', linestyle='--')
    g.ax_joint.set_xlabel(xlabel)
    g.ax_joint.set_ylabel(ylabel)
    handles, labels = g.ax_joint.get_legend_handles_labels()
    mapping = {
        '0': 'TARGET == 0',
        '1': 'TARGET == 1',
        appIdStr: appIdStr
    }
    g.ax_joint.legend(handles,
                      [mapping[l] for l in labels],
                      title='Legend',
                      alignment='left',
                      title_fontsize='large',
                      loc='upper right',
                      bbox_to_anchor=(1, 1))
    # compute figure layout
    g.figure.canvas.draw()
    # adjust the x-axis at right to avoid overlap with legend
    figWidth = g.figure.get_window_extent().width
    legendWidth = g.ax_joint.get_legend().get_window_extent().width
    dataDisplayWidth = g.ax_joint.spines['bottom'].get_window_extent().width
    resizefactor = dataDisplayWidth / (dataDisplayWidth - legendWidth * 0.8)
    xlim = g.ax_joint.get_xlim()
    g.ax_joint.set_xlim(right=xlim[0]+(xlim[1]-xlim[0])*resizefactor)
    g.figure.set_figwidth(g.figure.get_figwidth()*(1+legendWidth/figWidth))
    buffer = BytesIO()
    g.figure.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
    return buffer
    

def runDataAnalysis():
    usrInput = {
        'appId': state['curAppId'],
        'feature1': state['feature1'],
        'feature2': state['feature2'],
        'logScaleFeature1': state['logScaleFeature1'],
        'logScaleFeature2': state['logScaleFeature2'],
        'nn': numOfNeighborsOptions[state['nn']]
    }
    if state['dataAnalysisInput'] is not None:
        if usrInput != state['dataAnalysisInput']:
            if usrInput['feature2'] is None: # univariate distribution
                url = state['apiUrl'] + '/train-data-series'
                response = requests.get(url,
                                        params={
                                            'featureName': usrInput['feature1'],
                                            'nn': usrInput['nn'],
                                            'appId': usrInput['appId']
                                        })
                if response.status_code != 200:
                    return
                payload = json.loads(response.content)
                dataDf = pd.DataFrame(data={
                    usrInput['feature1']: json.loads(payload[usrInput['feature1']]),
                    'TARGET': json.loads(payload['TARGET'])
                })
                dataDf = dataDf.dropna(how='any')    
                xlabel = usrInput['feature1']
                x = state['appData']['data'][usrInput['feature1']]
                if usrInput['logScaleFeature1']:
                    if min(dataDf[usrInput['feature1']]) < 0:
                        dataDf[usrInput['feature1']] = dataDf[usrInput['feature1']].map(np.negative)
                        x = -x
                        xlabel = '- ' + xlabel
                    else:
                        xlabel = '+ ' + xlabel
                    if min(dataDf[usrInput['feature1']]) < 0:
                        return
                    dataDf[usrInput['feature1']] = dataDf[usrInput['feature1']].map(lambda x: np.log10(x + 1))
                    x = np.log10(x + 1)
                    xlabel = 'Log(1 ' + xlabel + ')'
                if usrInput['feature1'] in numCols: # numeric feature
                    state['dataAnalysisFig'] = drawHistPlot(dataDf,
                                                            usrInput['feature1'],
                                                            x,
                                                            xlabel,
                                                            state['curAppId'])
                else:
                    state['dataAnalysisFig'] = drawCountPlot(dataDf,
                                                             usrInput['feature1'],
                                                             x,
                                                             state['curAppId'])
            else: # bivariate distribution
                url = state['apiUrl'] + '/train-data-matrix'
                response = requests.get(url,
                                        params={
                                            'featX': usrInput['feature1'],
                                            'featY': usrInput['feature2'],
                                            'nn': usrInput['nn'],
                                            'appId': usrInput['appId']
                                        })
                if response.status_code != 200:
                    return
                payload = json.loads(response.content)
                dataDf = pd.DataFrame(data={
                    usrInput['feature1']: json.loads(payload[usrInput['feature1']]),
                    usrInput['feature2']: json.loads(payload[usrInput['feature2']]),
                    'TARGET': json.loads(payload['TARGET'])
                })
                dataDf = dataDf.dropna(how='any')    
                xlabel = usrInput['feature1']
                ylabel = usrInput['feature2']
                x = state['appData']['data'][usrInput['feature1']]
                y = state['appData']['data'][usrInput['feature2']]
                if usrInput['logScaleFeature1']:
                    if min(dataDf[usrInput['feature1']]) < 0:
                        dataDf[usrInput['feature1']] = dataDf[usrInput['feature1']].map(np.negative)
                        x = -x
                        xlabel = '- ' + xlabel
                    else:
                        xlabel = '+ ' + xlabel
                    if min(dataDf[usrInput['feature1']]) < 0:
                        return
                    dataDf[usrInput['feature1']] = dataDf[usrInput['feature1']].map(lambda x: np.log10(x + 1))
                    x = np.log10(x + 1)
                    xlabel = 'Log(1 ' + xlabel + ')'
                if usrInput['logScaleFeature2']:
                    if min(dataDf[usrInput['feature2']]) < 0:
                        dataDf[usrInput['feature2']] = dataDf[usrInput['feature2']].map(np.negative)
                        y = -y
                        ylabel = '- ' + ylabel
                    else:
                        ylabel = '+ ' + ylabel
                    if min(dataDf[usrInput['feature2']]) < 0:
                        return
                    dataDf[usrInput['feature2']] = dataDf[usrInput['feature2']].map(lambda x: np.log10(x + 1))
                    y = np.log10(y + 1)
                    ylabel = 'Log(1 ' + ylabel + ')'
                if usrInput['feature1'] in numCols and usrInput['feature2'] in numCols:
                    state['dataAnalysisFig'] = drawJoinGrid(dataDf,
                                                            usrInput['feature1'],
                                                            usrInput['feature2'],
                                                            x,
                                                            xlabel,
                                                            y,
                                                            ylabel,
                                                            usrInput['appId'])
                elif usrInput['feature1'] in catCols and usrInput['feature2'] in catCols:
                    state['dataAnalysisFig'] = drawContingencyTable(dataDf,
                                                                    usrInput['feature1'],
                                                                    usrInput['feature2'],
                                                                    x,
                                                                    y,
                                                                    usrInput['appId'])
                else: # mixed categorical & numeric
                    state['dataAnalysisFig'] = drawBoxPlot(dataDf,
                                                           usrInput['feature1'],
                                                           usrInput['feature2'],
                                                           x,
                                                           xlabel,
                                                           y,
                                                           ylabel,
                                                           usrInput['appId'])
            state['dataAnalysisInput'] = usrInput
                

def renderScorePage():
    if len(state['apiUrl']) > 0:        
        with st.container(
            width='stretch',
            height='content',
            horizontal=False,
            horizontal_alignment='left',
            vertical_alignment='bottom'
        ):
            st.write('Please, input an Application Reference')
            with st.container(
                width='stretch',
                height='content',
                horizontal=True,
                horizontal_alignment='right',
                vertical_alignment='bottom'
            ):
                st.text_input(
                    label='Please, input the Application Reference',
                    label_visibility='collapsed',
                    key='appIdInput',
                    type='default',
                    persist_state='session',
                    validate=(r'^\d{6}$', 'Application References are 6-digit numbers'),
                    on_change=updateAppData
                )
                st.button(
                    label='Calculate Score',
                    on_click=calculateScore,
                )
            dataTab, scoreTab, expTab, daTab = st.tabs(['Customer Data',
                                                        'Score',
                                                        'Score Interpretation',
                                                        'Data Analysis'])
            with dataTab:
                showAppData()
            with scoreTab:
                showScore()
            with expTab:
                showExplanation()
            with daTab:
                showDataAnalysis()
    else:
        st.write('Please, configure the settings first')


def showEditableAppData():
    if state['simuAppData'] is None:
        return

    simuData = state['simuAppData']

    def integerField(feature, suffix=None, minValue=None):
        if suffix is None:
            colLabel, colValue = st.columns([1, 2.5],
                                            vertical_alignment='center', gap='xxsmall')
        else:
            colLabel, colValue, suffixLabel = st.columns([1, 1.9, 0.6],
                                                         vertical_alignment='center', gap='xxsmall')
        with colLabel:
            st.markdown(':color[' + simuData[feature]['name'] + ']{foreground="black"}')
        with colValue:
            if simuData[feature]['value'] != 'Unknown':
                val = int(simuData[feature]['value'])
            else:
                val = None
            value = st.number_input(
                simuData[feature]['name'],
                value=val,
                min_value=minValue,
                step=1,
                key=f'app_{feature}',
                label_visibility='collapsed'
            )
        if suffix is not None:
            with suffixLabel:
                st.markdown(suffix)
        simuData[feature]['value'] = value


    def floatField(feature, suffix=None, minValue=None, maxValue=None, step=1.0):
        if suffix is None:
            colLabel, colValue = st.columns([1, 2.5],
                                            vertical_alignment='center', gap='xxsmall')
        else:
            colLabel, colValue, suffixLabel = st.columns([1, 1.9, 0.6],
                                                         vertical_alignment='center', gap='xxsmall')
        with colLabel:
            st.markdown(':color[' + simuData[feature]['name'] + ']{foreground="black"}')
        with colValue:
            if simuData[feature]['value'] != 'Unknown':
                val = float(simuData[feature]['value'])
            else:
                val = None
            value = st.number_input(
                simuData[feature]['name'],
                value=val,
                min_value=minValue,
                max_value=maxValue,
                step=step,
                format='%.2f',
                key=f'app_{feature}',
                label_visibility='collapsed'
            )
        if suffix is not None:
            with suffixLabel:
                st.markdown(suffix)
        simuData[feature]['value'] = value


    def selectField(feature, options):
        colLabel, colValue = st.columns([1, 2.5],
                                        vertical_alignment='center', gap='xxsmall')
        current_value = simuData[feature]['value']
        with colLabel:
            st.markdown(':color[' + simuData[feature]['name'] + ']{foreground="black"}')
        with colValue:
            value = st.selectbox(
                simuData[feature]['name'],
                options=options,
                index=options.index(current_value),
                key=f'app_{feature}',
                label_visibility='collapsed'
            )
        simuData[feature]['value'] = value

    custCol, incCol, loanCol = st.columns(3)
    with custCol:
        st.subheader('General Information')
        selectField(
            'CODE_GENDER',
            ['F', 'M']
        )
        integerField(
            'DAYS_BIRTH',
            suffix='year(s)',
            minValue=18
        )
        selectField(
            'NAME_FAMILY_STATUS',
            [
                'Married',
                'Single / not married',
                'Civil marriage',
                'Separated',
                'Widow',
                'Unknown'
            ]
        )
        integerField(
            'CNT_CHILDREN',
            minValue=0
        )
        selectField(
            'NAME_EDUCATION_TYPE',
            [
                'Secondary / secondary special',
                'Higher education',
                'Incomplete higher',
                'Lower secondary',
                'Academic degree'
            ]
        )
        selectField(
            'NAME_HOUSING_TYPE',
            [
                'House / apartment',
                'With parents',
                'Municipal apartment',
                'Rented apartment',
                'Office apartment',
                'Co-op apartment'
            ]
        )
        selectField(
            'FLAG_OWN_REALTY',
            ['Y', 'N']
        )
    with incCol:
        st.subheader('Financial Information')
        selectField(
            'NAME_INCOME_TYPE',
            [
                'Working',
                'Commercial associate',
                'Pensioner',
                'State servant',
                'Unemployed',
                'Student',
                'Businessman',
                'Maternity leave'
            ]
        )
        integerField(
            'DAYS_EMPLOYED',
            suffix='quarter(s)',
            minValue=0
        )
        floatField(
            'AMT_INCOME_TOTAL',
            suffix='$',
            minValue=0.0
        )
        floatField(
            'EXT_SOURCE_1',
            minValue=0.0,
            maxValue=100.0,
            step=0.01
        )
        floatField(
            'EXT_SOURCE_2',
            minValue=0.0,
            maxValue=100.0,
            step=0.01
        )
        floatField(
            'EXT_SOURCE_3',
            minValue=0.0,
            maxValue=100.0,
            step=0.01
        )
    with loanCol:
        st.subheader('Loan Characteristics')
        selectField(
            'NAME_CONTRACT_TYPE',
            [
                'Cash loans',
                'Revolving loans'
            ]
        )
        floatField(
            'AMT_CREDIT',
            suffix='$',
            minValue=0.0
        )
        floatField(
            'AMT_ANNUITY',
            suffix='$',
            minValue=0.0
        )


def showSimuScore():
    if state['simuScore'] is not None and state['simuThreshold'] is not None:
        colors = ['#2ca02c',
                  '#8ca232',
                  '#dec12c',
                  '#f19623',
                  '#e65518',
                  '#d62728']
        value = state['simuScore'] * 100
        threshold = state['simuThreshold'] * 100
        fig = go.Figure(go.Indicator(
            mode='gauge+number',
            value=value,
            number={'font': {'color': 'black'}},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={'axis': {'range': [None, 100],
                            'tickwidth': 2,
                            'tickcolor': '#2A3F5F'},
                   'bar': {'color': '#2A3F5F', 'thickness' : 0.4},
                   'steps': [
                       {'range': [0, threshold - 10],
                        'color': colors[0]},
                       {'range': [threshold - 10, threshold - 10/3],
                        'color': colors[1]},
                       {'range': [threshold - 10/3, threshold],
                        'color': colors[2]},
                       {'range': [threshold, threshold + 10/3],
                        'color': colors[3]},
                       {'range': [threshold + 10/3, threshold + 10],
                        'color': colors[4]},
                       {'range': [threshold + 10, 100],
                        'color': colors[5]}
                   ],
                   'threshold': {'line': {'color': 'red', 'width': 3},
                                 'thickness': 1.0,
                                 'value': threshold}}
        ))
        # The center is roughly (0.5, 0.05)
        # x = r*cos(theta), y = r*sin(theta)
        thresholdRad = np.pi * threshold / 100
        r = 0.5
        dx = r * np.cos(thresholdRad)
        dy = r * np.sin(thresholdRad)
        fig.add_annotation(x=0.5 + dx,
                           y=0.05 + dy,
                           xref='paper',
                           yref='paper',
                           text='Threshold:\n' + str(round(threshold, 1)),
                           font={'size': 20, 'color': 'black'},
                           showarrow=False)
        st.subheader('Simulated Default Score (lower is better)',
                     text_alignment='center')
        st.plotly_chart(fig)


def showSimuExplanation():
    if state['simuExplanation'] is not None:
        exp = state['simuExplanation']
        ix = np.argsort(np.abs(exp.values))[::-1]
        expTable = {
            ':color[**Features**]{foreground="black"}': np.array(exp.feature_names)[ix[:10]],
            ':color[**Values**]{foreground="black"}': np.array(exp.data)[ix[:10]],
            ':color[**Contributions**]{foreground="black"}': np.array(exp.values)[ix[:10]]
        }
        numOfFeatures = 10
        fig, ax = plt.subplots()
        shap.plots.bar(exp,
                       max_display=numOfFeatures + 1,
                       ax=ax)
        # Adjust the x axis to get more room at the boundaries
        spread = ax.get_xbound()[1] - ax.get_xbound()[0]
        ax.set_xbound(ax.get_xbound()[0] - 0.05 * spread,
                      ax.get_xbound()[1] + 0.05 * spread)
        # Adjust the color for the values (seen in the source code)
        for l in ax.yaxis.get_majorticklabels()[:numOfFeatures]:
            l.set_color('#555555')
        st.subheader('TOP 10 contributions to Simulated Default Score',
                     text_alignment='center')
        st.pyplot(fig, width=800)
        st.table(expTable, width='content', hide_header=False)


def simulateScore():
    if len(state['apiUrl']) == 0 or state['curAppId'] is None:
        return
    url = state['apiUrl'] + '/simulations/' + state['curAppId']
    modifiedData = {}
    for k, v in state['simuAppData'].items():
        if state['simuAppData'][k]['value'] != state['appData'][k]['value']:
            modifiedData[k] = state['simuAppData'][k]['value']
    response = requests.post(url, json=modifiedData)
    if response.status_code != 200:
        return
    payload = json.loads(response.content)
    state['simuThreshold'] = payload['threshold']
    state['simuScore'] = payload['score']
    explanationData = payload['explanation']
    state['simuExplanation'] = shap.Explanation(values=explanationData['values'],
                                                base_values=explanationData['base_values'],
                                                data=explanationData['data'],
                                                feature_names=explanationData['feature_names'])


def renderSimuPage():
    if len(state['apiUrl']) > 0:        
        with st.container(
            width='stretch',
            height='content',
            horizontal=False,
            horizontal_alignment='left',
            vertical_alignment='bottom'
        ):
            st.write('Please, input an Application Reference in page **Calculate Default Score**')
            with st.container(
                width='stretch',
                height='content',
                horizontal=True,
                horizontal_alignment='right',
                vertical_alignment='bottom'
            ):
                st.text_input(
                    label='Simulation',
                    label_visibility='collapsed',
                    disabled=True,
                    value=state['curAppId'],
                )
                st.button(
                    label='Simulate Score',
                    on_click=simulateScore,
                )
        dataTab, scoreTab, expTab = st.tabs(['Customer Data Overrides',
                                             'Simulated Score',
                                             'Simulated Score Interpretation'])
        with dataTab:
            showEditableAppData()
        with scoreTab:
            showSimuScore()
        with expTab:
            showSimuExplanation()
    else:
        st.write('Please, configure the settings first')


initState()

st.set_page_config(page_title='Prêt à dépenser',
                   page_icon=':credit_card:',
                   layout='wide')

addLogo()
nav = st.navigation(
    pages=[
        st.Page(renderHomePage,
                title='Home',
                icon=':material/home:'),
        st.Page(renderSettingsPage,
                title='Settings',
                icon=':material/settings:'),
        st.Page(renderScorePage,
                title='Calculate Default Score',
                icon=':material/credit_score:'),
        st.Page(renderSimuPage,
                title='Simulate Default Score',
                icon=':material/calculate:'),
    ]
)
nav.run()
