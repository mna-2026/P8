import pickle
import numpy as np
import pandas as pd
from zipfile import ZipFile
import shap
import json
from fastapi import FastAPI, HTTPException
import uvicorn
import gower

app = FastAPI()


def one_hot_name_combiner(input_feature, category):
    return 'ONEHOT__' + input_feature + '__' + str(category)


# Load the selected model
print('Loading model')
with open(r'../data/model_step7.pkl', 'rb') as f:
    model = pickle.load(f)
# Load the threshold for the selected model
print('Loading threshold')
with open(r'../data/model_threshold_step7.pkl', 'rb') as f:
    threshold = pickle.load(f)
print('Model loaded')
# Load the categorical feature encoders
with open(r'../data/context.pkl', 'rb') as f:
    context = pickle.load(f)
binEncoder = context['app_bin_cat_features_encoder']
multiEncoder = context['app_multi_cat_features_encoder']
context = None
# Load the imputation values
with open(r'../data/catImputValues.pkl', 'rb') as f:
    catImpValues = pickle.load(f)
with open(r'../data/quantImputValues.pkl', 'rb') as f:
    quantImpValues = pickle.load(f)


# Load the preprocessed test dataset: it contains the pending application
# data, ready to be processed by the model
print('Loading model-ready pending application data')
with ZipFile(r'../data/dataset_test.pkl.zip') as z:
    with z.open(r'dataset_test.pkl') as f:
        modelReadyData = pickle.load(f)
# Load the pending application information: only the application part,
# without any encoding, with feature engineering and missing
print('Loading untransformed pending application data (appli part only)')
with ZipFile(r'../data/dataset_test_application.pkl.zip') as z:
    with z.open(r'dataset_test_application.pkl') as f:
        pendingApp = pickle.load(f)
# Load the train application information: only the application part,
# without any encoding, with feature engineering and missing
print('Loading untransformed train application data (appli part only)')
with ZipFile(r'../data/dataset_train_application.pkl.zip') as z:
    with z.open(r'dataset_train_application.pkl') as f:
        trainApp = pickle.load(f)
print('Datasets loaded')

# Initialize the SHAP explainer
print('Initialize explanation')
explainer = shap.TreeExplainer(model,
                               feature_perturbation='tree_path_dependent')
print('Explaination initialized')

print('API ready')


@app.get('/health-check')
async def healthCheck():
    return 'The API is up and running'


@app.post('/simulations/{appId}')
async def simulateScore(appId: int, overriddenData: dict):
    if appId not in modelReadyData.index or appId not in pendingApp.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    overriddenModelReadyData = modelReadyData.loc[[appId]].copy()
    overriddenApp = pendingApp.loc[[appId]].copy()
    # Overriding
    for k in overriddenData.keys():
        value = overriddenData[k]
        match k:
            case 'EXT_SOURCE_1' | 'EXT_SOURCE_2' | 'EXT_SOURCE_3':
                value = value / 100.0
            case 'DAYS_BIRTH':
                value = int(round(value * -365.2425, 0))
            case 'DAYS_EMPLOYED':
                value = int(round(value * (-365.2425/4)))
        overriddenApp.loc[appId, k] = value
    # Encoding
    encData = binEncoder.transform(overriddenApp[binEncoder.feature_names_in_])
    overriddenApp.loc[:, 'BINARY__' + binEncoder.feature_names_in_] = encData
    encData = multiEncoder.transform(overriddenApp[multiEncoder.feature_names_in_])
    overriddenApp.loc[:, multiEncoder.get_feature_names_out()] = encData
    # Imputing
    cols = overriddenApp.columns[overriddenApp.isna().any()]
    cols = cols.intersection(pd.concat([catImpValues, quantImpValues]).index)
    overriddenApp[cols] = overriddenApp[cols].fillna(pd.concat([catImpValues, quantImpValues])[cols])
    # Integrating the overridden application data to the model-ready data
    overriddenModelReadyData.loc[:, overriddenApp.columns] = overriddenApp
    overriddenModelReadyData = overriddenModelReadyData[modelReadyData.columns]
    shapValues = explainer(overriddenModelReadyData.loc[[appId]])
    response = {
        'threshold': threshold,
        'score': model.predict_proba(overriddenModelReadyData)[0][1].item(),
        'explanation': {
            'values': shapValues[0].values.tolist(),
            'base_values': shapValues[0].base_values.tolist(),
            'data': shapValues[0].data.tolist(),
            'feature_names': shapValues[0].feature_names
        }
    }
    return response    


@app.get('/customer-info/{appId}')
async def customerInfo(appId: int):
    if appId not in modelReadyData.index or appId not in pendingApp.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    info = pendingApp.loc[appId, :]
    # Let's transform DAYS_BIRTH in age and DAYS_EMPLOYED in length of service
    age = int(np.floor(info['DAYS_BIRTH'] / -365.2425))
    if not np.isnan(np.floor(info['DAYS_EMPLOYED'])):
        los = int(np.floor(info['DAYS_EMPLOYED'] / (-365.2425/4)))
    else:
        los = 'Unknown'
    # Adjust external credit score to [0-100] range
    if not np.isnan(info['EXT_SOURCE_1']):
        es1 = round(info['EXT_SOURCE_1'] * 100, 2)
    else:
        es1 = 'Unknown'
    if not np.isnan(info['EXT_SOURCE_2']):
        es2 = round(info['EXT_SOURCE_2'] * 100, 2)
    else:
        es2 = 'Unknown'
    if not np.isnan(info['EXT_SOURCE_3']):
        es3 = round(info['EXT_SOURCE_3'] * 100, 2)
    else:
        es3 = 'Unknown'
    response = {
        'CODE_GENDER': {
            'name': 'Sex',
            'value': info['CODE_GENDER']
        },
        'DAYS_BIRTH': {
            'name': 'Age',
            'value': age
        },
        'NAME_FAMILY_STATUS': {
            'name': 'Family Status',
            'value': info['NAME_FAMILY_STATUS']
        },
        'CNT_CHILDREN': {
            'name': 'Number of Children',
            'value': info['CNT_CHILDREN'].item()
        },
        'NAME_EDUCATION_TYPE': {
            'name': 'Education',
            'value': info['NAME_EDUCATION_TYPE']
        },
        'NAME_HOUSING_TYPE': {
            'name': 'Housing',
            'value': info['NAME_HOUSING_TYPE']
        },
        'FLAG_OWN_REALTY': {
            'name': 'Real Estate Owner',
            'value': info['FLAG_OWN_REALTY']
        },
        'NAME_INCOME_TYPE': {
            'name': 'Income Type',
            'value': info['NAME_INCOME_TYPE']
        },
        'DAYS_EMPLOYED': {
            'name': 'Length of Service',
            'value': los
        },
        'AMT_INCOME_TOTAL': {
            'name': 'Total Income',
            'value': round(info['AMT_INCOME_TOTAL'], 2)
        },
        'NAME_CONTRACT_TYPE': {
            'name': 'Cash or Revolving',
            'value': info['NAME_CONTRACT_TYPE']
        },
        'AMT_CREDIT': {
            'name': 'Credit Amount',
            'value': round(info['AMT_CREDIT'], 2)
        },
        'AMT_GOODS_PRICE': {
            'name': 'Goods Price (Consumer Loan)',
            'value': round(info['AMT_GOODS_PRICE'], 2)
        },
        'AMT_ANNUITY': {
            'name': 'Annuity',
            'value': round(info['AMT_ANNUITY'], 2)
        },
        'EXT_SOURCE_1': {
            'name': 'Financial Score 1',
            'value': es1
        },
        'EXT_SOURCE_2': {
            'name': 'Financial Score 2',
            'value': es2
        },
        'EXT_SOURCE_3': {
            'name': 'Financial Score 3',
            'value': es3
        },
        'data': info.to_json()
    }
    return response


@app.get('/scores/{appId}')
async def score(appId: int):
    if appId not in modelReadyData.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    response = {
        'threshold': threshold,
        'score': model.predict_proba(modelReadyData.loc[[appId]])[0][1].item()
    }
    return response


@app.get('/explanations/{appId}')
async def shapExplanation(appId: int):
    if appId not in modelReadyData.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    shapValues = explainer(modelReadyData.loc[[appId]])
    return {
        'values': shapValues[0].values.tolist(),
        'base_values': shapValues[0].base_values.tolist(),
        'data': shapValues[0].data.tolist(),
        'feature_names': shapValues[0].feature_names
    }


@app.get('/train-data-series')
async def trainDataSeries(featureName: str, nn: int, appId: int):
    if featureName == 'TARGET':
        errMsg = 'TARGET cannot be requested this way.'
        raise HTTPException(status_code=400, detail=errMsg)
    if featureName not in trainApp.columns:
        errMsg = featureName + ' not found. '
        errMsg += 'It doesn\'t seem a valid application feature.'
        raise HTTPException(status_code=404, detail=errMsg)
    if nn < 0:
        errMsg = 'nn must be positive.'
        raise HTTPException(status_code=400, detail=errMsg)
    if nn > 0 and appId not in pendingApp.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    if nn > 0:
        currApp = pendingApp.loc[appId, :].dropna().to_frame().transpose()
        tmpDf = trainApp[currApp.columns].dropna(how='any', axis='index')
        if nn > tmpDf.shape[0]:
            nn = tmpDf.shape[0]
        top = gower.gower_topn(currApp, tmpDf, n=nn)
        idx = tmpDf.iloc[top['index']].index
    else:
        idx = trainApp.index
    return {
        featureName: json.dumps(trainApp.loc[idx, featureName].to_list()),
        'TARGET': json.dumps(trainApp.loc[idx, 'TARGET'].to_list()),
        'index': json.dumps(idx.to_list())
    }


@app.get('/train-data-matrix')
async def trainDataMatrix(featX: str, featY: str, nn: int, appId: int):
    if featX == 'TARGET' or featY == 'TARGET':
        errMsg = 'TARGET cannot be requested this way.'
        raise HTTPException(status_code=400, detail=errMsg)
    if featX not in trainApp.columns:
        errMsg = featX + ' not found. '
        errMsg += 'It doesn\'t seem a valid application feature.'
        raise HTTPException(status_code=404, detail=errMsg)
    if featY not in trainApp.columns:
        errMsg = featY + ' not found. '
        errMsg += 'It doesn\'t seem a valid application feature.'
        raise HTTPException(status_code=404, detail=errMsg)
    if nn < 0:
        errMsg = 'nn must be positive.'
        raise HTTPException(status_code=400, detail=errMsg)
    if nn > 0 and appId not in pendingApp.index:
        errMsg = str(appId) + ' not found. '
        errMsg += 'It doesn\'t seem a valid Application Reference.'
        raise HTTPException(status_code=404, detail=errMsg)
    if nn > 0:
        currApp = pendingApp.loc[appId, :].dropna().to_frame().transpose()
        tmpDf = trainApp[currApp.columns].dropna(how='any', axis='index')
        if nn > tmpDf.shape[0]:
            nn = tmpDf.shape[0]
        top = gower.gower_topn(currApp, tmpDf, n=nn)
        idx = tmpDf.iloc[top['index']].index
    else:
        idx = trainApp.index
    return {
        featX: json.dumps(trainApp.loc[idx, featX].to_list()),
        featY: json.dumps(trainApp.loc[idx, featY].to_list()),
        'TARGET': json.dumps(trainApp.loc[idx, 'TARGET'].to_list()),
        'index': json.dumps(idx.to_list())
    }


if __name__ == "__main__":
    print('Starting server')
    uvicorn.run(app, host='0.0.0.0', port=8000)
