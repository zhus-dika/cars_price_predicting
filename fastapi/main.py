from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
from fastapi.encoders import jsonable_encoder
import numpy as np
import re
from sklearn import impute
from sklearn.preprocessing import OneHotEncoder # или можно использовать get_dummies из библиотеки pandas
from sklearn.compose import make_column_transformer

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

@app.get("/")
async def root(request: Request, message='mlds students'):
    return templates.TemplateResponse('index.html',
                                      {"request": request,
                                       "message": message})


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]

def prepare_test_data(list_items):
    df_test = pd.DataFrame(jsonable_encoder(list_items))
    print(df_test.isna().sum().sum())
    df_test['mileage'] = df_test['mileage'].replace(to_replace='[\/a-zA-Z\/]+', value='', regex=True)
    df_test['mileage'] = df_test['mileage'].astype('float')
    df_test['engine'] = df_test['engine'].replace(to_replace='[\/a-zA-Z\/]+', value='', regex=True)
    df_test['engine'] = df_test['engine'].astype('float')
    df_test['max_power'] = df_test['max_power'].replace(to_replace='[ \/a-zA-Z\/]+', value='', regex=True)
    df_test['max_power'] = df_test['max_power'].replace(to_replace='', value=None, regex=True)
    df_test['max_power'] = df_test['max_power'].astype('float')
    max_speed_torque_list = []
    torque_list = []
    for i in range(len(df_test['torque'])):
        try:
            reg = re.findall(r'\d+[,.0-9+-~@knNa]+', df_test['torque'][i])
            if len(reg) == 2:
                torque = re.findall(r'\d+', reg[0])
                if len(torque) > 1:
                    torque = float('.'.join(torque))
                else:
                    torque = float(torque[0])
                if len(re.findall(r'kgm', df_test['torque'][i])) > 0:
                    torque *= 9.8067
                torque_list.append(torque)
                max_speed_torque = re.findall(r'\d+', reg[1])
                if len(max_speed_torque) == 1:
                    max_speed_torque = float(max_speed_torque[0])
                elif float(max_speed_torque[1]) < 1000:
                    if float(max_speed_torque[0]) > 1000:
                        max_speed_torque = float(max_speed_torque[0]) + float(max_speed_torque[1])
                    else:
                        max_speed_torque = float(max_speed_torque[0] + max_speed_torque[1])
                else:
                    max_speed_torque = float(max_speed_torque[1])
                max_speed_torque_list.append(max_speed_torque)
            else:
                max_speed_torque_list.append(None)
                torque_list.append(torque)
        except:
            max_speed_torque_list.append(None)
            torque_list.append(None)
            pass
    df_test['torque'] = torque_list
    df_test['max_torque_rpm'] = max_speed_torque_list
    cat_features_mask_test = (df_test.dtypes == "object").values  # категориальные признаки имеют тип "object"
    df_test_real = df_test[df_test.columns[~cat_features_mask_test]]
    mis_replacer = impute.SimpleImputer(strategy="median")
    df_test_no_mis_real = pd.DataFrame(data=mis_replacer.fit_transform(df_test_real), columns=df_test_real.columns)
    df_test_cat = df_test[df_test.columns[cat_features_mask_test]].fillna("")
    df_test_cat.reset_index(drop=True, inplace=True)
    df_test_no_mis = pd.concat([df_test_no_mis_real, df_test_cat], axis=1)
    df_test_no_mis['engine'] = df_test_no_mis['engine'].astype('int')
    df_test_no_mis['seats'] = df_test_no_mis['seats'].astype('int')
    X_test_cat = df_test_no_mis.drop(['selling_price', 'name'], axis=1)
    transformer = make_column_transformer(
        (OneHotEncoder(), ['fuel', 'seller_type', 'transmission', 'owner', 'seats']),
        remainder='passthrough')

    transformed = transformer.fit_transform(X_test_cat)
    transformed_df = pd.DataFrame(transformed, columns=transformer.get_feature_names_out())
    list_train_keys = ['onehotencoder__fuel_CNG',
     'onehotencoder__fuel_Diesel',
     'onehotencoder__fuel_LPG',
     'onehotencoder__fuel_Petrol',
     'onehotencoder__seller_type_Dealer',
     'onehotencoder__seller_type_Individual',
     'onehotencoder__seller_type_Trustmark Dealer',
     'onehotencoder__transmission_Automatic',
     'onehotencoder__transmission_Manual',
     'onehotencoder__owner_First Owner',
     'onehotencoder__owner_Fourth & Above Owner',
     'onehotencoder__owner_Second Owner',
     'onehotencoder__owner_Test Drive Car',
     'onehotencoder__owner_Third Owner',
     'onehotencoder__seats_2',
     'onehotencoder__seats_4',
     'onehotencoder__seats_5',
     'onehotencoder__seats_6',
     'onehotencoder__seats_7',
     'onehotencoder__seats_8',
     'onehotencoder__seats_9',
     'onehotencoder__seats_10',
     'onehotencoder__seats_14',
     'remainder__year',
     'remainder__km_driven',
     'remainder__mileage',
     'remainder__engine',
     'remainder__max_power',
     'remainder__torque',
     'remainder__max_torque_rpm']
    X_test_data = transformed_df
    diff = list(set(list_train_keys) - set(list(X_test_data.keys())))
    for i in diff:
        X_test_data[i] = np.zeros(X_test_data.shape[0])
    X_test_data = X_test_data[list_train_keys]  # выравниваем
    return X_test_data

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    # load saved pipeline
    pipeline = joblib.load('pipe.joblib')
    list = [item]
    # convert to formatted data
    X_test_data = prepare_test_data(list)
    pred = pipeline.predict(X_test_data)
    print('prediction',pred[0])
    print('real value',dict(item)['selling_price'])
    return pred[0]

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    pipeline = joblib.load('pipe.joblib')
    X_test_data = prepare_test_data(items)
    pred = pipeline.predict(X_test_data)
    print(pred)
    return list(pred)
