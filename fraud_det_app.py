# Importing dependencies
import uvicorn
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# loading the trained model
trained_model = 'artifacts/models/logistic_reg.pkl'

model = joblib.load((trained_model))


class FraudPred(BaseModel):
    OldBalance: float
    Amount: float
    NewBalance: float
    TransactionType: int
    TransactionHour: int


@app.get('/')
def index():
    return {'message': 'Fraud Prediction App'}

# Define the function, which will make prediction
# using the input data provided by the user


@app.post('/predict')
def predict_fraud_status(fraud_details: FraudPred):
    data = fraud_details.dict()
    oldbalanceOrg = data['OldBalance']
    amount = data['Amount']
    newbalanceOrig = data['NewBalance']
    trans_type = data['TransactionType']
    step = data['TransactionHour']


# Make predictions
    prediction = model.predict([[oldbalanceOrg,
                                amount, newbalanceOrig,
                                trans_type, step]])

    if prediction == 0:
        pred = 'Not Fraud'
    else:
        pred = 'Fraud'

    return {'status': pred}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
