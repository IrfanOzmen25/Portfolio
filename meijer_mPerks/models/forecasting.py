import pandas as pd
import xgboost as xgb
from prophet import Prophet

def train_xgb(sales_df):
    feats = ['store_id','sku_id','week',...]
    X, y = sales_df[feats], sales_df['units_sold']
    model = xgb.XGBRegressor()
    model.fit(X, y)
    return model

def predict_prophet(sales_history, periods=12):
    df = sales_history.rename(columns={'week':'ds','units_sold':'y'})
    m = Prophet().fit(df)
    future = m.make_future_dataframe(periods=periods, freq='W')
    return m.predict(future)
