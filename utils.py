import pandas as pd

def preprocess_data(df):
    df.fillna(0, inplace=True)
    df['TotalSum'] = df['Quantity'] * df['UnitPrice']
    return df
