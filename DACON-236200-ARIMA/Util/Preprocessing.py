import pandas as pd

def date_preprocessing(df):
    df.index = pd.DatetimeIndex(df['일시']).to_period('D')
    df.drop('일시', axis=1, inplace=True)
