import pandas as pd

def date_preprocessing(df):
    df['time'] = df['date_time'].apply(lambda x: int(x[-2:]))
    df['weekday'] = df['date_time'].apply(lambda x: pd.to_datetime(x[:10]).weekday())
    df.drop('date_time', axis=1, inplace=True)

def export_preprocessing(df):
    export_df = pd.DataFrame(columns=['비전기냉방설비운영', '태양광보유'])

    for idx in range(0, len(df)):
        export_idx = df.loc[idx, 'num']
        export_df.loc[export_idx, '비전기냉방설비운영'] = 1.0 if df.loc[idx, '비전기냉방설비운영'] == 1.0 else 0.0
        export_df.loc[export_idx, '태양광보유'] = 1.0 if df.loc[idx, '태양광보유'] == 1.0 else 0.0

    export_df.to_pickle('./Output/encoder.pkl')

def import_preprocessing(df):
    import_df = pd.read_pickle('./Output/encoder.pkl')

    for idx in range(len(df)):
        df.loc[idx, '비전기냉방설비운영'] = import_df.loc[df['num'][idx], '비전기냉방설비운영']
        df.loc[idx, '태양광보유'] = import_df.loc[df['num'][idx], '태양광보유']
