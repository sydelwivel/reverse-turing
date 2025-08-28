# utils.py
import json

def read_csv_to_dicts(fp):
    import pandas as pd
    df = pd.read_csv(fp)
    return df.to_dict(orient='records')
