import pandas as pd


def read_data(path):
    data = pd.read_csv(path)
    
    return data.values