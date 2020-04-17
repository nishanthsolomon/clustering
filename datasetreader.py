import pandas as pd


def get_data():
    data = pd.read_csv('./data/data.csv')

    return data.values


def get_pca_data():
    pca_data = pd.read_csv('./data/pca_data.csv')

    return pca_data.values
