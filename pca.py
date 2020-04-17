from datasetreader import get_data

from sklearn.decomposition import PCA
import pandas as pd

X = get_data()

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

principalDf.to_csv('./data/pca_data.csv', sep=',')