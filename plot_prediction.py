from datasetreader import get_pca_data
import pandas as pd
import matplotlib.pyplot as plt


def plot_prediction(predictions, num_clusters):
    principalDf = pd.DataFrame(data=get_pca_data(), columns=[
        'principal component 1', 'principal component 2'])

    finalDf = pd.concat([principalDf, pd.DataFrame(
        predictions, columns=['target'])], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('GMM - ' + str(num_clusters), fontsize=20)

    if (num_clusters == 3):
        targets = [1, 2, 3]
        colors = ['r', 'g', 'b']
    elif (num_clusters == 5):
        targets = [1, 2, 3, 4, 5]
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'black', 'yellow']
    elif (num_clusters == 7):
        targets = [1, 2, 3, 4, 5, 6, 7]
        colors = ['red', 'blue', 'green', 'cyan', 'magenta', 'black', 'yellow']

    for target, color in zip(targets, colors):
        indicesToKeep = finalDf['target'] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
                   finalDf.loc[indicesToKeep, 'principal component 2'], c=color, s=50)
    ax.legend(targets)
    ax.grid()
    plt.show()
