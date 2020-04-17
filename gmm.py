from scipy.stats import multivariate_normal
import numpy as np
import matplotlib.pyplot as plt
from datasetreader import get_data
from plot_prediction import plot_prediction


class GMM:

    def __init__(self, X, number_of_sources, iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.cov = None
        self.XY = None

    def run(self):
        self.reg_cov = 1e-6*np.identity(len(self.X[0]))
        x, y = np.meshgrid(np.sort(self.X[:, 0]), np.sort(self.X[:, 1]))
        self.XY = np.array([x.flatten(), y.flatten()]).T

        """ 1. Set the initial mu, covariance and pi values"""
        self.mu = np.random.randint(min(self.X[:, 0]), max(self.X[:, 0]), size=(self.number_of_sources, len(
            self.X[0])))
        self.cov = np.zeros((self.number_of_sources, len(self.X[0]), len(self.X[0])))
        for dim in range(len(self.cov)):
            np.fill_diagonal(self.cov[dim], 5)

        self.pi = np.ones(self.number_of_sources) / \
            self.number_of_sources
        log_likelihoods = []

        for _ in range(self.iterations):

            """E Step"""
            r_ic = np.zeros((len(self.X), len(self.cov)))

            for m, co, p, r in zip(self.mu, self.cov, self.pi, range(len(r_ic[0]))):
                co += self.reg_cov
                mn = multivariate_normal(mean=m, cov=co)
                r_ic[:, r] = p*mn.pdf(self.X)/np.sum([pi_c*multivariate_normal(mean=mu_c, cov=cov_c).pdf(
                    self.X) for pi_c, mu_c, cov_c in zip(self.pi, self.mu, self.cov+self.reg_cov)], axis=0)

            self.mu = []
            self.cov = []
            self.pi = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:, c], axis=0)
                mu_c = (1/m_c)*np.sum(self.X *
                                      r_ic[:, c].reshape(len(self.X), 1), axis=0)
                self.mu.append(mu_c)

                self.cov.append(((1/m_c)*np.dot((np.array(r_ic[:, c]).reshape(
                    len(self.X), 1)*(self.X-mu_c)).T, (self.X-mu_c)))+self.reg_cov)

                self.pi.append(m_c/np.sum(r_ic))

            log_likelihoods.append(np.log(np.sum([k*multivariate_normal(self.mu[i], self.cov[j]).pdf(
                self.X) for k, i, j in zip(self.pi, range(len(self.mu)), range(len(self.cov)))])))

        fig2 = plt.figure(figsize=(10, 10))
        ax1 = fig2.add_subplot(111)
        ax1.set_title('Number of Iterations vs Log-Likelihood for GMM - ' + str(self.number_of_sources))
        ax1.plot(range(0, self.iterations, 1), log_likelihoods)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Log-Likelihood')
        plt.show()

    """Predict the membership of an unseen, new datapoint"""

    def predict(self, Y):
        prediction = []
        for m, c in zip(self.mu, self.cov):
            prediction.append(multivariate_normal(mean=m, cov=c).pdf(
                Y)/np.sum([multivariate_normal(mean=mean, cov=cov).pdf(Y) for mean, cov in zip(self.mu, self.cov)]))

        return (prediction.index(max(prediction)) + 1)


def run_gmm(num_clusters):
    X = get_data()
    gmm = GMM(X, num_clusters, 50)
    gmm.run()

    predictions = []
    for x in X:
        pred = gmm.predict([x])
        predictions.append(pred)

    plot_prediction(predictions, num_clusters)

if __name__ == "__main__":
    run_gmm(3)
    run_gmm(5)
    run_gmm(7)