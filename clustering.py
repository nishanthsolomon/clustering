import configparser

from datasetreader import read_data


class Clustering():
    def __init__(self, config):
        data_config = config['data']
        data_path = data_config['data_path']
        self.data = read_data(data_path)
              
    def run_kmeans(self):
        pass
    
    def run_gmm(self):
        pass

    def plot_results(self):
        pass


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read('./clustering.conf')

    clustering = Clustering(config)