import gvar as gv
import matplotlib.pyplot as plt

class Diagnostic(object):

    def __init__(self, data):
        self.data = data
        self._svd_diagnosis = gv.dataset.svd_diagnosis(self.data)
    

    def calculate_svdcut(self):
        return self._svd_diagnosis.svdcut
    

    def plot_svdcut(self):
        fig = plt.figure()
        self._svd_diagnosis.plot_ratio(show=False)
        fig = plt.gcf()
        plt.close()
        return fig