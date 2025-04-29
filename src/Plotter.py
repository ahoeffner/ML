import matplotlib.pyplot as plt

class Plotter:
    def __init__(self):
        pass


    def scatter_plot(self, x, y, title, x_axis, y_axis, s = None):
        plt.title(title)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        if s is None : plt.scatter(x, y)
        else: plt.scatter(x, y, s)



    def frequency_histogram(self, x, bins, title, x_axis, y_axis):
        plt.hist(x, bins = bins, edgecolor = 'black', alpha = 0.7)
        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.title(title)
        plt.xticks(bins)  # Ensure ticks are placed correctly
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()