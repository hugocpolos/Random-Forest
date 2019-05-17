import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_to_png(x_values, y_values, filename):

    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, 'o-')
    plt.savefig(filename)