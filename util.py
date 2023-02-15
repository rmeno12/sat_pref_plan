from matplotlib import pyplot as plt


def plot_loss(tlosses, vlosses, filename):
    plt.plot(tlosses, label="Training loss")
    plt.plot(vlosses, label="Validation loss")
    plt.legend(frameon=False)
    plt.savefig(filename)
