from typing import List
from matplotlib import pyplot as plt


def plot_loss(tlosses: List[float], vlosses: List[float], filename: str) -> None:
    plt.plot(tlosses, label="Training loss")
    plt.plot(vlosses, label="Validation loss")
    plt.legend(frameon=False)
    plt.savefig(filename)
