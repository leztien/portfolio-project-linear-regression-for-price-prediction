#!/usr/bin/env python3

"""
TODO
"""


from pathlib import Path
import matplotlib.pyplot as plt


def save_fig(name):
    """Saves the current figure"""
    path = Path() / "images"
    if not path.exists():
        path.mkdir()
    path /= f"{name}.png"
    plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    return str(path)
