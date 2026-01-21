#!/usr/bin/env python3
"""
Load multiple saved TCL2 population data files from results/ and plot them together.

Usage:
    python plot_results.py
        -> plots all results/TCL2_pop_*.npy

    python plot_results.py results/file1.npy results/file2.npy ...
        -> plots exactly the provided files
"""

import os
import re
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt


def extract_label(path: str) -> str:
    """Make a nice label from the filename, e.g. TCL2_pop_TD3_w1_gamma0.2.npy -> TD3"""
    base = os.path.basename(path)
    m = re.search(r"TD\d+(?:\.\d+)?", base)
    if m:
        return m.group(0)
    # fallback: filename without extension
    return os.path.splitext(base)[0]


def load_one(path: str):
    """Return (times, pop) from supported npy formats."""
    data = np.load(path, allow_pickle=True)

    # dict saved via np.save(..., {...})
    if isinstance(data, np.ndarray) and data.dtype == object:
        d = data.item()
        times = d.get("times", None)
        pop = d.get("pop", None)
        if times is None or pop is None:
            raise ValueError(f"{path}: dict must contain keys 'times' and 'pop'")
        return np.asarray(times, float), np.asarray(pop)

    # plain array saved via np.save(..., pop)
    pop = np.asarray(data)
    times = np.arange(len(pop), dtype=float)
    return times, pop


def main():
    # Collect files
    if len(sys.argv) > 1:
        files = sys.argv[1:]
    else:
        
        
        files = sorted(files, key=m)

        if not files:
            raise FileNotFoundError("No files found: results/TCL2_pop_*.npy")

    # Plot
    plt.figure()
    for path in files:
        t, pop = load_one(path)
        label = extract_label(path)
        plt.plot(t, np.real(pop), marker="", label=label)

    plt.xlabel("t")
    plt.ylabel(r"$\langle P_{11} \rangle$")
    plt.title("TCL2 population (multiple runs)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.save()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import re

def extract_td_value(path):
    m = re.search(r"TD(\d+(?:\.\d+)?)", os.path.basename(path))
    return float(m.group(1))

def extract_label(path):
    return f"TD={extract_td_value(path)}"

files = glob.glob("results/TCL2_pop_*.npy")
files = sorted(files, key=extract_td_value)

for f in files:
    data = np.load(f, allow_pickle=True).item()
    plt.plot(data["times"], data["pop"], label=extract_label(f))

plt.legend()
plt.xlabel("t")
plt.ylabel(r"$\langle P_{11} \rangle$")
plt.grid(True)
plt.savefig('giantat.png')
plt.show()

if __name__ == "__main__":
    main()
