# %%
import sys
import os
from pathlib import Path

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from config.const import DIR_PROCESSED_DATA, DIR_FIGURES

import numpy as np
import pandas as pd
import emd

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

# %%
usdcup_buy = pd.read_pickle(DIR_PROCESSED_DATA / "usdcup_time_serie_buy.pickle")

usdcup_sell = pd.read_pickle(DIR_PROCESSED_DATA / "usdcup_time_serie_sell.pickle")

blue_dollar_buy = pd.read_pickle(DIR_PROCESSED_DATA / "blue_dollar_time_serie_buy.pickle")
blue_dollar_sell = pd.read_pickle(DIR_PROCESSED_DATA / "blue_dollar_time_serie_sell.pickle")


eur = pd.read_pickle(DIR_PROCESSED_DATA / "EURUSD.pickle")


# %%

def get_imfs(serie):
    imf = emd.sift.sift(serie.values)
    nIMFs = imf.shape[1]
    
    return imf.T



def plot_emd(serie, exchange_rate):
    imfs = get_imfs(serie)


    plt.figure(figsize=(15, 18))
    plt.subplot(len(imfs) + 1, 1, 1)
    plt.plot(serie.index, serie[exchange_rate].values, "r")
    plt.title(f"{exchange_rate}")
    plt.ylabel("Original Serie", fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)

    for n, imf_n in enumerate(imfs):
        plt.subplot(len(imfs) + 1, 1, n + 2)
        plt.plot(serie[exchange_rate].index, imf_n, "b")
        plt.ylabel("r" if n == len(imfs)-1 else f"IMF{n+1}", fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True)

    plt.tight_layout()
    plt.grid(True)
    plt.savefig(DIR_FIGURES / f"emd_{exchange_rate}.pdf", bbox_inches='tight')
    plt.show()



# %%
plot_emd(usdcup_buy, 'USDCUP_Buy')

# %%
plot_emd(blue_dollar_buy, 'Dollar_Blue_BUY')

# %%
plot_emd(eur, 'EURUSD')
