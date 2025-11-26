# %%
import sys
import os
from pathlib import Path

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from config.const import DIR_PROCESSED_DATA, DIR_FIGURES

import numpy as np
import pandas as pd
from hmmlearn import hmm

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
def hidden_MM(time, serie, currency, num_states, n_iter):

    data_frame = pd.DataFrame({'Time':time, 'Currency':serie[currency], 'Returns':serie[currency].diff()})
    data_frame = data_frame.dropna()
    returns = data_frame[['Returns']].values

    model = hmm.GaussianHMM(
        n_components=num_states,
        covariance_type = "full",
        n_iter = n_iter,
        random_state = 42,
        algorithm='map'
    )
    model.fit(returns)

    Z = model.predict(returns)

    states = pd.unique(Z)

    data_frame['States'] = Z

    print("Unique states:")
    print(states)
    print("\nStart probabilities:")
    print(model.startprob_)
    print("\nTransition matrix:")
    print(model.transmat_)
    print("\nGaussian distribution means:")
    print(model.means_)
    print("\nGaussian distribution covariances:")
    print(model.covars_)

    plt.figure(figsize = (15, 10))
    plt.subplot(2,1,1)
    for i in states:
        want = (Z == i)
        x = data_frame['Time'].iloc[want]
        y = data_frame['Currency'].iloc[want]
        plt.plot(x, y, '.')
    plt.legend(states, fontsize=16)
    plt.grid(True)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel(currency, fontsize=16)

    plt.subplot(2,1,2)
    for i in states:
        want = (Z == i)
        x = data_frame['Time'].iloc[want]
        y = data_frame['Returns'].iloc[want]
        plt.plot(x, y, '.')
    plt.legend(states, fontsize=16)
    plt.grid(True)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlabel("Time", fontsize=16)
    plt.ylabel(f"Return {currency}" , fontsize=16)
    plt.savefig(DIR_FIGURES / f"hmm_{currency}.pdf", bbox_inches='tight')
    plt.show()

    data_frame.reset_index(inplace=True)
    # del data_frame['index']

    return data_frame




# %%
df_states = hidden_MM(time = usdcup_sell.index, serie=usdcup_sell, currency="USDCUP_Sell", num_states=2, n_iter=50)


# %%
df_states = hidden_MM(time = usdcup_buy.index, serie=usdcup_buy, currency="USDCUP_Buy", num_states=2, n_iter=50)

# %%
df_states = hidden_MM(time = blue_dollar_sell.index, serie=blue_dollar_sell, currency="Dollar_Blue_SELL", num_states=2, n_iter=50)


# %%
df_states = hidden_MM(time = eur.index, serie=eur, currency="EURUSD", num_states=2, n_iter=50)
