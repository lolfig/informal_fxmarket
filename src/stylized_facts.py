# %%
import sys
import os
from pathlib import Path

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from config.const import DIR_PROCESSED_DATA, DIR_FIGURES

import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.stats import gaussian_kde
from scipy.stats import norm
import math


from statsmodels.graphics.tsaplots import plot_acf
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
def plot_log_return_acf(series_dict, lags=20, save_dir=None):

    for name, series in series_dict.items():

        log_ret = np.log(series / series.shift(1)).dropna()
        
        fig, ax = plt.subplots(figsize=(6,4))
        plot_acf(log_ret, lags=lags, ax=ax)
        ax.set_title(f'ACF of log-returns: {name}')
        ax.set_ylabel('Autocorrelation')
        ax.set_xlabel('Lag')
        plt.tight_layout()
        
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_dir / f"acf_logret_{name}.pdf", bbox_inches='tight')
        
        plt.show()


# %%
series_dict = {
    "usdcup_sell": usdcup_sell,
    "blue_dollar_sell": blue_dollar_sell,
    "eur": eur
}


# %%
plot_log_return_acf(series_dict, lags=20, save_dir=DIR_FIGURES)




# %%
log_ret_usdcup_sell = np.log(usdcup_sell / usdcup_sell.shift(1)).dropna()
log_ret_usdcup_sell = log_ret_usdcup_sell['USDCUP_Sell'].values


epsilon = 1e-14
zeros = np.abs(log_ret_usdcup_sell) < epsilon
non_zeros = log_ret_usdcup_sell[~zeros]



mu, sigma = norm.fit(non_zeros)

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(log_ret_usdcup_sell, bins=50, density=True, alpha=0.6, color='tab:blue', edgecolor='black')

# Superponer la gaussiana
x = np.linspace(non_zeros.min(), non_zeros.max(), 200)
ax.plot(x, norm.pdf(x, mu, sigma)*(1 - zeros.mean()), 'r-', lw=2, label='Gaussian fit')

# Marcar la delta en cero
ax.bar(0, zeros.mean(), width=0.0005, color='Green', label='Delta at 0')


ax.set_xlabel('Log-returns')
ax.set_ylabel('Density')
ax.set_title('Histogram with Gaussian + Delta')
ax.legend()
plt.tight_layout()
plt.show()


# %%
def plot_hist_gaussian_delta(log_returns, name='series', save_dir=None, bins=50, epsilon=1e-14):
    log_returns = np.asarray(log_returns).ravel()
    
    zeros = np.abs(log_returns) < epsilon
    non_zeros = log_returns[~zeros]
    
    mu, sigma = norm.fit(non_zeros)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(log_returns, bins=bins, density=True, alpha=0.6, color='tab:blue', edgecolor='black')
    
    # Gaussiana ajustada
    x = np.linspace(non_zeros.min(), non_zeros.max(), 200)
    ax.plot(x, norm.pdf(x, mu, sigma)*(1 - zeros.mean()), 'r-', lw=2, label='Gaussian fit')
    
    # Delta en cero
    ax.bar(0, zeros.mean(), width=0.0005, color='green', label='Delta at 0')
    
    ax.set_xlabel('Log-returns')
    ax.set_ylabel('Density')
    ax.set_title(f'Histogram with Gaussian + Delta: {name}')
    ax.legend()
    plt.tight_layout()
    
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / f"hist_{name}.pdf", bbox_inches='tight')
    
    plt.show()
# %%
log_ret_usdcup_sell = np.log(usdcup_sell / usdcup_sell.shift(1)).dropna()['USDCUP_Sell'].values

log_ret_blue_dollar_sell = np.log(blue_dollar_sell / blue_dollar_sell.shift(1)).dropna()['Dollar_Blue_SELL'].values

log_ret_eur = np.log(eur / eur.shift(1)).dropna()['EURUSD'].values

# %%
plot_hist_gaussian_delta(log_ret_usdcup_sell, name='USDCUP_Sell', save_dir=DIR_FIGURES)


# %%
plot_hist_gaussian_delta(log_ret_blue_dollar_sell, name='BlueDollar_Sell', save_dir=DIR_FIGURES)

# %%
plot_hist_gaussian_delta(log_ret_eur, name='EUR', save_dir=DIR_FIGURES)