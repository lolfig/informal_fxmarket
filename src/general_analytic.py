# %%
import sys
import os
from pathlib import Path

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from config.const import DIR_PROCESSED_DATA

import math
import numpy as np
import pandas as pd
import emd
from hmmlearn import hmm
import scipy.stats as sps

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')

from statsmodels.graphics.tsaplots import plot_acf




# %%
usdcup = pd.read_pickle(DIR_PROCESSED_DATA / "all_info_USDCUP.pickle")
serie_cup = pd.Series(usdcup['Venta']['avg'].values, index = pd.to_datetime(usdcup['Venta']['_id'].values))
serie_cup = serie_cup.loc["2021-08-01":"2024-03-27"]
serie_cup = pd.DataFrame(serie_cup, columns=['USDCUPVenta'])
log_returns_cup = np.log(serie_cup).diff()
log_returns_cup = log_returns_cup['USDCUPVenta'].dropna()
log_returns_cup_no_zero = log_returns_cup[log_returns_cup != 0]



blue = pd.read_csv(DIR_PROCESSED_DATA / "blue_dollar_daily_INTERPOLATED_30D_2021-07-01_to_2024-03-27.csv")
blue.index = pd.to_datetime(blue['date'])
serie_blue = blue[['venta']]
serie_blue.columns = ['Dollar_Blue']


EURUSD = pd.read_pickle(DIR_PROCESSED_DATA / "EURUSD.pickle")
log_returns_EURUSD = np.log(EURUSD).diff()
log_returns_EURUSD = log_returns_EURUSD['EURUSD'].dropna()
log_returns_EURUSD_no_zero = log_returns_EURUSD[log_returns_EURUSD != 0]

serie_blue = serie_blue.loc["2021-08-01":]
log_returns_blue = np.log(serie_blue).diff()
log_returns_blue = log_returns_blue['Dollar_Blue'].dropna()
log_returns_blue_no_zero = log_returns_blue[log_returns_blue != 0]



serie_cup = serie_cup.loc["2021-08-01":"2024-03-27"]
log_returns_cup = np.log(serie_cup).diff()
log_returns_cup = log_returns_cup['USDCUPVenta'].dropna()
log_returns_cup_no_zero = log_returns_cup[log_returns_cup != 0]

time = serie_blue.index


# %%
fig, ax1 = plt.subplots(figsize=(10, 4))

ax1.plot(serie_blue, color='blue', label='Blue')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Blue', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(serie_cup, color='red', label='USD/CUP')
ax2.set_ylabel('USD/CUP', color='red')
ax2.tick_params(axis='y', labelcolor='red')

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.show()



# %%
plt.figure(figsize=(10, 6))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(serie_blue, label='Blue Dollar')
ax1.legend()
ax2.plot(log_returns_blue, color='orange', label='log Returns')
ax2.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(serie_cup, label='USDCUP', color='red')
ax1.legend()
ax2.plot(log_returns_cup, color='orange', label='log Returns')
ax2.legend()
plt.tight_layout()
plt.show()



# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

ax1.hist(log_returns_cup_no_zero, bins=15)
ax1.set_xlabel('Log Returns')
ax1.set_ylabel('Density')
ax1.set_title('USDCUP')

ax2.hist(log_returns_blue_no_zero, bins=15)
ax2.set_xlabel('Log Returns')
ax2.set_ylabel('Density')
ax2.set_title('Dollar Blue')

ax3.hist(log_returns_EURUSD, bins=15)
ax3.set_xlabel('Log Returns')
ax3.set_ylabel('Density')
ax3.set_title('EURUSD')

plt.tight_layout()
plt.show()

# %%
fig, axes = plt.subplots(1, 3, figsize=(10, 4))  # 3 filas, 1 columna

xlim = (0.5, 25)
ylim = (-0.15, 0.4)
lags = 50

plot_acf(np.diff(serie_cup['USDCUPVenta']), lags=lags, ax=axes[0], auto_ylims=False)
axes[0].set_xlim(xlim)
axes[0].set_ylim(ylim)
axes[0].set_title('ACF: USDCUP Log Returns')

plot_acf(log_returns_blue_no_zero, lags=lags, ax=axes[1], auto_ylims=False)
axes[1].set_xlim(xlim)
axes[1].set_ylim(ylim)
axes[1].set_title('ACF: Dollar Blue Log Returns')


plot_acf(log_returns_EURUSD_no_zero, lags=lags, ax=axes[2], auto_ylims=False)
axes[2].set_xlim(xlim)
axes[2].set_ylim(ylim)
axes[2].set_title('ACF: EUR/USD Log Returns')

plt.tight_layout()
plt.show()



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
    plt.show()

    data_frame.reset_index(inplace=True)
    # del data_frame['index']

    return data_frame


# %%
df_states = hidden_MM(time = time, serie=serie_blue, currency="Dollar_Blue", num_states=3, n_iter=50)

# %%
df_states = hidden_MM(time = serie_cup.index, serie=serie_cup, currency="USDCUPVenta", num_states=2, n_iter=50)

# %%
df_states = hidden_MM(time = time, serie=EURUSD, currency="EURUSD", num_states=2, n_iter=50)


# %%
def get_imfs(serie):
    imf = emd.sift.sift(serie.values)
    nIMFs = imf.shape[1]
    
    return imf.T

# %%
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
    plt.show()

# %%
plot_emd(serie_blue, 'Dollar_Blue')




# %%
def runsTest(serie, serie_median):
    """
    Perform a runs test on a series of data.

    Parameters:
    series (pandas.Series): The series of data.

    Returns:
    tuple: A tuple containing the result string, z-statistic, and p-value.
    """
    runs, n1, n2 = 0, 0, 0

    # Checking for start of new run
    for i in range(len(serie)):

        # no. of runs
        if (serie[i] >= serie_median and serie[i-1] < serie_median) or \
                (serie[i] < serie_median and serie[i-1] >= serie_median):
            runs += 1  

        # no. of positive values
        if(serie[i]) >= serie_median:
            n1 += 1   

        # no. of negative values
        else:
            n2 += 1   

    runs_exp = ((2*n1*n2)/(n1+n2))+1
    stan_dev = math.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/ \
                       (((n1+n2)**2)*(n1+n2-1)))

    z = abs((runs-runs_exp)/stan_dev)

    pvalue = (2*(1 - sps.norm.cdf(abs(z))))

    result = (f'Z_stat = {z} ----- P_value = {pvalue}')    

    return result, z, pvalue


def Variance_Ratio_Test(data, k):

    log_data = np.log(data)
    rets = np.diff(log_data)
    T = len(rets)
    mu = np.mean(rets)
    var_1 = np.var(rets, ddof=1, dtype=np.float64)
    rets_k = (log_data - np.roll(log_data, k))[k:]
    m = k * (T - k + 1) * (1 - k / T)
    var_k = 1/m * np.sum(np.square(rets_k - k * mu))

    # Varianza
    vr = var_k / var_1
    # Phi1
    phi1 = 2 * (2*k - 1) * (k-1) / (3*k*T)
    # Phi2

    def delta(j):
        res = 0
        for t in range(j+1, T+1):
            t -= 1  # el índice de la matriz es t-1 para el elemento t-th
            res += np.square((rets[t]-mu)*(rets[t-j]-mu))
        return res / ((T-1) * var_1)**2

    phi2 = 0

    for j in range(1, k):
        phi2 += (2*(k-j)/k)**2 * delta(j)

    return vr, (vr - 1) / np.sqrt(phi1), (vr - 1) / np.sqrt(phi2)

k = 5


# %%
print(np.median(np.diff(serie_blue['Dollar_Blue'].values)))


# %%
print(f'Run test Dollar_Blue: {runsTest(np.diff(serie_blue['Dollar_Blue'].values), np.median(np.diff(serie_blue['Dollar_Blue'].values)))}')

print(f'Variance Ratio Test Dollar_Blue: {Variance_Ratio_Test(serie_blue["Dollar_Blue"], k = k)}')


print('='*100)

print(f'Run test USD CUP: {runsTest(np.diff(serie_cup['USDCUPVenta'].values), np.median(np.diff(serie_blue['Dollar_Blue'].values)))}')

print(f'Variance Ratio Test USD CUP: {Variance_Ratio_Test(serie_cup["USDCUPVenta"], k = k)}')

# %%
print('='*100)

print(f'Run test EURUSD: {runsTest(np.diff(EURUSD['EURUSD'].values), np.median(np.diff(EURUSD['EURUSD'].values)))}')

print(f'Variance Ratio Test EURUSD: {Variance_Ratio_Test(EURUSD["EURUSD"], k = k)}')