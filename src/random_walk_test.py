# %%
import sys
import os
from pathlib import Path

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from config.const import DIR_PROCESSED_DATA, DIR_RESULTS

import numpy as np
import pandas as pd

import math
import scipy.stats as sps

import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')



# %%
usdcup_buy = pd.read_pickle(DIR_PROCESSED_DATA / "usdcup_time_serie_buy.pickle")

usdcup_sell = pd.read_pickle(DIR_PROCESSED_DATA / "usdcup_time_serie_sell.pickle")

blue_dollar_buy = pd.read_pickle(DIR_PROCESSED_DATA / "blue_dollar_time_serie_buy.pickle")
blue_dollar_sell = pd.read_pickle(DIR_PROCESSED_DATA / "blue_dollar_time_serie_sell.pickle")

usdeur = pd.read_pickle(DIR_PROCESSED_DATA / "EURUSD.pickle")


# %%
def runsTest(serie, serie_median=None):
    """
    Perform a Runs Test for randomness relative to the median.
    
    Parameters:
    serie (array-like): Numeric time series.

    Returns:
    tuple: (Z-statistic, p-value, number_of_runs, n1, n2)
    """

    if serie_median is None:
        serie_median = np.median(serie)

    # Transformo en binaria + o -
    signs = [1 if x >= serie_median else 0 for x in serie]

    # Contar runs
    runs = 1
    for i in range(1, len(signs)):
        if signs[i] != signs[i-1]:
            runs += 1

    n1 = signs.count(1)
    n0 = signs.count(0)

    # Valores esperados
    runs_exp = (2 * n1 * n0) / (n1 + n0) + 1
    std = math.sqrt((2 * n1 * n0 * (2 * n1 * n0 - n1 - n0)) / (((n1+n0)**2) * (n1+n0 - 1)))
    
    z = (runs - runs_exp) / std
    pvalue = 2 * (1 - sps.norm.cdf(abs(z)))

    return z, pvalue, runs, n1, n0


def Variance_Ratio_Test(data, k=5):
    log_data = np.log(data)
    rets = np.diff(log_data)
    T = len(rets)

    mu = np.mean(rets)
    var_1 = np.var(rets, ddof=1, dtype=np.float64)

    rets_k = (log_data - np.roll(log_data, k))[k:]
    m = k * (T - k + 1) * (1 - k/T)
    var_k = (1/m) * np.sum((rets_k - k*mu)**2)

    # Variance Ratio
    VR = var_k / var_1

    # --- Z-test homoscedástico (Lo-MacKinlay clásico) ---
    phi1 = 2*(2*k-1)*(k-1) / (3*k*T)
    Z_homoskedastic = (VR - 1) / np.sqrt(phi1)

    # --- Z-test heteroscedástico robusto (Lo-MacKinlay HAC) ---
    def delta(j):
        num = np.sum((rets[j:] - mu)*(rets[:-j] - mu))**2
        den = ((T-j)*var_1)**2
        return num/den

    phi2 = np.sum([((2*(k-j)/k)**2) * delta(j) for j in range(1, k)])
    Z_hetero = (VR - 1) / np.sqrt(phi2) if phi2 != 0 else np.nan

    return VR, Z_homoskedastic, Z_hetero



# %%

results = {
    "Serie": ["USD/CUP Sell", "Blue Dollar Sell", "EUR/USD"],
    "Z-stat": [],
    "p-value": [],
    "Runs": [],
    "n+ (>= mediana)": [],
    "n- (< mediana)": []
}

series_list = [
    np.diff(usdcup_sell['USDCUP_Sell'].values),
    np.diff(blue_dollar_sell['Dollar_Blue_SELL'].values),
    np.diff(usdeur['EURUSD'].values)
]

for serie in series_list:
    z, p, r, n1, n0 = runsTest(serie)
    results["Z-stat"].append(round(z, 4))
    results["p-value"].append(round(p, 4))
    results["Runs"].append(r)
    results["n+ (>= mediana)"].append(n1)
    results["n- (< mediana)"].append(n0)

# Convertimos a DataFrame para output limpio y formateado
df_results = pd.DataFrame(results)
print("\n🔥 Test de Corridas de Wald-Wolfowitz (sobre diferencias de precio)\n")
print(df_results)



# %%
results_VR = {
    "Serie": ["USD/CUP Sell", "Blue Dollar Sell", "EUR/USD"],
    "VR(k=5)": [],
    "Z-Homoskedastic": [],
    "Z-Heteroskedastic": [],
    "p-value Homo": [],
    "p-value Hetero": []
}

for serie, name in zip(
    [usdcup_sell['USDCUP_Sell'].values,
     blue_dollar_sell['Dollar_Blue_SELL'].values,
     usdeur['EURUSD'].values],
    results_VR["Serie"]
):
    vr, Z_h, Z_hc = Variance_Ratio_Test(serie, k=5)
    results_VR["VR(k=5)"].append(round(vr,4))
    results_VR["Z-Homoskedastic"].append(round(Z_h,4))
    results_VR["Z-Heteroskedastic"].append(round(Z_hc,4))
    results_VR["p-value Homo"].append(round(2*(1 - sps.norm.cdf(abs(Z_h))),4))
    results_VR["p-value Hetero"].append(round(2*(1 - sps.norm.cdf(abs(Z_hc))),4))

df_vr = pd.DataFrame(results_VR)
print("\n📈 Test Lo-MacKinlay Variance Ratio (k=5)\n")
print(df_vr)

# %%
with open(DIR_RESULTS / "runs_test_results.txt", "w", encoding="utf-8") as f:
    f.write("Test de Corridas de Wald-Wolfowitz (sobre diferencias de precio)\n\n")
    f.write(df_results.to_string(index=False))
    f.write("\n\n")

# Guardar resultados del Variance Ratio Test
with open(DIR_RESULTS / "variance_ratio_results.txt", "w", encoding="utf-8") as f:
    f.write("Test Lo-MacKinlay Variance Ratio (k=5)\n\n")
    f.write(df_vr.to_string(index=False))
