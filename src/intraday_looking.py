# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science'])


# %%
data = pd.read_csv(r"C:\Users\agarc\Projects\argentina_informal_fxmarket\data\usdt_ars_intraday.csv")

# %%
data



# %%
plt.figure()
data['available_amount'].hist()
plt.yscale('log')
plt.show()

