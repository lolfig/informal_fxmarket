# %%
import sys
import os
from pathlib import Path

project_root = Path().resolve().parent
sys.path.insert(0, str(project_root))

from config.const import DIR_PROCESSED_DATA


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# %%
usdcup = pd.read_pickle(DIR_PROCESSED_DATA / "all_info_USDCUP.pickle")

date_range = pd.date_range(usdcup['Venta']['_id'].min(), usdcup['Venta']['_id'].max())

# %%
serie_buy = pd.DataFrame(
    {
        'USDCUP_Buy':usdcup['Compra']['avg'].values
    },
    index=usdcup['Compra']['_id'].values
)

serie_buy = serie_buy[~serie_buy.index.duplicated(keep='last')]

serie_buy_complete = serie_buy.reindex(date_range)


serie_buy_complete['USDCUP_Buy'] = serie_buy_complete['USDCUP_Buy'].interpolate(method='linear')


serie_buy_complete.to_pickle(DIR_PROCESSED_DATA / "usdcup_time_serie_buy.pickle")






# %%
serie_sell = pd.DataFrame(
    {
        'USDCUP_Sell':usdcup['Venta']['avg'].values
    },
    index=usdcup['Venta']['_id'].values
)

serie_sell = serie_sell[~serie_sell.index.duplicated(keep='last')]

serie_sell_complete = serie_sell.reindex(date_range)


serie_sell_complete['USDCUP_Sell'] = serie_sell_complete['USDCUP_Sell'].interpolate(method='linear')


serie_sell_complete.to_pickle(DIR_PROCESSED_DATA / "usdcup_time_serie_sell.pickle")


# %%
blue = pd.read_csv(DIR_PROCESSED_DATA / "blue_dollar_daily_INTERPOLATED_30D_2021-07-01_to_2024-03-27.csv")
blue.index = pd.to_datetime(blue['date'])
serie_blue_SELL = blue[['venta']]
serie_blue_SELL.columns = ['Dollar_Blue_SELL']

serie_blue_BUY = blue[['compra']]
serie_blue_BUY.columns = ['Dollar_Blue_BUY']



# %%
serie_blue_SELL.to_pickle(DIR_PROCESSED_DATA / "blue_dollar_time_serie_sell.pickle")
serie_blue_BUY.to_pickle(DIR_PROCESSED_DATA / "blue_dollar_time_serie_buy.pickle")