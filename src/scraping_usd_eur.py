# %%
import yfinance as yf
import pandas as pd



# %%
start_date = "2021-08-01"
end_date = "2024-03-27"

eurusd = yf.download("EURUSD=X", start=start_date, end=end_date, interval="1d")

# %%
serie = pd.Series(eurusd['Close'].values.reshape(-1), index=eurusd.index)

# %%
full_date_range = pd.date_range(start="2021-08-01", end="2024-03-27", freq='D')

serie_full = serie.reindex(full_date_range)

serie_full = serie_full.interpolate(method='linear')

serie_full = serie_full.bfill()


serie_full = serie_full.ffill()



# %%
serie = pd.DataFrame(serie_full,columns=['EURUSD'])



# %%
serie.to_pickle(r"C:\Users\agarc\Projects\argentina_informal_fxmarket\data\EURUSD.pickle")