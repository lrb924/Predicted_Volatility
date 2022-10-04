# %%
# Importing Modules 
import os 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt



# import warnings
# warnings.filterwarnings('ignore')

# %%
def get_signals(df):
    df['signal'] = 0
    df['signal'] = np.where(df["prediction"]>df["observed"].shift().describe().loc["25%"], float(1),\
                    np.where(df["prediction"]<df["observed"].shift().describe().loc["75%"], float(-1),0))

    # print(df_trading['signal'].value_counts())
    
    return df
    
# get_signals(df_trading)

# %%
def load_etfs(directory):
    
    df = pd.DataFrame()
    
    for root,dirs,files in os.walk(directory):
        for filename in files: 
            etf_path  = (os.path.join(root, filename))
            
            df_etf = pd.read_csv(etf_path, parse_dates=True, infer_datetime_format=True, index_col='Date')
            df_etf.rename(columns={'Close': etf_path[-7:-4]}, inplace=True)
            
            df = pd.concat([df, df_etf.pct_change()], axis=1)
            df.dropna(inplace=True)
            
    return df

# df_cyc = load_etfs('./DATA/ETF_CYC')
# df_def = load_etfs('./DATA/ETF_DEF')

# display(df_cyc.head())
# display(df_def.head())

# %%
def compute_return(df_etfs, df_signals):
    
    final_returns = {}
    
    # df_etfs = df_etfs.copy().iloc[-len(df_signals):]
    
    df_etfs['return'] = df_etfs.sum(axis=1)
    df_etfs['strategy_return'] = df_etfs['return'] * df_signals['signal']
    df_etfs.dropna(inplace=True)
    
    df_etfs['cumulative_return'] = (1 + df_etfs["return"]).cumprod()
    df_etfs['cumulative_strategy_return'] = (1 + df_etfs["strategy_return"]).cumprod()
    
    final_returns['final_cumulative_return'] = np.round(df_etfs['cumulative_return'].iloc[-1], 3)
    final_returns['final_cumulative_strategy_return'] = np.round(df_etfs['cumulative_strategy_return'].iloc[-1], 3)
    
    return df_etfs, final_returns

# df_cyc, final_cyc_returns = compute_return(df_cyc, df_signals)
# df_def, final_def_returns = compute_return(df_def, df_signals)

# %%
# df_pred = pd.read_csv('./predictions_test.csv', parse_dates=True, infer_datetime_format=True, index_col='Date')
# df_signals = get_signals(df_pred)

# df_cyc = load_etfs('./DATA/ETF_CYC')
# df_def = load_etfs('./DATA/ETF_DEF')

# %%

    
    # print(final_cyc_returns)
    # print(final_def_returns)
    
    # defensive_plot = (1 + df_cyc[["cumulative_return", "cumulative_strategy_return"]]).plot.line()
    # (1 + df_def[["cumulative_return", "cumulative_strategy_return"]]).plot.line()

# %%
# defensive_plot = (1 + df_cyc[["return", "strategy_return"]]).cumprod().plot.line()
# cyclical_plot = (1 + df_def[["return", "strategy_return"]]).cumprod().plot.line()

# %%
# def_stratreturns = (1 + defensive_etfs[['return', 'strategy_return']]).cumprod().iloc[-1]
# cyc_stratreturns = (1 + cyclical_etfs[['return', 'strategy_return']]).cumprod().iloc[-1]
# # 
# display(def_stratreturns)
# display(cyc_stratreturns)


