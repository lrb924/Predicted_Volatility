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


# %%
def compute_return(df, df_signals):
    
    final_returns = {}
    
    # df_etfs = df_etfs.copy().iloc[-len(df_signals):]
    
    df['return'] = df.sum(axis=1)
    df['strategy_return'] = df['return'] * df_signals['signal']
    df.dropna(inplace=True)
    
    df['cumulative_return'] = (1 + df["return"]).cumprod()
    df['cumulative_strategy_return'] = (1 + df["strategy_return"]).cumprod()
    
    final_returns['final_return'] = np.round(df['cumulative_return'].iloc[-1], 3)
    final_returns['final_strategy_return'] = np.round(df['cumulative_strategy_return'].iloc[-1], 3)
    
    return df, final_returns


# %%
# df_pred = pd.read_csv('./predictions_test.csv', parse_dates=True, infer_datetime_format=True, index_col='Date')
# df_signals = get_signals(df_pred)

# df_cyc = load_etfs('./DATA/ETF_CYC')
# df_def = load_etfs('./DATA/ETF_DEF')


