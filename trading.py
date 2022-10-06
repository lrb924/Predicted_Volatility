# Importing Modules 
import os 

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings('ignore')

def get_signals(df):
    df['signal'] = 0
    df['signal'] = np.where(df["prediction"]>df["observed"].shift().describe().loc["25%"], float(1),\
                    np.where(df["prediction"]<df["observed"].shift().describe().loc["75%"], float(-1),0))
    
    return df

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

def compute_return(df, df_signal):
    
    final_returns = {}
    
    df['return'] = df.sum(axis=1)
    df['return'].iloc[0] = 1
    df['strategy_return'] = df['return'] * df_signal['signal']
    df.dropna(inplace=True)
    
    df['cumulative_return'] = (1 + df["return"]).cumprod()
    df['cumulative_strategy_return'] = (1 + df["strategy_return"]).cumprod()
    
    final_returns['final_return'] = np.round(df['cumulative_return'].iloc[-1], 3)
    final_returns['final_strategy_return'] = np.round(df['cumulative_strategy_return'].iloc[-1], 3)
    
    return df, final_returns

def plot_predictions(df):
    fig = plt.figure()
    df.plot(figsize=(20,5),title='volatility forecast error',color=['blue','purple','green'],style=['-','-',':'])
    plt.legend(loc=('upper left'));
    return fig

def plot_returns(df, title):
    fig = plt.figure()
    df[["cumulative_return", "cumulative_strategy_return"]].plot()
    plt.title(title)
    plt.legend(loc=('upper left'));
    return fig