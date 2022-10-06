#!/bin/python3

import pandas as pd
from varmax import run_varmax_predictions
from trading import get_signals, load_etfs, compute_return, plot_returns, plot_predictions
import matplotlib.pyplot as plt


if __name__ == '__main__':
    
    # # Uncomment to run VARMAX    
    df_pred = run_varmax_predictions()
    
    # For testing purposes using results
    #   from a previously ran test
    #
    # df_pred = pd.read_csv(
    #     './predictions_test.csv',
    #     parse_dates=True,
    #     infer_datetime_format=True,
    #     index_col='Date'
    # )
    
    plot_predictions(df_pred)
    
    # Compute the signals based on predicted volatility
    df_signals = get_signals(df_pred)
    
    # Load ETF data
    df_cyc = load_etfs('./DATA/ETF_CYC')
    df_def = load_etfs('./DATA/ETF_DEF')
    
    # Compute the actual and strategy returns for each basket of ETFs
    df_cyc, final_cyc_returns = compute_return(df_cyc, df_signals)
    df_def, final_def_returns = compute_return(df_def, df_signals)
    
    # Show the plotted returns
    plot_returns(df_cyc, 'Cyclical ETFs Returns')
    plot_returns(df_def, 'Defensive ETFs Returns')
    
    # Print the return at the end of the testing period
    print('\n-----------------------')
    print('Cyclical ETFs Results:')
    print(final_cyc_returns)
    print('-----------------------')
    print('Defensive ETFs Results:')
    print(final_def_returns)
    
    plt.show()