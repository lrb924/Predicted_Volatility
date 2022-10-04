#!/bin/python3

from varmax import run_varmax_predictions
from trading import get_signals, load_etfs, compute_return


if __name__ == '__main__':

    df_pred = run_varmax_predictions()
    df_signals = get_signals(df_pred)
    
    df_cyc = load_etfs('./DATA/ETF_CYC')
    df_def = load_etfs('./DATA/ETF_DEF')
    
    df_cyc, final_cyc_returns = compute_return(df_cyc, df_signals)
    df_def, final_def_returns = compute_return(df_def, df_signals)
    
    print('Cyclical ETFs Results:')
    print(final_cyc_returns)
    print('-----------------------')
    print('Defensive ETFs Results:')
    print(final_def_returns)
    
    # defensive_plot = (1 + df_cyc[["cumulative_return", "cumulative_strategy_return"]]).plot.line()
    # (1 + df_def[["cumulative_return", "cumulative_strategy_return"]]).plot.line()