Indices
* 4
  * Concat into one df

ETFs
* 5 Denfensive
  * Concat into one df
* 5 Cyclical
  * Concat into one df

GARCH
* Input: Log returns from Index df 
* Output: Volatility predictions
  * Four different measures of volatility (one for each index)
  * Get average between all volatilities for each day
  * Determine cutoff point, and assign label to each side of the cutoff point (1 or 0)
    * Use average standard deviation for each day
  
VAR
* Input:
* Output:

VARMAX
* Input:
* Output:

Algo Trading
* Two strategies for each ETF type
  * Buy or sell based on predicted signal

Predict volatility of indices based on indices themsevles
