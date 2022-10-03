# Predicted Volatility: Applying Predicted Volatility to Determine Profitability of Cyclical and Defensive ETFs

## **Project Overview**

### *Package Requirements and Versions*
`pip install x` ; where 'x' is the package listed below:
* `python == 3.7.13+`
* `arch == 5.3.1`
* `json == 2.0.9`
* `matplotlib == 3.5.1`
* `numpy == 1.21.5`
* `pandas == 1.3.5+`
* `scikit-learn == 1.1.2`
* `scipy == 1.9.1`
* `seaborn == 0.12`
* `statsmodels == 0.13.2`
* `tensorflow == 2.10.0`

### *Files Navigation*
* `Data`: Directory containing all original CSV files
* `Plots`: Directory containing all images of plots created in Jupyter Notebook
* `Notebook 1`
* `Notebook 2`

### *Purpose of Use*

Our team decided to investigate the effect of volatility of the S&P 500 Index on the profitability of cyclical and and defensive exchange-traded funds (ETFs). 

The business question we hope to answer is: *is trading defensive or cyclical ETFs based on future volatility profitable? If so, which set of ETFs is more profitable?*

Our motivation for taking on this challenge is to find out if, based on historical volatility data, we could predict future volatility. And using the predicted future volatility we then wanted to find out if cyclical or defensive ETFs were more profitable during the period of predicted volatility.

For reference, a **cyclical stock** is "a stock that's price is affected by macroeconomic or systematic changes in the overall economy. Cyclical stocks are known for following the cycles of an economy through expansion, peak, recession, and recovery. Most cyclical stocks involve companies that sell consumer discretionary items that consumers buy more during a booming economy but spend less on during a recession." [1]

A **defensive stock** is "a stock that provides consistent dividends and stable earnings regardless of the state of the overall stock market." [2]

And one important thing to note is that "[c]yclical stocks are generally the opposite of defensive stocks. Cyclical stocks include discretionary companies, such as Starbucks or Nike, while defensive stocks are staples, such as Campbell Soup." [1]

We hope to answer our business question by using historic closing data for the S&P 500, and the historic closing data for five cyclical ETFs, and five defensive ETFs. All of this data can be accessed through the Google Finance API.

In order to get the predicted volatility based on the S&P 500, we used both a GARCH model and a VARMAX model

The specific ETFs analyzed:
* Cyclical:
  * ITB: iShares U.S. Home Construction ETF (Cboe BZX Exchange)
  * IYC: iShares US Consumer Discretionary ETF (NYSE Arca Exchange)
  * PEJ: Invesco Dynamic Leisure & Entertainment ETF (NYSE Arca Exchange)
  * VCR: Vanguard Consumer Discretionary Index Fund ETF (NYSE Arca Exchange)
  * XLY: Consumer Discretionary Select Sector SPDR Fund (NYSE Arca Exchange)
* Defensive:
  * IYK: iShares US Consumer Staples ETF (NYSE Arca Exchange)
  * KXI: iShares Global Consumer Staples ETF (NYSE Arca Exchange)
  * PBJ: Invesco Dynamic Food & Beverage ETF (NYSE Arca Exchange)
  * VDC: Vanguard Consumer Staples Index Fund ETF (NYSE Arca Exchange)
  * XLP: Consumer Staples Select Sector SPDR Fund (NYSE Arca Exchange)

The time periods analyzed include:
* S&P 500: October 5, 2010 - September 30, 2022
* Cyclical ETFs: September 27/28, 2010 - September 23, 2022
* Defensive ETFs: September 27/28, 2010 - September 23, 2022

--------------

## Data Pre-Processing/Gathering Steps (Cleaning and Manipulation)



## Visuals and Explanations

## Additional Explanations and Major Findings

## Challenges, Limitations, and Future Developments

## Conclusion

## References

1. Cyclical Stocks Definition: https://www.investopedia.com/terms/c/cyclicalstock.asp
2. Defensive Stocks Definition: https://www.investopedia.com/terms/d/defensivestock.asp

Google Finance Data API

## Team Members
1. Lara Barger
2. Alec Gladkowski
3. Billel Loubari
4. Alejandro Palacios

