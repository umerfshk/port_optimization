#Modern portfolio theory : Portfolio Optimisation
#Author : Umer Shaikh 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
plt.style.use("seaborn")
pd.options.display.float_format = '{:.4f}'.format

#Importing Data from Yahoo finance
ticker = ["AMZN", "BA", "DIS", "IBM", "KO", "MSFT"]
stocks = yf.download(ticker, start = "2014-01-01", end = "2018-12-31")
stocks = stocks["Adj Close"].copy()
stocks.to_csv("port_stocks.csv")

stocks = pd.read_csv("port_stocks.csv", parse_dates= ["Date"], index_col= "Date")
ret = stocks.pct_change().dropna()

def ann_risk_return(returns_df):
    summary = returns_df.agg(["mean", "std"]).T 
    summary.columns = ["Return", "Risk"]
    summary.Return = summary.Return*260
    summary.Risk = summary.Risk * np.sqrt(260)
    return summary

#Generating random portfolios with random weightages to backtest optimal portfolio.  

summary = ann_risk_return(ret)
noa = len(stocks.columns)           #Number of Assets
nop = 15000                         #Number of Portfolios


matrix = np.random.random(noa * nop).reshape(nop, noa)
matrix
matrix.sum(axis = 1, keepdims= True)
weights = matrix / matrix.sum(axis = 1, keepdims= True)
weights.sum(axis = 1, keepdims= True)

port_ret = ret.dot(weights.T)


port_summary = ann_risk_return(port_ret)

#Calculating and using sharpe ratio in risk vs return Parameters
risk_free_return = 0.017
risk_free_risk = 0
rf = [risk_free_return, risk_free_risk]
rf

summary["Sharpe"] = (summary["Return"].sub(rf[0]))/summary["Risk"]
port_summary["Sharpe"] = (port_summary["Return"].sub(rf[0]))/port_summary["Risk"]



#Plotting the Optimal portfolios

plt.figure(figsize = (15, 8))
plt.scatter(port_summary.loc[:, "Risk"], port_summary.loc[:, "Return"], s= 20, 
            c = port_summary.loc[:, "Sharpe"], cmap = "flare", vmin = 0.76, vmax = 1.18, alpha = 0.8)
plt.colorbar()
plt.scatter(summary.loc[:, "Risk"], summary.loc[:, "Return"],s= 50, marker = "D", c = "black")
plt.xlabel("ann. Risk(std)", fontsize = 15)
plt.ylabel("ann. Return", fontsize = 15)
plt.title("Sharpe Ratio", fontsize = 20)
plt.show()


