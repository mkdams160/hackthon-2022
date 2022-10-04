import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize as sco

def z_score(df,decimals):
    df.columns = [x + "_zscore" for x in df.columns.tolist()]
    df_z_score = ((df - df.mean())/df.std(ddof=0))
    df_z_score = round(df_z_score,decimals)
    return df_z_score


df = pd.read_csv('historical.csv')
df = df.assign(logret=np.log(df.close).groupby(df.symbol).diff())
df = df[~df.symbol.isin(['TUSD','PAX','USDC','USDT','TGBP','TAUD','TCAD','BUSD','DAI','GUSD','XSGD','WBTC'])]

result = df.groupby(['symbol'], as_index=False).agg({'logret':['mean','std'],'volume':'mean', 'marketCap':['mean','last']})

#### add zscore + filtering here #####
result.columns=['symbol','log_mean','log_std','vol_mean','market_cap_mean','market_cap_last']
#calculate information ratio here as one of filtering logic
#calculte mean(volume)/mean(market_cap) to gauge liquidity
result['vol_makcap'] = result['vol_mean']/result['market_cap_mean']
result['info_ratio'] = result['log_mean']/result['log_std']
result = result.drop(columns=['market_cap_mean'])
result = result.set_index('symbol')
z_score = z_score(result,2)

cols = ['market_cap_last_zscore','info_ratio_zscore','vol_makcap_zscore','log_std_zscore']
z_score = z_score.sort_values(cols, ascending=[False,False,False,True])

final_symbol = z_score.head(30)
final_symbol = final_symbol.reset_index()

# Optimization source : https://github.com/otosman/Python-for-Finance/blob/master/Portfolio%20Optimization%20401k.ipynb
#function for computing portfolio returns, std, and shapre
def port_stats(weights):
    w = np.array(weights)
    r = np.sum(w*mean_rets)
    std = np.sqrt(np.dot(w.T,np.dot(cov,w)))
    sharpe = r/std
    return np.array([r,std,sharpe])

rets = df[['timeClose','symbol','logret']]
rets = rets.reset_index().pivot_table(index='timeClose', columns='symbol', values='logret').dropna()
rets = rets.loc[:, rets.columns.isin(final_symbol.symbol.values.tolist())]
mean_rets = rets.mean() * 365
cov = rets.cov() * 365
syms = len(rets.columns)

#boundaries
bnds = tuple((.0,.15) for x in range(syms))
print(bnds)
bnds = list(bnds)
bnds[final_symbol[final_symbol['symbol'].str.contains('BTC')].index.values[0]] = (.1,.25)
bnds[final_symbol[final_symbol['symbol'].str.contains('ETH')].index.values[0]] = (.1,.25)
bnds[final_symbol[final_symbol['symbol'].str.contains('CRO')].index.values[0]] = (.1,.15)
bnds = tuple(bnds)
#constraints
cons = ({'type':'eq','fun':lambda x: np.sum(x)-1},
       {'type':'ineq','fun':lambda x: x})

#returns negative of sharpe b/c minimum of negative sharpe is the maximum of the sharpe ratio
def max_sharpe(weights):
    return -port_stats(weights)[2]

guess = syms*[1/syms]
sharpe_opt = sco.minimize(max_sharpe, guess, method = 'SLSQP',constraints=cons, bounds=bnds)

#now lets find the min variance portfolio

def min_var(weights):
    return port_stats(weights)[1]

min_var_opt = sco.minimize(min_var, guess, method = 'SLSQP',constraints=cons, bounds=bnds)
VaR_99 = norm.ppf(1-0.99, port_stats(min_var_opt.x)[0], port_stats(min_var_opt.x)[1])

##CSV OUTPUT
output = pd.DataFrame(columns=['symbol','Defensive','Balanced'])
output['symbol'] = final_symbol['symbol']
output['Defensive'] = min_var_opt.x
output['Balanced'] = sharpe_opt.x

print(port_stats(min_var_opt.x))
print(port_stats(sharpe_opt.x))
output.to_csv('portfolio_weight.csv')
