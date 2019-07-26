
import sys
sys.path.append('C:/Dev/Python/pypbo/')

import pbo
import math

import pypbo as pbo
import pypbo.perf as perf


def metric(x):
    return np.sqrt(255) * perf.sharpe_iid(x)

S = 16

tsm = ts.rolling(5).sum()
tsm 

pbox = pbo.pbo(ts, S=S,
               metric_func=metric, threshold=1, n_jobs=4,
               plot=True, verbose=False, hist=False)


16*15*14*13*12*11*10*9

dd = perf.drawdown_from_rtns(ts, log=False)

dd.iloc[:, 0].plot()

np.log(pd.concat([1+dd.iloc[:, 0], tsc.iloc[:, 0]], axis=1)).plot()


dd2 = tsc.sub(tsc.cummax())
dd2.iloc[:, 0].plot()

pbox = []


