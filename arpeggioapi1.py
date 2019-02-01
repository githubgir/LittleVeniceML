import flask
from flask import request, jsonify
from flask_cors import CORS
#import json
import simplejson as json
import numpy as np
import dateutil
import datetime

import sys
import os
import psutil

import matplotlib
matplotlib.use('Agg')
import mpld3
import StrategyModule as sm
import StrategyUtils as su
import matplotlib.pyplot as plt

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      if obj.dtype == 'float64':
        obj = obj.round(6).tolist()
        return obj
#      elif obj.dtype == 'datetime':
#        d = [d_.toordinal() - 730122 + 36526 for d_ in d]
        return d
      elif obj.dtype == 'datetime64[ns]':
        d = obj.astype('datetime64[D]').astype(datetime.datetime)
        d = [d_.toordinal() - 730122 + 36526 for d_ in d]
        return d
      else:
        return obj.tolist()

    elif isinstance(obj, np.int64):
      return int(obj)

    return json.JSONEncoder.default(self, obj)

def getStrategy(DisplayName, request=None):
  StrategyNames = StrategyModuleSpecific.StrategyNames
  DisplayNames = StrategyModuleSpecific.DisplayNames

  StrategyName = [sn for dn, sn in zip(DisplayNames, StrategyNames) if dn == DisplayName ]
  s = sm.Strategy.from_pickle(StrategyName[0])

  #s.Date.values = su.datetime64_to_datetime(s.Date)
  
  if request is not None:
    if 'fromDate' in request.args:
      fromDate = dateutil.parser.parse(request.args["fromDate"])
      s.xr_ChangeDimension('Date', su.datetime64_to_datetime(s.Date) >= fromDate.date())
      print(s.Date.min())

    if 'toDate' in request.args:
      toDate = dateutil.parser.parse(request.args["toDate"])
      # truncate dates
      s.xr_ChangeDimension('Date', su.datetime64_to_datetime(s.Date) <= toDate.date())
      print(s.Date.max())

  return s

def getFundTimeSeries(request):
  Fund = request.args['fund']
  Item = request.args['item']
  
  p = sm.Strategy.from_pickle(Fund)
  

  if Item in ['ReturnContribution', 'ReturnContributionCum', 'VolContribution', 'Weight']:
    res = p.GetTimeSeries(Item)
    res = {
      'Format': 'List',
      'Fund': p.StrategyName,
      'Strategy': p.ID.values, 
      'Date': res[0].Date.values,
      'Item': request.args['item'],
      'Values': [{
        'ID': s.ID.values, 
        'Values': s.values.transpose([1, 0])}
        for s in res
      ]
    }
  else:
    res = p.GetTimeSeries(Item)
    res = {
      'Format': 'Single',
      'Fund': p.StrategyName,
      'Date': res.Date.values,
      'ID': res.ID.values,
      'Values': res.values.transpose([1, 0])
    }

  return res


def getTimeSeries(request):
  strategyName = request.args['strategyName']

  s = getStrategy(strategyName, request)

  res = s[request.args['item']]

  #res = res[~res.isnull()]

  res = {'Value': res.values,
    'Date': res.Date.values,
    'Item': request.args['item'],
    'StrategyName': request.args['strategyName'],
    'TSType': request.args['plotType']
  }

  return res

def getTimeSeriesByID(request):
  strategyName = request.args['strategyName']

  s = getStrategy(strategyName)

  res = s[request.args['item']]

  res = {'Values': res.values.transpose([1, 0]),
    'ID': res.ID.values,
    'Date': res.Date.values,
    'Item': request.args['item'],
    'StrategyName': request.args['strategyName'],
    'TSType': request.args['plotType']
  }    

  return res

def getPayoffProfile(request):
  strategyName = request.args['strategyName']

  s = getStrategy(strategyName, request)

  fig, ax = plt.subplots()
  s.CalcPayoffProfile()
  return mpld3.fig_to_dict(fig)



def getScatterPlotGrid(request):
    strategyName = request.args['strategyName']
    items = request.args["items"].split(",")
    type = request.args["type"]
    
    s = getStrategy(strategyName, request)

    df, r = su.ScatterPlotGrid.Prepare(
        s,
        Freq=int(request.args["freq"]),
        Items=items,
        Regime=request.args["regime"],
        QuantileBuckets=eval(request.args["quantileBuckets"]))
    
    if type == 'LinkedBrush':
      su.ScatterPlotGrid.LinkedBrush(df, r)
    else:
      su.ScatterPlotGrid.Seaborn(df)

    a = mpld3.fig_to_dict(plt.gcf())

    return a

def getEventPathsFig(request, strategyOrPortfolio=True):
    if 'bands' in request.args:
      bands = int(request.args["bands"])!=0
    else:
      bands = False
      
    mPath, sPath = getEventPathsAll(request, strategyOrPortfolio)

    if bands:
      res, _ = su.EventPath.Plotd3(mPath, sPath)
    else:
      res, _ = su.EventPath.Plotd3(mPath, sPath*0)
    
    plt.close('all')
    return res

def getEventPaths(request, strategyOrPortfolio=True):
    print(getEventPaths)
    mPath, sPath = getEventPathsAll(request, strategyOrPortfolio)

    res = {'Offset': mPath.index.values,
           'Mean': mPath.values,
           'Std': sPath.values,
           'Regime': mPath.columns.values}
    return res

def getEventPathsAll(request, strategyOrPortfolio=True):
    # parse request.args
    plotType = request.args['plotType']
    strategyName = request.args['strategyName']
    regime = request.args["regime"]

    if 'valueBuckets' in request.args:
      valueBuckets = eval(request.args["valueBuckets"])
    else:
      valueBuckets = None
    
    if 'quantileBuckets' in request.args:
      quantileBuckets = eval(request.args["quantileBuckets"])
    else:
      quantileBuckets = None

    if 'off' in request.args:
      off = eval(request.args["off"])
    else:
      off = [0, 260]

    if 'strategyOrPortfolio' in request.args:
      strategyOrPortfolio = int(request.args['strategyOrPortfolio'])!=0
    else:
      strategyOrPortfolio = True

    # get strategy file
    s = getStrategy(strategyName, request)

    # get regime
    Regime = su.EventPath.PrepRegime(s[regime], ValueBuckets=valueBuckets, QuantileBuckets=quantileBuckets)

    if strategyOrPortfolio:
      Return = s.StrategyReturn.to_pandas()
      mPath, sPath, _ = su.EventPath.EventPaths(Return, Regime, off=off, plot=False)
    else:
      Weight = s.RawWeight.to_pandas()
      Return = s.Return.to_pandas()
      mPath, sPath, _ = su.EventPath.EventPortfolioPaths(Return, Weight, Regime, off=off, plot=False)

    return mPath, sPath

def getEventPortfolioPaths(request):
    return getEventPaths(request, False)





def getBacktest2Range(request):
  strategyName = request.args['strategyName']

  # get strategy file
  s = getStrategy(strategyName)

  grp1, prm1, val1 = request.args["prm1"].split('.')
  if prm1 == 'IncludeOverride':
    val1, lab1 = s.BacktestParam_Includes(val1)
    lab1 = list(range(0, len(lab1)))
  else:
    val1 = eval(val1)
    lab1 = val1

  grp2, prm2, val2 = request.args["prm2"].split('.')
  if prm2 == 'IncludeOverride':
    val2, lab2 = s.BacktestParam_Includes(val2)
    lab2 = list(range(0, len(lab2)))
  else:
    val2 = eval(val2)
    lab2 = val2

  prms = [
    {'Group': grp1, 
    'Param': prm1, 
    'Values': val1},

    {'Group': grp2, 
    'Param': prm2, 
    'Values': val2}
  ]

  item = request.args['item']
  items = [item, "StrategyReturn", "StrategyReturnCum", "StrategyReturnNet", "StrategyReturnNetCum"]

  res = s.BacktestParamsX(prms, items, plot=False)
  for iI, i in enumerate(items):
    res_ = {'Date': s.Date.values,
          'Values': res[:, iI, :, :].transpose([1, 2, 0]),
          'X1': lab1,
          'X2': lab2,
          'Item': request.args['item'],
          'StrategyName': request.args['strategyName'],
          'Param1': grp1+'.'+prm1,
          'Param2': grp2+'.'+prm2,
          'TSType': request.args['plotType']
    }          
    url_ = request.url.replace(items[0], i)
    CacheBacktest(url_, res_)

  res_ = {'Date': s.Date.values,
        'Values': res[:, 0, :, :].transpose([1, 2, 0]),
        'X1': lab1,
        'X2': lab2,
        'Item': request.args['item'],
        'StrategyName': request.args['strategyName'],
        'Param1': grp1+'.'+prm1,
        'Param2': grp2+'.'+prm2,
        'TSType': request.args['plotType']
  }
  return res_

def CacheBacktest(url, res):
    res = json.dumps(res, ignore_nan=True, cls=NumpyEncoder)

    if url not in cache:
      print('writing cache')
      cache[url] = res

def getBacktest2Val(request):
  import warnings
    
  strategyName = request.args['strategyName']

  # get strategy file
  s = getStrategy(strategyName)
  item = request.args['item']

  grp1, prm1, val1 = request.args["prm1"].split('.')
  grp2, prm2, val2 = request.args["prm2"].split('.')

  prms = [
    {'Group': grp1, 
    'Param': prm1, 
    'Value': int(val1)},

    {'Group': grp2, 
    'Param': prm2, 
    'Value': int(val2)}
  ]

  with warnings.catch_warnings(record=True):
    warnings.simplefilter("once")
    res = s.BacktestParamsSingle(prms)

  
  res = {'Value': res[item].values,
    'Date': res.Date.values}

  return res

def getBacktestFig(request):
    strategyName = request.args['strategyName']

    # get strategy file
    s = getStrategy(strategyName, request)
    prms = [
        {'Group': "GenerateSignals", 
        'Param': "Lookback", 
        'Values': list(range(50, 500, 50))},

        {'Group': "GenerateSignals", 
        'Param': "LookbackExclude", 
        'Values': list(range(0, 40, 5))}
    ]

    fig, ax = plt.subplots()
    res = s.BacktestParamsX(prms, plot=True)
    return mpld3.fig_to_dict(fig)


############### WEB API ##################

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})

app.config["DEBUG"] = True

global cache
cache = dict()

@app.route('/', methods=['GET', 'POST'])
def home():
    return '''<h1>arpeggio ONE</h1><p>This is the main entry point of arpeggio ONE web api.</p>'''


@app.route('/api/any', methods=['GET', 'POST'])
def api_any():
    
    plotType = request.args['plotType']
    print(plotType)

    global cache    
    if plotType=='clearCache':
      cache = dict()
      fig = {"Cache": "Cleared"}
      return json.dumps(fig, ignore_nan=True, cls=NumpyEncoder)

    print(request.args)

    # check if in cache first
    if request.url in cache:
      print('cache hit')
      return cache[request.url]

    if plotType=='eventPaths':
      fig = getEventPaths(request)
    elif plotType=='eventPathsFig':
      fig = getEventPathsFig(request)
    elif plotType=='eventPortfolioPathsFig':
      fig = getEventPathsFig(request, False)
    
    elif plotType=='payoffProfile':
      fig = getPayoffProfile(request)

    elif plotType=='backtestFig':
      fig = getBacktestFig(request)
    elif plotType=='backtest2Range':
      fig = getBacktest2Range(request)
    elif plotType=='backtest2Val':
      fig = getBacktest2Val(request)
    elif plotType=='backtest3':
      fig = getBacktest3(request)

    elif plotType=='timeSeries':
      fig = getTimeSeries(request)
    elif plotType=='timeSeriesByID':
      fig = getTimeSeriesByID(request)
    elif plotType=='scatterPlotGrid':
      fig = getScatterPlotGrid(request)
    elif plotType=='FundTimeSeries':
      fig = getFundTimeSeries(request)
    else:
      pass


    res = json.dumps(fig, ignore_nan=True, cls=NumpyEncoder)

    if request.url not in cache:
      print('writing cache')
      cache[request.url] = res

    process = psutil.Process(os.getpid())
    print(process.memory_info().rss/1e6)

    return res

app.run(host='0.0.0.0' , port=5000)#, ssl_context='adhoc')

