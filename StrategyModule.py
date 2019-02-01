# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 10:47:43 2018

@author: Andreas
"""

import pandas as pd
import dateutil
import datetime
import numpy as np
import xarray as xr
import re
import pickle
import seaborn as sns

from StrategyUtils import *
import matplotlib.pyplot as plt
from IPython.core.debugger import set_trace
import math
from xarray.core.pycompat import (OrderedDict, iteritems)

#region Helpers

class StrategyParams(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, d = None):
      if d is not None:
        for k, v in d.items():
          if isinstance(v, dict):
            self[k] = StrategyParams(v)
          else:
            self[k] = v

    def merge(self, other, overrideOther=True):
      '''
      merge parameters
      if overrideOther then self overrides other
      otherwise other overrides self
      '''
      if overrideOther:
        old, new = other, self
      else:
        old, new = self, other

      for k, v in zip(old.keys(), old.values()):
        if k in new.keys():
          v.update(new[k])
        
        new[k] = v

      return new

    def to_dict(self):
      d = dict()
      for k, v in self.items():
        if isinstance(v, StrategyParams):
          d[k] = v.to_dict()
        else:
          d[k] = v
      return d

    def copy(self):
      return StrategyParams(self.to_dict())


class TSDB():
  DataDir = 'C:/Dev/BBData/'
  def __init__(self):
    # load time series
    TS1 = pd.read_pickle(TSDB.DataDir + 'RP.pkl')
    TS2 = pd.read_pickle(TSDB.DataDir + 'AssetTS.pkl')

    # combine no duplicates
    iExclude = [np.any([c == d for d in TS2.columns]) for c in TS1.columns]
    iExclude = np.array(iExclude)
    TS1 = TS1.iloc[:, ~iExclude]
    TS = pd.concat([TS1, TS2], axis=1)

    # remove PX_LAST
    TS.columns = TS.columns.droplevel(1)
    
    # add dummies so to ensure all references are in the df
    dummy = pd.DataFrame(columns=['none', 'dummy'], index=TS.index)
    TS = pd.concat([TS, dummy], axis=1)

    # store
    self.TS = TS

  def IsAvailable(self, RequestedTSID):
    RequestedTSID = np.array(RequestedTSID)
    bAvail = [t in self.TS.columns for t in RequestedTSID.flatten()]
    bAvail = np.array(bAvail).reshape(RequestedTSID.shape)

    MissingTSID = np.unique(RequestedTSID[~bAvail])
    #AvailTSID = np.unique(RequestedTSID[bAvail])

    return bAvail, MissingTSID

  def GetTS(self, RequestedTSID, ErrorOrNanIfMissing=False):
    # check if any missing
    _, MissingTSID = self.IsAvailable(RequestedTSID)

    TS = self.TS
    # and either error or add nans
    if MissingTSID.size > 0 and ErrorOrNanIfMissing:
      print(MissingTSID)
      raise ValueError("Requested TSID is missing in the database.")
    elif MissingTSID.size > 0 and ~ErrorOrNanIfMissing:
      import warnings
      warnings.warn("Requested TSID is missing in the database, filling with nans")

      # add missing TS
      for iT in set(MissingTSID):
        TS[iT] = np.nan
      
    RequestedTSID = pd.DataFrame(RequestedTSID)

    # now can get all ts
    RequestedTSID = pd.DataFrame(RequestedTSID).unstack()
    TS = TS[RequestedTSID.values]
    TS.columns = RequestedTSID.index

    return TS

class StrategyDefinition:
  DataDir = 'C:/Dev/Data/'
  def __init__(self, StrategyName):
    self.StrategyName = StrategyName
    self.Load()    

  def Load(self):
    self.Params = StrategyDefinition.LoadParams(self.StrategyName)
    self.LoadTSxID()
    self.LoadParamsxID()
    
  @staticmethod
  def LoadDefFile(FileName, StrategyName):
    # load all, then filter
    cont = pd.read_excel(FileName)
    cont = cont[cont.StrategyName == StrategyName]

    #remove empty columns and StrategyName column
    cont = cont.loc[:, ~cont.isnull().all()]
    cont = cont.drop(['StrategyName'], axis=1)

    return cont

  def LoadParamsxID(self):
    ParamsxID = StrategyDefinition.LoadDefFile(
      StrategyDefinition.DataDir + 'ParamsxID.xlsx', 
      self.StrategyName)

    ParamsxID = ParamsxID.sort_index(axis=0, ascending=True)
    ParamsxID['SortOrder'] = range(0, len(ParamsxID.index))
    ParamsxID = ParamsxID.set_index('ID')
    self.ParamsxID = ParamsxID
    return ParamsxID

  def LoadTSxID(self):
    TSxID = StrategyDefinition.LoadDefFile(
        StrategyDefinition.DataDir + 'TSxID.xlsx', 
        self.StrategyName)
    
    TSxID = TSxID.set_index('ID')
    self.TSxID = TSxID
    return TSxID

  @staticmethod
  def SpecialParam(p, v):
    # treat special cases of params i.e.  _items, _eval
    if p.endswith(r'_items'):
        p = re.sub(r'_items$', '', p)
        if isinstance(v, str):
          v = re.split(r'\W+', v)
        else:
          v = []

    elif p.endswith(r'_items_int'):
        p = re.sub(r'_items_int$', '', p)
        if isinstance(v, str):
          v = re.split(r'\W+', v)
          v = [int(i) for i in v]
        else:
          v = []
            
    elif p.endswith(r'_eval'):
        p = re.sub(r'_eval$', '', p)
        if isinstance(v, str):
          v = eval(v)
        else:
          v = None

    return p, v

  @staticmethod
  def List2Params(p):
    # convert
    iMulti = p[['Group', 'Param']].values # lists
    iMulti = [tuple(iI) for iI in iMulti] # tuples
    iMulti = pd.MultiIndex.from_tuples(iMulti) # multi index
    
    p = p.set_index(iMulti) # apply as multi index

    sp = StrategyParams()
    
    # 1st loop over groups
    # 2nd loop over Params in each group
    for Group, dfGroup in p.groupby(level=0):
      
      g = StrategyParams()
      for _, row in dfGroup.iterrows():
        if isinstance(row.Param, str) and len(row.Param) > 0:
          p, v = StrategyDefinition.SpecialParam(row.Param, row.Value)
          g[p] = v
        
      sp[Group] = g

    return sp

  @staticmethod
  def LoadParams(StrategyName):
    
    p = StrategyDefinition.LoadDefFile(
          StrategyDefinition.DataDir + 'Params.xlsx',
          StrategyName)

    p = StrategyDefinition.List2Params(p)

    dpname = p.Global.Default
    if isinstance(dpname, str) and len(dpname)>0:
      # if specified load default Params
      sp = StrategyDefinition.LoadParams(dpname)
      
      #merge
      p.merge(sp)

    return p

  def to_xarray(self):
    f = self.ParamsxID.to_xarray()
    p = self.Params

    ts = self.TSxID.unstack()
    ts.index.levels[0].name = 'TSItem'
    ts = ts.to_xarray()
    ts.name = 'TSID'

    x = xr.merge([f, ts])
    x.attrs['Params'] = p

    self.x = x
    return x



#endregion

#region Strategy
class Strategy(xr.Dataset):
  #region xr
  def xr_ChangeDimension(self, dim, coord):
    self.isel_inplace({dim: coord})

  def isel_inplace(self, indexers=None, drop=False, **indexers_kwargs):
    # copied from Dataset.isel to allow for inplace argument
    #indexers = either_dict_or_kwargs(indexers, indexers_kwargs, 'isel')

    indexers_list = self._validate_indexers(indexers)

    variables = OrderedDict()
    for name, var in iteritems(self._variables):
        var_indexers = {k: v for k, v in indexers_list if k in var.dims}
        new_var = var.isel(indexers=var_indexers)
        if not (drop and name in var_indexers):
            variables[name] = new_var

    coord_names = set(variables).intersection(self._coord_names)
    selected = self._replace_vars_and_dims(variables,
                                            coord_names=coord_names)

    # Extract coordinates from indexers
    coord_vars = selected._get_indexers_coordinates(indexers)
    variables.update(coord_vars)
    coord_names = (set(variables)
                    .intersection(self._coord_names)
                    .union(coord_vars))
    self._replace_vars_and_dims(variables, coord_names=coord_names, inplace=True)


  def xr_get_vars_with_dims(s, dims):
    return [i[0] for i in s.variables.items() if s[i[0]].dims == dims]

    for i in s.variables.items():
        m = i[0]
        if s[m].dims == dims:
            print(m)

  @staticmethod
  def xr_NewDataArray(ds, dims, data=None, fill_value=np.nan):

      coords = [ds[d] for d in dims]
      shape = [len(ds[d]) for d in dims]
      if data is None:
          data = np.full(shape, fill_value=fill_value)

      return xr.DataArray(data, coords=coords, dims=dims)

  def xr_AddDataArray(self, name, dims, data=None, fill_value=np.nan):

      coords = [self[d] for d in dims]
      shape = [len(self[d]) for d in dims]
      if data is None:
          data = np.full(shape, fill_value=fill_value)    
    
      if isinstance(name, list):
        for n in name:
          self[n] = xr.DataArray(data, coords=coords, dims=dims)
      elif isinstance(name, str):
          self[name] = xr.DataArray(data, coords=coords, dims=dims)
      else:
          raise Exception('name should be a string or a list of strings')

  @staticmethod
  def xr_NewDim(ds, dim, coord):
    xr.DataArray(
      np.full(len(coord)),
      coords = coord,
      dims = dim)

  def copy(self, deep=False):
    cp = super(Strategy, self).copy(deep=deep)
    
    if deep:
      # special treatment for some attributes

      if "Params" in self.attrs:
        cp.attrs["Params"] = self.attrs["Params"].copy()

    return cp

  #endregion

  #region constructors
  def __init__(self, StrategyName, DisplayName=None):
    super().__init__()
    if DisplayName is None:
      DisplayName = StrategyName

    self.attrs['StrategyName'] = StrategyName
    self.attrs['DisplayName'] = DisplayName
    self.attrs['DataCleaningDone'] = False
    self.attrs['Def'] = None
    self.attrs['Params'] = None
    pass
  
  @staticmethod
  def StrategyFactory(StrategyName, DisplayName=None):
    cls = globals()['Strategy_' + StrategyName]

    if DisplayName is None:
      DisplayName = StrategyName

    return cls(StrategyName, DisplayName)

  AnnualisationFactor = 250
  Implemented_StrategyNames = StrategyModuleSpecific.Implemented_StrategyNames
  Implemented_DisplayNames = StrategyModuleSpecific.Implemented_DisplayNames
  DisplayNames2StrategyNames = StrategyModuleSpecific.DisplayNames2StrategyNames

  @staticmethod
  def LoadAllImplemented():
    Ss = []
    for s, d in zip(Strategy.Implemented_StrategyNames, Strategy.Implemented_DisplayNames):
      Ss.append(Strategy.from_pickle(s))

    return Ss

  @staticmethod
  def CalcAllImplemented():
    Ss = []

    for s, d in zip(Strategy.Implemented_StrategyNames, Strategy.Implemented_DisplayNames):
      S = Strategy.StrategyFactory(s, d)
      S.Load()
      S.CalcAll()
      S.to_pickle()
      Ss.append(S)
    
    return Ss

  @staticmethod
  def from_pickle(StrategyName = None, FileName = None):
    if FileName is None:
      FileName = 'Strategy_' + StrategyName

    FileHandler = open(FileName, 'rb')
    self = pickle.load(FileHandler)
    self.attrs["Params"] = StrategyParams(self.attrs["Params"])
    FileHandler.close()

    return self

  def to_pickle(self, FileName = None):
    print("TODO: use parent implementation instead")
    if FileName is None:
      FileName = 'Strategy_' + self.StrategyName
    
    s = self.copy(deep=True)
    s.attrs["Params"] = s.attrs["Params"].to_dict()
    s.attrs["Def"] = []
    FileHandler = open(FileName, 'wb')
    pickle.dump(s, FileHandler)
    FileHandler.close()

  def prep_powerbi(self, dim):
    # to update the lists below run with ('Date', ) and ('Date', 'ID')
    # s.xr_get_vars_with_dims(('Date',))
    #
    ns1 = ['CorrAverage',
     'CorrCrowding',
     'CorrDispersion',
     'CorrWeighted',
     'Crowding',
     #'Date',
     'DiversificationRatio',
     'Herfindal',
     'IC',
     'ICAnn',
     'Intensity',
     'IsRebalanceDay',
     'Leverage',
     'NetExposure',
     'SignalAutoCorr',
     'SignalAutoCorr1m',
     'SignalAverage',
     'SignalDispersion',
     'SignalDuration',
     'SignalWeightCorr',
     'SignalWeighted',
     'StrategyReturn',
     'StrategyReturnCum',
     'StrategyReturnNet',
     'StrategyReturnNetCum',
     'StrategyVol',
     'TC',
     'Turnover',
     'TurnoverAnn',
     'VolAverage',
     'VolCrowding',
     'VolDispersion',
     'VolExante',
     'VolWeighted']

    ns2 = [
     'Return',
     'ReturnCum',
     'Signal',
     'RawSignal',
     'Weight',
     'RawWeight',
     'ReturnContribution',
     'ReturnContributionCum',
     'VolContribution',
     'Vol',
     'CorrToStrategy']

    
    if dim == 1:
      ns = ns1
      ps = [self[m].to_pandas() for m in ns]
    elif dim == 2:
      ns = ns2
      ps = [self[m].stack(z=('Date', 'ID')).to_pandas() for m in ns]
    else:
      raise Exception('not implemented')

    pp = pd.concat(ps, axis=1)
    pp.columns = ns
    
    pp['StrategyName'] = self.DisplayName
    pp.set_index('StrategyName', append=True, inplace=True)
    
    return pp
  
  def PrintParams():
    for g, c, in s.Params.items():
        for p, v in c.items():
            print(g, '\t', p, '\t', v)    

  @staticmethod
  def to_powerbi(S):
    X1 = pd.concat([s.prep_powerbi(1) for s in S], axis=0)
    X1.to_csv('X1.csv')

    X2 = pd.concat([s.prep_powerbi(2) for s in S], axis=0)
    X2.to_csv('X2.csv')


  #endregion

  #region load
  def LoadDefinition(self):
    self.attrs['Def'] = StrategyDefinition(self.StrategyName)
    self.attrs['Params'] = self.Def.Params
    pass

  def LoadAdditionalData(self):
    pass

  def LoadTS(self):
    #load TSxID
    tsdb = TSDB()
    tsxid = tsdb.GetTS(self.Def.TSxID)
    tsxid.columns.levels[0].name = "TSItem"
    tsxid.index.name = "Date"

    #get all additional TS 
    p = self.Params.AddTS
    # add strategy index
    if ~isinstance(p.StrategyReturnCompare, str) or len(p.StrategyReturnCompare)==0:
      p.StrategyReturnCompare = self.StrategyName + ' Index'
    
    self.Params.AddTS = p

    req = [v for k, v in p.items()]
    key = [k for k, v in p.items()]

    # load TS from p.AddTS
    tsdb = TSDB()
    ts = tsdb.GetTS(req)

    ts.index.name = "Date"
    ts.columns = pd.Index(key, name='TSItem')

    # add
    tsxid = tsxid.stack().to_xarray()
    ts = ts.to_xarray()
    pxid = self.Def.ParamsxID.to_xarray()

    xadd = xr.merge([tsxid, ts, pxid])
    self.merge(xadd, inplace=True)

  def Load(self):
    self.LoadDefinition()
    self.LoadTS()
    self.LoadAdditionalData()

  #endregion

  #region Calc Performance
  def Prepare(self):
    self.attrs['Variables_DatexID'] = \
          ['Signal', 'RawSignal', 'Weight', 'RawWeight', 
           'ReturnContribution', 'VolContribution', 'Vol', 'CorrToStrategy',
           'ReturnContributionCum', 'ReturnCum']

    self.attrs['Variables_Date'] = \
          ['TC', 'Turnover', 'TurnoverAnn',
           'VolCrowding', 'CorrCrowding', 'Crowding', 'DiversificationRatio', 'Herfindal',
           'Leverage', 'NetExposure',
           'SignalWeighted', 'Intensity', 'SignalAverage', 'SignalDispersion', 'SignalDuration', 
           
           'SignalWeightCorr', 'SignalAutoCorr', 'SignalAutoCorr1m',
           
           'StrategyVol', 'VolExante', 'VolWeighted', 'VolAverage', 'VolDispersion',
           
           'CorrAverage', 'CorrWeighted', 'CorrDispersion',
           'IC', 'ICAnn',

           'StrategyReturn', 'StrategyReturnNet', 
           'StrategyReturnCum', 'StrategyReturnNetCum',
           'StrategyReturn1m', 'StrategyReturn3m', 'StrategyReturn6m', 'StrategyReturn12m'
           ]

    self.xr_AddDataArray(self.attrs['Variables_DatexID'], dims=['Date', 'ID'])
    self.xr_AddDataArray(self.attrs['Variables_Date'], dims=['Date'])

    self.xr_AddDataArray(
          ['IsRebalanceDay'], 
          dims=['Date'],
          fill_value = False)

  def ForceCalcOnAllWeekdays(self):
    p = self.Params.DataCleaning

    p.FillForward = p.FillForward + p.RemoveDateIfNanAny
    p.RemoveDateIfNanAny = []

    valid = [0<=datetime.datetime.weekday(d)<=4 for d in datetime64_to_datetime(self.Date)]
    self.xr_ChangeDimension('Date', valid)

    pass

  def DataCleaning(self):
    
    if self.attrs['DataCleaningDone']:
      return

    p = self.Params.DataCleaning
    
    # for syncronous trading across strategies we may need to force calculation on all weekdays
    if p.ForceCalcOnAllWeekdays:
      self.ForceCalcOnAllWeekdays()

    # Remove Nans
    for m in p.RemoveDateIfNanAny:
      if self[m].dims == ('Date', 'ID'):
        valid = ~self[m].isnull().any(dim='ID')
      elif self[m].dims == ('Date',):
        valid = ~self[m].isnull()
      else:
        # should never happen
        raise Exception('{} has invalid dimensions {}'.format(m, self[m].dims))

      # remove invalid dates on every iteration
      self.xr_ChangeDimension('Date', valid)

    # FillForward
    for m in p.FillForward:
      self[m] = self[m].ffill(dim='Date')

    # Price2Return
    for m in p.Price2Return:
      # need to go via pandas
      self[m] = self[m].to_pandas().pct_change(axis=0)
      self

    self.attrs['DataCleaningDone'] = True

  def SetInclude(self):
    p = self.Params.DataCleaning

    if isinstance(p.IncludeOverride, np.ndarray):
      self.Include.values = p.IncludeOverride

    self.xr_ChangeDimension('ID', self.Include.values!=0)

  def SetRebalanceDays(x):
    #
    # https://dateutil.readthedocs.io/en/stable/rrule.html
    # from dateutil.rrule import *
    # r = rrule(MONTHLY, count=5, byweekday=(MO,TU,WE,TH,FR), bysetpos=(-2,15))
    # DTSTART:20180922T025156
    # FREQ=MONTHLY;COUNT=5;BYSETPOS=-2,15;BYDAY=MO,TU,WE,TH,FR
    #
    # 
    from dateutil.rrule import rrule, rrulestr
    from datetime import datetime, date
    
    p = x.Params.SetRebalanceDays
    
    # get list of rebalance dates as defined by the rule
    rstr = p.Rule + ';COUNT=1'
    r = rrulestr(rstr)

    datefrom = datetime64_to_datetime(x.Date[0])
    dateto = datetime64_to_datetime(x.Date[-1])

    r = r.replace(dtstart=datefrom, until=dateto, count=None)

    RebalanceDates = list(r)

    # make sure rebalance dates are in the Date dim
    RebalanceDates = x.Date.sel({'Date': RebalanceDates}, method='bfill').values
    # mark in x
    x.IsRebalanceDay.loc[RebalanceDates] = True

    return RebalanceDates
  
  def GenerateSignals(x):
    p = x.Params.GenerateSignals
    x.RawSignal.values = x[p.Signal];
    
  def LagWeight(self, wgt):
    p = self.Params.CalcPerformance
    return wgt.shift(Date=p.ImplementationLag)

  def ConstructWeights(x):
    p = x.Params.ConstructWeights

    if 'CSorTS' not in p:
      p["CSorTS"] = 1
    if 'VolTargetAssetsLookback' not in p:
      p["VolTargetAssetsLookback"] = np.nan

    raw = x.RawSignal.values
    
    if p.CSorTS==1:
      sig = TransformSignal(raw, p.Transform)
      wgt = Signal2Weight(sig, p.Method, p.MethodParam)
      wgt = NormaliseWeight(wgt, p.NormaliseMethod)
      wgt = wgt * p.Multiplier
    else:  
      sig = raw
      wgt = np.sign(sig)/len(x.ID)

    wgt[np.isnan(wgt)] = 0

    # rescale weights so each individual asset is of the same vol
    if not p.VolTargetAssetsLookback is np.nan:
      Vol = (x.Return
        .rolling(Date=p.VolTargetAssetsLookback)
        .std()
        *x.AnnualisationFactor**0.5)
      wgt = wgt * 0.1 / Vol

    x.Signal.values = sig
    x.RawWeight.values = wgt

    # smooth rebalance over view days
    x.Weight.values = x.RawWeight.rolling(Date=p.SmoothRebalance).mean()

    # and now we shift
    x.Weight.values = x.LagWeight(x.Weight)

  def VolTargeting(s):
    if 'VolTargeting' not in s.Params:
      p = StrategyParams()
      p.VolTarget = 0.1
      p.ImplementationLag = 2
      p.LeverageLimit = 2
      p.Enabled = True
      #print('No VolTargeting parameters found, add these as example to proceed.')
      #print(p)
      return p

    p = s.Params.VolTargeting

    if not p.Enabled:
      # not enabled so nothing to do
      return;

    # need StrategyReturn and VolExante for the calculations below
    s.CalcPerformance()
    s.CalcRisk()

    lev = p.VolTarget / s.VolExante.shift(Date=p.ImplementationLag)
    lev[lev>p.LeverageLimit] == p.LeverageLimit
    lev[np.isinf(lev.values)] = np.nan

    # adjust Weights
    s.Weight.values = s.Weight * lev.values[:, None]
    pass

  def StopLoss(s):
    from scipy.signal import convolve

    if 'StopLoss' not in s.Params:
      p = StrategyParams()
      p.VolThreshold = 2
      p.ExposureReduction = 0.1
      p.ImplementationLag = 2
      p.ExposureRecovery = 10
      p.Enabled = True
      p.AggPeriods = 3 # check cum 3 days return vs. 3 day stdev
      #print('No StopLoss parameters found, add these as example to proceed.')
      #print(p)
      return p

    p = s.Params.StopLoss
    if not p.Enabled:
      # not enabled so nothing to do
      return;

    # need StrategyReturn and VolExante for the calculations below
    s.CalcPerformance()
    s.CalcRisk()

    v = s.VolExante.shift(Date=p.ImplementationLag)/math.sqrt(250/p.AggPeriods)
    r = s.StrategyReturn.rolling(Date=p.AggPeriods).sum()

    triggered = (r.to_pandas().abs() > v*p.VolThreshold)*1
    filt = 1-np.linspace(0., 1., p.ExposureRecovery)

    lev = convolve(triggered, filt, mode='same')
    lev[lev>1] = 1.
    lev = 1 - lev*(1-p.ExposureReduction)

    # adjust Weights
    s.Weight.values = s.Weight * lev[:, None]
    return

  def CalcPerformance(x):
    p = x.Params.CalcPerformance
    
    w_ = x.Weight

    contrib = (w_ * x.Return)
    contrib[w_.isnull().all(dim='ID'), :] = np.nan
    x.ReturnContribution.values = contrib

    ret = contrib.sum(axis=1)
    x.StrategyReturn.values = ret

    # turnover and transaction costs
    dwgt = np.abs(x.Weight - x.Weight.shift(Date=-1))
    to = dwgt.sum(dim='ID')
    tc = to * p.BidAsk/2
    net = ret - tc
    x.StrategyReturnNet.values = net
    x.Turnover.values = to
    x.TurnoverAnn.values = x.Turnover.rolling(Date=250).sum()
    x.TC.values = tc

    x.ReturnCum.values = r2p(x.Return)
    x.ReturnContributionCum.values = r2p(x.ReturnContribution)
    x.StrategyReturnCum.values = r2p(x.StrategyReturn)
    x.StrategyReturnNetCum.values = r2p(x.StrategyReturnNet)

  def CalcAllPerformance(self, plot=True):
    self.Prepare()
    self.DataCleaning()
    self.SetRebalanceDays()
    self.GenerateSignals()
    self.ConstructWeights()
    self.VolTargeting()
    self.StopLoss()    
    self.CalcPerformance()
    if plot:
      self.PlotPerformance()


  def RecalcAllPerformance(self):
    self.SetInclude()
    self.SetRebalanceDays()
    self.GenerateSignals()
    self.ConstructWeights()
    self.VolTargeting()
    self.StopLoss()
    self.CalcPerformance()

  #endregion


  #region calc other
  def CalcAll(self):
    self.CalcAllPerformance()
    self.CalcRisk()
    self.CalcAnalytics()
    
  def CalcRisk(self, w_agg=5, w_roll=250):

    # aggregate to weekly returns
    wret = self.Return.rolling(Date=w_agg).sum(dim='Date')

    # adjust annualisation factor to reflect weekly returns
    AF = w_roll/w_agg

    # simple risk model
    S = RiskModel.SimpleRiskModel(wret, self.Weight, w_roll=w_roll, AF=AF)

    # define what we want to copy out of S
    copy_mx = [
       #'Date',
       #'ID',
       #'ID2',
       'VolWeighted',
       #'Cov',
       'VolContribution',
       'CorrToStrategy',
       'CorrWeighted',
       'VolExante',
       #'Corr',
       'Herfindal',
       'CorrAverage',
       'VolAverage',
       'VolDispersion',
       'Vol',
       'CorrDispersion',
       'DiversificationRatio',
       'VolCrowding',
       'CorrCrowding']

    for m in copy_mx:
      self[m].values = S[m]

    # realised strategy vol
    swret = self.StrategyReturn.rolling(Date=w_agg).sum(dim='Date')
    self.StrategyVol.values = swret.rolling(Date=w_roll).std()*math.sqrt(AF)

    # simple risk model for crowing
    S0 = RiskModel.SimpleRiskModel(wret, self.Weight*0+1/len(self.ID), w_roll=w_roll, AF=AF)

    self.Crowding.values = S.VolExante / S0.VolExante - 1

    # calc corr for all IDs in the risk model, not just invested in this strategy
    RM = RiskModel.from_pickle("RiskModel")
    RM.CalcCov()

    W = self.Weight
    W2 = xr.DataArray(W, dims=('Date', 'ID2'), coords=[W.Date, W.ID])

    # corr of ID to Fund
    Cov_Strat_ID = (W2 * RM.Cov).sum(dim='ID2')
    Vol_Strat = (W * RM.Cov * W2).sum(dim=['ID', 'ID2'])**0.5
    Corr_Strat_ID = Cov_Strat_ID / RM.Vol / Vol_Strat    

    self.attrs["Risk"] = {"Corr_Strat_ID": Corr_Strat_ID};

  def CalcAnalytics(self):
    self.Intensity.values = (self.Weight * self.RawSignal).sum(dim='ID')
    self.SignalWeighted.values = self.Intensity
    self.SignalAverage.values = self.RawSignal.mean(dim='ID')
    self.SignalDispersion.values = self.RawSignal.std(dim='ID')

    self.Leverage.values = np.abs(self.Weight).sum(axis=1)
    self.NetExposure.values = self.Weight.sum(dim='ID')

    # calc auto corr on monthly basis
    self.SignalAutoCorr1m.values = xr_corr(self.RawSignal.T, self.RawSignal.shift(Date=20).T)
    self.SignalAutoCorr.values = xr_corr(self.RawSignal.T, self.RawSignal.shift(Date=1).T)
    
    self.SignalWeightCorr.values = xr_corr(self.RawSignal.T, self.Weight.T)

    fwd_ret = self.ReturnCum.shift(Date=-20) / self.ReturnCum - 1
    self.IC.values = xr_corr(self.RawSignal.T, fwd_ret.T)
    self.ICAnn.values = self.IC.rolling(Date=250).mean()

    #calc duration of the weights
    # TODO check if this should be done on Weight, RawWeight or Signal 
    # and if shifted by implemtation lag or not
    wgt = self.LagWeight(self.RawWeight)
    wgt1 = wgt.shift(Date=1)
    dur = (np.sign(wgt) == np.sign(wgt1)).all(dim='ID')+0

    dur[0] = 0
    for iD in range(0, len(self.Date)):
      if dur[iD]:
        dur[iD] = dur[iD-1]+1
      else:
        dur[iD] = 0

    self.SignalDuration.values = dur

  def CalcAlphaDecay(s):
    delays = list(range(0, 250))
    srs = []
    SR = np.full(len(delays), fill_value=np.nan)
    for d, delay in enumerate(delays):
      sr = (s.LagWeight(s.RawWeight).shift(Date=delay) * s.Return).sum(dim='ID')
      SR[d] = SharpeRatio(sr)
      srs.append(sr)

    srs = xr_concat(srs)
    plt.plot(delays, SR)
    plt.show()

    r2p(srs.iloc[:, 0::10]).plot()

    return SR

  def CalcPayoffProfile(s, agg=20, degree=3, quantiles=10):
    ret = s.Return.rolling(Date=agg).sum()
    cont = ret*s.LagWeight(s.RawWeight)

    if False:
      plt.scatter(ret, cont, alpha=0.1)
      plt.show()

    x = pd.concat([cont.to_pandas().unstack(), ret.to_pandas().unstack()], axis=1)
    x.columns = ('Contribution', 'Return')

    x = x[~x.Return.isnull()]
    x['ReturnBucket'] = pd_quantiles(x.Return, quantiles)

    gm = x.groupby('ReturnBucket').mean()
    gs = x.groupby('ReturnBucket').std()

    p = np.polynomial.polynomial.polyfit(x.Return, x.Contribution, degree)
    xx = np.linspace(gm.Return.min(), gm.Return.max(), 100)
    yy = np.polynomial.polynomial.polyval(xx, p)
    if True:
      plt.scatter(gm.Return, gm.Contribution)
      plt.plot(xx, yy)
      plt.show()
    
    if False:
      sns.boxplot(x="ReturnBucket", y="Contribution", data=x)
      plt.show()


  #endregion

  #region Plot
  def PlotPerformance(self):

    y = xr_concat([self.StrategyReturn, self.StrategyReturnCompare])
    print(y.rolling(5).sum().corr())
    r2p(y).plot()

    y = xr_concat([self.StrategyReturn, self.StrategyReturnNet, self.TC])
    r2p(y).plot()

  def PlotAnalytics(self):
    plots = [
        ['Turnover', 'TurnoverAnn'],
        ['VolCrowding', 'CorrCrowding', 'Crowding'],
        ['Leverage', 'NetExposure'],
        ['DiversificationRatio', 'Herfindal'],
        ['Intensity', 'SignalAverage', 'SignalDispersion'],
        ['SignalDuration'],
        ['SignalWeightCorr', 'SignalAutoCorr'],
        ['StrategyVol', 'VolExante'],
        ['VolWeighted', 'VolAverage', 'VolDispersion'],
        ['CorrWeighted', 'CorrAverage', 'CorrDispersion'],
        ['ICAnn'],
        ['StrategyReturnCum', 'StrategyReturnNetCum']
      ]

    for ms in plots:
      ms_ = self[ms].to_dataframe()
      ms_.plot()
      plt.legend(ms, loc='upper left')
      plt.show()
      sns.violinplot(data=ms_)
      plt.show()


    ms = ['CorrWeighted', 'CorrAverage', 'CorrDispersion', 'CorrCrowding']
    df = self[ms].to_dataframe()
    df['CorrCrowding'] = pd_quantiles(df.CorrCrowding, 2, axis=0)

    df = df.melt(['CorrCrowding'])
    sns.violinplot(data=df, x='variable', y='value', split=True, hue='CorrCrowding')

  def PlotWeights(self):

    r2p(self.Return.to_pandas()).plot()
    plt.title('Return')

    self.RawSignal.to_pandas().plot()
    plt.title('Raw Signal')

    self.Signal.to_pandas().plot()
    plt.title('Signal')

    self.RawWeight.to_pandas().plot()
    plt.title('Raw Weight')

    self.Weight.to_pandas().plot()
    plt.title('Weight')

  def PlotEventPaths(self, regime='Crowding', off=90, buckets=5):
    Return = self.StrategyReturn.to_pandas()

    Regime = EventPath.PrepRegime(self[regime], buckets)

    # strategy performance by crowding
    with Timer('Strategy'):
      EventPath.EventPaths(Return, Regime, off=off)

    # basket performance by crowding
    with Timer('Basket'):
      EventPath.EventPortfolioPaths(
        self.Return.to_pandas(), 
        self.Weight.to_pandas(), 
        Regime, 
        off=off, plot=True)

    # rebalance paths by asset
    EventPath.EventPathByAsset(
      self.Return.to_pandas(), 
      self.Weight.to_pandas(), 
      Regime*0, 
      off=90, plot=True)

  #endregion


  #region Backtesting
  def BacktestParams1(self, prms, plot=False, f_perf = SharpeRatio):
    '''
    Similar to BacktestParamsX but here if several parameters are passed in prm
    each parameter will be tried with the original set of parameters
    i.e. we get n+m tries rather than n*m in the X version

    as a result we get line plots, whereas before we had image plots or 3d plots etc
    '''
    import warnings
    
    # remember original params
    p = self.Params
    # copy self, any changes only on s_, not self
    s_ = self.copy(deep=True)
    
    
    if isinstance(prms, dict):
      prms = [prms]

    # allocate result
    res = []

    with warnings.catch_warnings(record=True) as w, Timer('Loop'):
      # cause all warnings to be triggered only once
      warnings.simplefilter("once")

      for iPs, prm in enumerate(prms): # for each parameter

        # allocate for results
        res_shape = [len(self.Date), len(prm["Values"])]
        res_ = np.full(res_shape, fill_value=np.nan)

        for iP, val in enumerate(prm["Values"]): # for each parameter value
          # assign new parameter value to strategy s_
          s_ = self.copy(deep=True)
          p_ = p.copy()
          p_[prm["Group"]][prm["Param"]] = val
          s_.attrs["Params"] = p_

          s_.RecalcAllPerformance()
          res_[:, iP] = s_.StrategyReturn
    
        if plot:
          x = prm["Values"]
          if isinstance(x, np.ndarray):
            x = list(range(0, len(x)))

          plt.plot(x, f_perf(res_))
          plt.title(prm["Group"] + "." + prm["Param"])
          plt.show()

        res.append(res_)
      
        
    return res

  def BacktestParamsX(self, prms, item="StrategyReturn", plot=False, f_perf = SharpeRatio):
    '''
      Function backtests parameter by calculating strategy performance with new parameters
      prm - dict with elements Group, Param and Values or a list of such dicts
            if a list is provided each combination of all parameters is calculated
      
      e.g. 

      prms = [
        {'Group': "CalcPerformance", 
        'Param': "ImplementationLag", 
        'Values': list(range(0, 40))},

        {'Group': "GenerateSignals", 
        'Param': "Lookback", 
        'Values': list(range(50, 500, 50))}
      ]

      res = s.BacktestParamsX(prms) will create a 2d grid of results
      each point in the grid represents a combination of 
      (ImplementationLag, Lookback) parameter values

      res
    '''
    import warnings
    
    # remember original params
    p = self.Params
    # copy self, any changes only on s_, not self
    s_ = self.copy(deep=True)
    
    if isinstance(item, str):
      item = [item]

    if isinstance(prms, dict):
      prms = [prms]

    # check if several params provided and prepare for recursion
    # split prm into prm and prms for recursion
    res_shape = [len(prm["Values"]) for prm in prms]
    prm, *prms = prms

    # allocate for results
    res = np.full([len(self.Date), len(item)] + res_shape, fill_value=np.nan)

    with warnings.catch_warnings(record=True) as w, Timer('Loop'):
        # cause all warnings to be triggered only once
        warnings.simplefilter("once")
    
        for iP, val in enumerate(prm["Values"]):
            # assign new parameter value to strategy s_
            s_ = self.copy(deep=True)
            p_ = p.copy()
            p_[prm["Group"]][prm["Param"]] = val
            s_.attrs["Params"] = p_

            # calc strategy, directly or recursively
            if len(prms)>0:
                res[:, :, iP, :] = s_.BacktestParamsX(prms, item=item, plot=False)
            else:
                s_.RecalcAllPerformance()
                res_ = [s_[i] for i in item]
                res_ = xr_concat(res_).values
                res[:, :, iP] = res_

    if plot:
      sr = f_perf(res)
      cPrms = len(sr.shape)
      if cPrms == 1:
        sns.plot(prm["Values"], sr)
        plt.title(prm["Group"]+"."+prm["Param"])

      else:
        if "Labels" in prm:
          ylabels = prm['Labels']
        else:
          ylabels = prm['Values']
        if "Labels" in prms[0]:
          xlabels = prms[0]['Labels']
        else:
          xlabels = prms[0]['Values']

        if cPrms > 2:
          agg_axis = tuple(range(2, cPrms))
          sr = sr.max(axis=agg_axis)

        sns.heatmap(sr,
                    yticklabels = ylabels,
                    xticklabels = xlabels)
        plt.ylabel(prm["Group"]+"."+prm["Param"])
        plt.xlabel(prms[0]["Group"]+"."+prms[0]["Param"])
   
    return res
  
  def BacktestParamsSingle(self, prms):
    # remember original params
    p_ = self.Params.copy()

    # copy self, any changes only on s_, not self
    s_ = self.copy(deep=True)
    
    for prm in prms:
      p_[prm["Group"]][prm["Param"]] = prm["Value"]

    s_.attrs["Params"] = p_

    s_.RecalcAllPerformance()
    #res = s_.StrategyReturn

    return s_

  def BacktestParam_Includes(self, method='leave1out', arg=1, count=None):
    if method == 'leave1out':
      INCL = 1-np.eye(len(self.ID)+1, dtype=int)
      INCL = INCL[:, 1:]

    elif method == 'leavenout':
      import itertools
      COMB = itertools.combinations(range(len(self.ID)), arg)
      INCL = [np.ones(len(self.ID))]
      for iC in enumerate(COMB):
        INCL.append(iC)
        if len(INCL) == count:
          break

    elif method == 'random':
      INCL = (np.random.random([count, len(self.ID)])>arg)*1
      INCL[0, :] = 1

    LABEL = ['x'+'x'.join(self.ID[iI==0].values) for iI in INCL]
   
    return INCL, LABEL

  def BacktestParam_OLD(self, prm, plot=True):
    import warnings
    
    # remember original params
    p = self.Params 
    
    # copy self, all changes on s_ should not impact self
    s_ = self.copy(deep=True)
    
    res = np.full([len(self.Date), len(prm["Values"])], np.nan)
    
    
    with warnings.catch_warnings(record=True) as w, Timer('Loop'):
        # cause all warnings to be triggered only once
        warnings.simplefilter("once")
    
        for iP, val in enumerate(prm["Values"]):
            p_ = p.copy()
            p_[prm["Group"]][prm["Param"]] = val
            s_.attrs["Params"] = p_

            s_.RecalcAllPerformance()

            res[:, iP] = s_.StrategyReturn
    
    if plot:
        sr = res.mean(axis=0)/res.std(axis=0)*math.sqrt(250)
        plt.plot(sr)
        plt.show()
    
    return res

  def BacktestParams2(self, prm1, prm2):
    import warnings
    
    p = self.Params
    s_ = self

    res = np.full([len(self.Date), len(prm1["Values"]), len(prm2["Values"])], np.nan)
    
    with warnings.catch_warnings(record=True) as w, Timer('Loop'):
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
    
        for iP1, val1 in enumerate(prm1["Values"]):
            p_ = p.copy()
            p_[prm1["Group"]][prm1["Param"]] = val1
            for iP2, val2 in enumerate(prm2["Values"]):
                p_[prm2["Group"]][prm2["Param"]] = val2
                s_.attrs["Params"] = p_

                s_.RecalcAllPerformance()

                res[:, iP1, iP2] = s_.StrategyReturn
    
    sr = res.mean(axis=0)/res.std(axis=0)*math.sqrt(250)
    sns.heatmap(sr, xticklabels=prm2["Values"], yticklabels=prm1["Values"])    
    return res

  #endregion


  #region Usage Examples
  def Example_VolTargeting_StopLoss(s):
    # define params
    ssl_params = StrategyParams({
        'VolThreshold': 1.8, 
        'ExposureReduction': 0.1, 
        'ImplementationLag': 1, 
        'AggPeriods': 1, 
        'Enabled': True, 
        'ExposureRecovery': 10})
    s.Params["StopLoss"] = ssl_params

    svt_params = StrategyParams({
        'Enabled': True, 
        'VolTarget': 0.25, 
        'LeverageLimit': 1.,
        'ImplementationLag': 2})

    s.Params["VolTargeting"] = svt_params

    # additional calcs
    S = [s, s.copy(), s.copy(), s.copy()]
    S[1].VolTargeting()
    S[2].StopLoss()

    S[3].VolTargeting()
    S[3].StopLoss()

    # recalc performance
    [s.CalcPerformance() for s in S]

    y = xr_concat([s.StrategyReturn for s in S])
    y.columns = ["Original", "Vol-Target", "Stop-Loss", "Both"]
    r2p(y).plot()
    print(SharpeRatio(y))
    return S
  
  def Example_Backtest(self):
    prms = [
        {'Group': "CalcPerformance", 
        'Param': "ImplementationLag", 
        'Values': list(range(0, 10)) + list(range(10, 50, 5))},

        {'Group': "GenerateSignals", 
        'Param': "Lookback", 
        'Values': list(range(50, 500, 50))},

        {'Group': "GenerateSignals", 
        'Param': "LookbackExclude", 
        'Values': list(range(0, 40, 5))}
    ]

    res1 = self.BacktestParamsX(prms[:2], plot=True)
    res2 = self.BacktestParams1(prms, plot=True)

    # override default
    prm = {
      "CalcPerformance": {"ImplementationLag": 10},
      "GenerateSignals": {"LookbackExclude": 15}
    }
    self.Params.merge(prm, overrideOther = False)
    self.CalcAllPerformance()

    return res1

  def Example_TestAll(self):
    s.Load()
    s.CalcAll()
    s.CalcAlphaDecay()
    s.CalcPayoffProfile()

    s.PlotAnalytics()
    s.PlotWeights()
    s.PlotEventPaths('SignalDuration', 90, 3)
    s.Example_Backtest()

  def Example_CSvsTS(s):
    s.Load()
    sTS = s.copy()

    s.Params.ConstructWeights.CSorTS = 1
    s.CalcAll()

    sTS.Params.ConstructWeights.CSorTS = 0
    sTS.CalcAll()

    y = xr_concat([sTS.StrategyReturn, s.StrategyReturn, s.StrategyReturnCompare])
    y = r2p(y)
    y.columns = ['TS', 'CS', 'Original']
    y.plot()

    print(SharpeRatio(y))

    return [s, sTS]

  def Example_Processes(self):
    ## process for single strategy
    s = Strategy.StrategyFactory('FX-Carry')
    s.Load()
    s.CalcAll()
    s.to_pickle()    

    ## short cut for all implemented strategies
    Strategy.CalcAllImplemented()

    ## process for RiskModel
    RM = RiskModel()
    RM.to_pickle()
    RM = RiskModel.from_pickle("RiskModel")
    RM.CalcCov()


    ## process for fund
    p = Strategy.StrategyFactory('Fund')
    p.Load()
    p.CalcAll()
    p.to_pickle()

    ## powerbi->sql server view->web api process
    p = Strategy.from_pickle('Fund')
    res = p.GetTimeSeries('VolContribution')
    res[0].to_pandas().plot()


#endregion

#endregion

#region Strategy_xxx
class Strategy_Momentum(Strategy):
  def GenerateSignals(x):
    p = x.Params.GenerateSignals
    if "TechnicalSignal" not in p:
      p["TechnicalSignal"] = "Momentum"
	
    if p["TechnicalSignal"] == 'Momentum':
      x.RawSignal.values = \
        x.MomentumSignal.shift(Date=p.LookbackExclude) / \
        x.MomentumSignal.shift(Date=p.Lookback) - 1

    elif p["TechnicalSignal"] == 'Reversal':
      x.RawSignal.values = \
        x.MomentumSignal.shift(Date=p.Lookback) / \
        x.MomentumSignal.shift(Date=p.LookbackExclude) - 1

    elif p["TechnicalSignal"] == 'LowVol':

      x.RawSignal.values = - (x
        .MomentumSignal
        .to_pandas()
        .pct_change()
        .rolling(p.Lookback, axis=0)
        .std())

    elif p["TechnicalSignal"] == 'LowSkew':

      x.RawSignal.values = - (x
        .MomentumSignal
        .to_pandas()
        .pct_change()
        .rolling(p.Lookback, axis=0)
        .skew())


    elif p["TechnicalSignal"] == 'LowKurt':
      x.RawSignal.values = - (x
        .MomentumSignal
        .to_pandas()
        .pct_change()
        .rolling(p.Lookback, axis=0)
        .kurt())

    else:
      error('Not implemented')


class Strategy_Trend(Strategy_Momentum):
  def ConstructWeights(x):
    warning('obsolete, use Strategy_Momentum with ConstructWeights.CSorTS = 0');
    p = x.Params.ConstructWeights
    
    raw = x.RawSignal.values

    sig = raw
    
    wgt = np.sign(sig)/len(x.ID)

    x.Signal.values = sig
    x.RawWeight.values = wgt

    wgt = pd.DataFrame(wgt).rolling(p.SmoothRebalance, axis=0).mean().values

    x.Weight.values = wgt

    # and now we shift
    x.Weight.values = x.LagWeight(x.Weight)
    
class Strategy_FXValue(Strategy):
  def GenerateSignals(x):
    p = x.Params.GenerateSignals
    ppp = 1/x.PPP.shift(Date=p.SignalLag)
    x.RawSignal.values = ppp / x.Price - 1

class Strategy_Carry(Strategy):
  def GenerateSignals(x):
    x.RawSignal.values = 1 - x.F12 / x.F0
	
class Strategy_RiskModel(Strategy): 
  def Prepare(s):
    pass

  def GenerateSignals(s):
    pass

  def ConstructWeights(s):
    pass

class Strategy_Fund(Strategy):
  def LoadAdditionalData(s):
    s.ForceCalcOnAllWeekdays()
    Return = []
    for iID, ID in enumerate(s.ID.values):
      s_ = Strategy.from_pickle(Strategy.DisplayNames2StrategyNames[ID])
      DateMatch = (s.Date.values == s_.Date.values).all()
      if not DateMatch:
        raise Exception('Date do not match');
      
      # need cum return because all asset returns come in as cum from bb
      Return.append(s_.StrategyReturnCum) 

    Return = xr_concat(Return)
    s.Return.values = Return.values

  def GenerateSignals(s):
    p = s.Params.GenerateSignals
    p["Lookback"] = 250
    p["TargetVol"] = 0.1
    p["MinObs"] = 200
    p["TailImpliedVol"] = 0.5

    r = s.Return
    r = r.where(r!=0, np.nan)
    Vol = r.rolling(Date=p.Lookback, min_periods=p.MinObs)\
      .std() \
      .bfill('Date') \
      * math.sqrt(s.AnnualisationFactor)

    if p.TailImpliedVol > 0:
      VaR = r.rolling(Date=p.Lookback, min_periods=p.MinObs) \
        .construct('xxx') \
        .quantile(0.05, dim='xxx')
        
      iVol = RiskModel.TailImpliedVol(VaR, 0.05)*math.sqrt(s.AnnualisationFactor)

      avgVol = iVol.where(iVol > Vol, Vol) * p.TailImpliedVol + Vol * (1-p.TailImpliedVol)
    else:
      avgVol = Vol

    TargetVolAsset = p.TargetVol / RiskModel.PFVolOfSameCorrAssets(len(s.ID), rho=0)

    wgt = TargetVolAsset/avgVol

    s.RawSignal.values = wgt.values
    s.Signal.values = wgt.values

  def ConstructWeights(s):
    s.RawWeight.values = s.Signal.values
    s.Weight.values = s.RawWeight.values

  def LoadSubStrategies(p):
    S = [Strategy.from_pickle(Strategy.DisplayNames2StrategyNames[s]) for s in p.ID.values]
    return S

  def CalcRisk(p):
    # get risk model
    RM = RiskModel.from_pickle("RiskModel")
    RM.CalcCov()
    
    # load sub strategies
    S = p.LoadSubStrategies();

    # calc W of ID in Fund
    W_Strat_IDs = [s.Weight * p.Weight[:, iS] for iS, s in enumerate(S)]
    W_Strat_ID = xr.concat(W_Strat_IDs, pd.Index(p.ID.values, name="Strategy"))
    W_ID = W_Strat_ID.sum(dim='Strategy')
    W = W_ID

    # transpose W
    W2 = xr.DataArray(W, dims=('Date', 'ID2'), coords=[W.Date, W.ID]) 

    # corr of ID to Fund
    Cov_Fund_ID = (W2 * RM.Cov).sum(dim='ID2')
    Vol_Fund = (W * RM.Cov * W2).sum(dim=['ID', 'ID2'])**0.5
    Corr_Fund_ID = Cov_Fund_ID / RM.Vol / Vol_Fund
    Vol_ID = RM.Vol

    # contrib by ID
    VolContrib_ID = Corr_Fund_ID * Vol_ID * W_ID
    
    R = {
        'Cov_Fund_ID': Cov_Fund_ID,
        'Vol_Fund': Vol_Fund,
        'Corr_Fund_ID': Corr_Fund_ID,
        'Vol_ID': Vol_ID,
        'W_ID': W_ID,
        'VolContrib_ID': VolContrib_ID,
        'W_Strat_IDs': W_Strat_IDs
        }
    
    p.attrs["Risk"] = R

  def GetTimeSeries(p, Item):
    R = StrategyParams(p.attrs["Risk"])

    with Switch(Item) as case:
      if case('VolContribution'):
        res = [R.Corr_Fund_ID * R.Vol_ID * s for s in R.W_Strat_IDs]
      elif case('ReturnContribution'):
        S = p.LoadSubStrategies()
        res = [S[iS].ReturnContribution * p.Weight[:, iS] for iS, s in enumerate(S)]
      elif case('ReturnContributionCum'):
        S = p.LoadSubStrategies()
        res = [S[iS].ReturnContribution * p.Weight[:, iS] for iS, s in enumerate(S)]        
        res = [r2p(s)-1 for s in res]
      elif case('Weight'):
        res = R.W_Strat_IDs
      elif case('Corr'):
        res = R.Corr_Fund_ID

      elif case('R2'):
        res = R.Corr_Fund_ID**2
        
      elif case('FundExante'):
        res = R.Vol_Fund
      elif case('VolExante'):
        res = R.Vol_Fund
      elif case('FundVol'):
        res = p.StrategyReturn\
          .rolling(Date=5).sum()\
          .rolling(Date=250).std() * math.sqrt(52)

      return res
    # Return contrib
#endregion	