from scipy.stats import norm
import numpy as np
import pandas as pd
import xarray as xr
import time
import datetime
import matplotlib.pyplot as plt


from StrategyModule import *

PowerBIColorPalette = ["#01B8AA",
  "#374649",
  "#FD625E",
  "#F2C80F",
  "#5F6B6D",
  "#8AD4EB",
  "#FE9666",
  "#A66999"]

class ScatterPlotGrid:
  @staticmethod
  def Prepare(
        s, 
        Freq=20, 
        Items=["StrategyReturnCum", "Intensity", "CorrCrowding", "SignalDuration"], 
        Regime="Intensity",
        QuantileBuckets=3):

    #df = s.ReturnCum[::freq, 0:-1].to_pandas().pct_change().iloc[1:, :]
    df = []
    
    # at most a gird of 5
    Items = Items[:min(5, len(Items))]
    Items = Items+[Regime]
    
    for iI in Items:
        if iI in ['StrategyReturn', 'StrategyReturnCum']:
            # forward return at requested frequency
            sr = s.StrategyReturnCum[::Freq]
            df.append(sr.shift(Date=-1) / sr -1)
        else:
            df.append(s[iI][::Freq])

    # get data at that frequency
    df = xr_concat(df)

    # data cleaning
    df = df.loc[~df.isnull().any(axis=1), :]
    df = df.loc[df.iloc[:, 0]!=0, :]

    r = EventPath.PrepRegime(xr.DataArray(df.iloc[:, -1]), QuantileBuckets=3)
    df = df.iloc[:, :-1]
    
    return df, r
  
  @staticmethod
  def Seaborn(df):
    sns.set(style="white")

    g = sns.PairGrid(df, diag_sharey=False)
    g.map_lower(sns.kdeplot)
    g.map_upper(sns.scatterplot)
    g.map_diag(sns.kdeplot)

    labels = df.columns
    ax = g.axes
    cX = len(labels)

    for i in range(cX):
      ax[0, i].set_title(labels[i])

    [a.set_xlabel("") for a in ax.flat]
    [a.set_ylabel("") for a in ax.flat]

  @staticmethod
  def LinkedBrush(df, r):
    import mpld3
    from mpld3 import plugins

    X = df.values
    labels = df.columns
    cX = len(df.columns)

    fig, ax = plt.subplots(cX, cX, sharex="col", sharey="row", figsize=(8,8))
    fig.tight_layout()
    #fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
    #                    hspace=0.1, wspace=0.1)

    for i in range(cX):
      for j in range(cX):
        #points = ax[cX-1 - i, j].scatter(X[:, j], X[:, i], s=40, alpha=0.6)
        points = ax[i, j].scatter(X[:, j], X[:, i], s=40, alpha=0.6, c=r)

    for i in range(cX):
      ax[0, i].set_title(labels[i])
      #ax[cX-1,i].set_xlabel(labels[i])
      #ax[i,0].set_ylabel(labels[i])

    # remove tick labels
    #for axi in ax.flat:
    #    for axis in [axi.xaxis, axi.yaxis]:
    #        axis.set_major_formatter(plt.NullFormatter())

    # Here we connect the linked brush plugin
    plugins.connect(fig, plugins.LinkedBrush(points))

class EventPath:
  def __init__(self):
    pass

  @staticmethod
  def PrepRegime(Regime, ValueBuckets=None, QuantileBuckets=None):
    '''
    for a given regime variable create regimes
    based on either ValueBuckets (bounds of bins) 
    or QuantileBuckets (bounds of bins on number of bins)

    examples:
    EventPath.PrepRegime(s.Intensity, QuantileBuckets=3);
    EventPath.PrepRegime(s.Intensity, QuantileBuckets=[.2, .8]);
    EventPath.PrepRegime(s.Intensity, ValueBuckets=0);
    EventPath.PrepRegime(s.Intensity, ValueBuckets=[0, 0.1]);
    '''
    if QuantileBuckets is not None:
      if isinstance(QuantileBuckets, int):
        QuantileBuckets = np.arange(1/QuantileBuckets, 1-0.00001, 1/QuantileBuckets)

      ValueBuckets = Regime.quantile(QuantileBuckets).values

    ValueBuckets = np.concatenate([
        Regime.min().values.reshape(-1),
        np.array(ValueBuckets).reshape(-1)
        ])

    Regimes = (Regime.values[:, None] >= ValueBuckets[None, :]).astype(int).sum(axis=1)-1

    Regimes = pd.Series(Regimes)
    Regimes.index = Regime.to_pandas().index

    return Regimes


  @staticmethod
  def EventPortfolioPaths(Return, Weight, Regime, off=90, plot=True):
    if isinstance(off, int):
      #make a list
      off=[0, 90]

    #off_sel = off
    #off = [-260, 260]

    # define t's to loop over, skip first and last few
    ts = list(range(off[0], len(Return.index)-off[1]))
    # define offsets
    offs = np.array(range(off[0], off[1]+1))
    # alloc result 
    Path = np.full([len(ts), len(offs)], fill_value = np.nan)

    for iT, t in enumerate(ts):
      # get weights for ID on date t
      wgt_ = Weight.iloc[t, :]
      # get returns for IDs around date t
      ret_ = Return.iloc[t+offs, :]
      # calc return of the basket formed on date t
      Path[iT, :] = ret_.dot(wgt_).values


    # calc TRI across offsets
    Path = np.cumsum(Path, axis=1)
    # normalise
    Path = Path - Path[:, offs==0]
    Path = pd.DataFrame(Path, columns=offs)

    #Path = Path[:, offs>=off_sel[0]]
    #Path = Path[:, offs<=off_sel[1]]

    

    grs = Path.groupby(Regime[ts].values)
    mPath = grs.mean().T
    sPath = grs.std().T
  
    if plot:
      EventPath.PlotEventPaths(mPath, sPath, Path)

    return mPath, sPath, Path

  @staticmethod
  def EventPaths(Return, Regime, off=90, plot=True):
    Price = r2p(Return)
    if isinstance(off, int):
      #make a list
      off=[0, off]

    offs = list(range(off[0], off[1]+1))

    Path = [Price.shift(-off, axis=0) / Price - 1 for off in offs]

    Path = pd.concat(Path, axis=1)
    Path.columns = offs

    # remove shifted nans
    Path = Path.iloc[-off[0]:-off[1]-1, :]
    Regime = Regime[-off[0]:-off[1]-1]

    grs = Path.groupby(Regime.values)
    mPath = grs.mean().T
    sPath = grs.std().T
    
    if plot:
      EventPath.PlotEventPaths(mPath, sPath, Path)
    
    
    return mPath, sPath, Path#, Regime

  @staticmethod
  def EventPath(Return, Regime, off=90):
    raise Exception("Obsolete")
    '''

    '''
    sPath = []
    mPath = []

    Price = r2p(Return)

    offs = list(range(-off, off+1))
    for off in offs:
      r_ = Price.shift(off, axis=0) / Price - 1
      Path = pd.DataFrame({'Regime': Regime, off: r_})

      mPath.append(Path.groupby('Regime').mean().T)
      sPath.append(Path.groupby('Regime').std().T)

    mPath = pd.concat(mPath, axis=0)
    sPath = pd.concat(sPath, axis=0)

    EventPath.PlotEventPaths(mPath, sPath)
    
    return mPath, sPath, None
  
  @staticmethod
  def EventPathByAsset(Return, Weight, Regime, off=90, plot=True):
    raise Exception("Obsolete")
    # rebalance paths by asset
    mPaths = []
    sPaths = []

    IDs = Return.columns
    for iID, ID in enumerate(IDs):
      with Timer(ID):
        mPath, sPath, _ = EventPath.EventPortfolioPaths(
            Return.iloc[:, iID:iID+1], 
            Weight.iloc[:, iID:iID+1], 
            Regime, 
            off=off,
            plot=False)

        mPaths.append(pd_addColumnLevel(mPath, ID))
        sPaths.append(pd_addColumnLevel(sPath, ID))

    mPaths = pd.concat(mPaths, axis=1)
    sPaths = pd.concat(sPaths, axis=1)

    return mPaths, sPaths

  @staticmethod
  def Plotd3(mPath, sPath=None, plot=False, InteractiveLegend=False):
    import mpld3
    from mpld3 import plugins
    # plot line + confidence interval
    fig, ax = plt.subplots()
    plt.tight_layout()
    ax.grid(True, alpha=0.3)
    ax.set_frame_on(False)
    colors = PowerBIColorPalette

    if sPath is None:
      sPath = mPath * 0

    for iK, key in enumerate(mPath.columns):
        mPath_ = mPath[key]
        
        l, = ax.plot(mPath_.index, mPath_.values, label=key, color=colors[iK])

        sPath_ = sPath[key]
        ax.fill_between(sPath_.index.tolist(),
                    (mPath_.values-sPath_.values).tolist(), 
                    (mPath_.values+sPath_.values).tolist(), 
                    color=l.get_color(), alpha=.2)

    # define interactive legend
    handles, labels = ax.get_legend_handles_labels() # return lines and labels
    interactive_legend = plugins.InteractiveLegendPlugin(zip(handles, ax.collections),
      labels,
      alpha_unsel=0.1,
      alpha_over=1.5, 
      start_visible=True)
    plugins.connect(fig, interactive_legend)

    vals = ax.get_yticks()
    valsfmt = plotHelp.pctFormat(vals)

    ax.set_yticks(vals)
    ax.set_yticklabels(valsfmt)

    return mpld3.fig_to_dict(fig), fig

  @staticmethod
  def Plotd31(mPath, sPath=None, plot=False, InteractiveLegend=False):
    import mpld3
    from mpld3 import plugins
    # plot line + confidence interval
    fig, ax = plt.subplots()
    ax.grid(True, alpha=0.3)

    for key in mPath.columns:
        mPath_ = mPath[key]
        
        l, = ax.plot(mPath_.index, mPath_.values, label=key)

        if sPath is not None:
            sPath_ = sPath[key]
            ax.fill_between(sPath_.index.tolist(),
                        (mPath_.values-sPath_.values).tolist(), 
                        (mPath_.values+sPath_.values).tolist(), 
                        color=l.get_color(), alpha=.2)

    #from matplotlib.ticker import FuncFormatter
    #ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y))) 

    #vals = ax.get_yticks()
    #ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    if ~InteractiveLegend:
      v = [int(v) for v in mPath.columns.values]
      #v[0] = 'bottom'
      #v[-1] = 'top'
      plt.legend(v, loc='upper left')
    else:
      # define interactive legend
      handles, labels = ax.get_legend_handles_labels() # return lines and labels
      interactive_legend = plugins.InteractiveLegendPlugin(zip(handles, ax.collections),
        labels,
        alpha_unsel=0.1,
        alpha_over=1.5, 
        start_visible=True)
      plugins.connect(fig, interactive_legend)

    if plot:
      mpld3.display()

    #mpld3.save_html(fig, 'fig.html')
    #mpld3.fig_to_dict(fig)
    return fig
    

  @staticmethod
  def PlotEventPaths(mPath, sPath, Path=None):
    plt.figure()

    As = plt.plot(mPath)
    plt.legend(list(mPath.columns))
    plt.show()


    As = plt.plot(mPath)
    Bs = plt.plot(mPath+sPath, linestyle='--', linewidth=0.5)
    [b.set_c(a.get_c()) for a, b in zip(As, Bs)]

    Bs = plt.plot(mPath-sPath, linestyle='--', linewidth=0.5)
    [b.set_c(a.get_c()) for a, b in zip(As, Bs)]

    plt.show()    
    
    if not Path is None:
      plt.imshow(Path, aspect='auto')
      plt.show()
        
      Path.iloc[0::100, :].T.plot(legend=None)
      plt.show()


class Switch:
  def __init__(self, value): self._val = value
  def __enter__(self): return self
  def __exit__(self, type, value, traceback): return False # Allows traceback to occur
  def __call__(self, *mconds): return self._val in mconds

def winsorise(x, Method='std', q=2):
  if len(q)==1:
    q = [-q, q]

  with Switch(Method) as case:
    if case('std'):
        
      v_max = x.mean(axis=1) + q[1] * x.std(axis=1)
      v_min = x.mean(axis=1) + q[0] * x.std(axis=1)

      x[x>v_max] = v_max


    elif case('iqr', 'quantile'):
      v_max = x.quantile(q=q[1], axis=1)
      v_min = x.quantile(q=q[0], axis=1)
    elif case('values'):
      pass

  return x

class Timer(object):
    def __init__(self, name=None, filename=None):
        self.name = name
        self.filename = filename

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        message = 'Elapsed: %.2f seconds' % (time.time() - self.tstart)
        if self.name:
            message = '[%s] ' % self.name + message
        print(message)
        if self.filename:
            with open(self.filename,'a') as file:
                print(str(datetime.datetime.now())+": ",message,file=file)


def np_rank(x, axis=0):
    a = pd.DataFrame(x).rank(axis).values
    # if nan's present, they are kept as nans, and rank stops at lower count
    # we want nan's to be in the middle of the rank, i.e. top bottom
    # to be not nan's, i.e. need to scale the results accordingly
    # a = a / np.nanmax(a, axis=axis, keepdims=True) * a.shape[axis]
    return a

def np_rank01(x, axis=0):
    cv = (~np.isnan(x)).sum(axis=axis, keepdims=True)
    return (np_rank(x, axis=axis)-0.5)/cv

def np_quantiles(x, q, axis=0):
    return np.floor(np_rank01(x, axis) * q)

def pd_quantiles(x, q, axis=0):

    if not isinstance(x, (pd.DataFrame, pd.Series)):
        x = x.to_pandas()
    x = (x.rank(axis) - 0.5)
    x = x / (~x.isnull()).sum()
    x = np.floor(x * q)

    return x

def pd_addColumnLevel(df, C):
  df.columns = pd.MultiIndex.from_product([[C], df.columns])
  return df

def TransformSignal(sig, Transform, TransformParams=None):
    with Switch(Transform) as case:
        if case('rank'):
          sig = np_rank(sig, axis=1)
        
        elif case('quantile'):
          sig = np_rank01(sig)
          #cv = (~np.isnan(sig)).sum(axis=1, keepdims=True)
          #sig = (np_rank(sig, axis=1)-0.5)/cv
            
        elif case('proportional', 'demean'):
          sig = sig - np.nanmean(sig, axis=1, keepdims=True)
        
        elif case('passthrough', 'none'):
          pass
    
        elif case('normalise'):
            from scipy.stats import norm
            sig = norm.ppf(np_rank01(sig, axis=1))
            #cv = (~np.isnan(sig)).sum(axis=1, keepdims=True)
            #sig = np_rank(sig, axis=1)/cv-0.5/cv
            #sig = norm.ppf(sig);
            
        elif case('standardise'):
            sig = (sig - np.nanmean(sig, axis=1, keepdims=1))
            sig = sig / np.nanstd(sig, axis=1, keepdims=1)
            
        elif case('winsorise'):
            sig = winsor(sig, TransformParams)
            
        else:
            error('Not implemented');
    
    return sig

def Signal2Weight(sig, Method, MethodParam):
  wgt = np.full_like(sig, fill_value=np.nan);
 
  with Switch(Method) as case:
    if case('count'):
        wgt[sig <= MethodParam] = -1
        smax = np.nanmax(sig, axis=1, keepdims=True) - MethodParam + 1
        wgt[sig >= smax] = 1

    elif case('quantile'):
        wgt[sig<MethodParam] = -1;
        wgt[sig>1-MethodParam] = 1;
        
    elif case('proportional'):
        wgt = sig - np.nanmean(sig, axis=1, keepdims=1)
        
    elif case('passthrough', 'none'):
        wgt = sig
        
    elif case('threshold', 'value'):
        wgt[sig<-MethodParam] = -1
        wgt[sig>MethodParam] = 1
        
    else:
        error('Not implemented')
        
  return wgt

def NormaliseWeight(wgt, NormalisationMethod):
  with Switch(NormalisationMethod) as case:
    if case('sum0abs1'):
      # normalise to gross up to 1, net to 0
      wgt = wgt - np.nanmean(wgt, axis=1, keepdims=1)
      wgt = wgt / np.nansum(np.abs(wgt), axis=1, keepdims=1)
        
    elif case('none'):
      #nothing to do
      pass
    
    else:
      error('Not implemented')
        
    return wgt


def r2p(x):
  return (x+1).cumprod(axis=0)

class DatasetExt(xr.Dataset):
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

  def to_pickle(self, FileName):
    FileHandler = open(FileName, 'wb')
    pickle.dump(self, FileHandler)
    FileHandler.close()

  @staticmethod
  def from_pickle(FileName):
    FileHandler = open(FileName, 'rb')
    self = pickle.load(FileHandler)
    FileHandler.close()

    return self


class RiskModel(DatasetExt):
  def __init__(self, version="RiskModel", w_agg=5, w_roll=250, AF=52):
    super().__init__()

    self.attrs["w_agg"] = w_agg
    self.attrs["w_roll"] = w_roll
    self.attrs["AF"] = AF
    self.attrs["version"] = version

    s = Strategy.StrategyFactory(version)
    s.Load()
    s.DataCleaning()

    add_ds = s[["Return", "AssetClass", "Core"]]
    self.merge(add_ds, inplace=True)

  def to_pickle(self):
    super().to_pickle(self.attrs["version"])

  def CalcCov(self):
    ret = self.Return
    w_agg = self.attrs["w_agg"]
    w_roll = self.attrs["w_roll"]
    AF = self.attrs["AF"]

    #return
    coords = [ret.Date, ret.ID]
    
    ret = ret.rolling(Date=w_agg).sum()
    ret.name = "Return"

    # rolling moments
    ret2 = xr.DataArray(ret, dims=('Date', 'ID2'), coords=coords)
    ret12 = ret * ret2

    varxID = (ret**2).rolling(Date=w_roll).mean() * AF
    volxID = varxID**0.5
    volxID2 = xr.DataArray(volxID, dims=('Date', 'ID2'), coords=coords)
    cov = ret12.rolling(Date=w_roll).mean() * AF
    corr = cov/volxID/volxID2

    # add to self
    volxID.name = 'Vol'
    cov.name = 'Cov'
    corr.name = 'Corr'

    add_ds = xr.merge([volxID, cov, corr])
    self.merge(add_ds, inplace=True)

  def CalcRisk(self, wgt):
    # select only required IDs
    RM = self.sel({"ID": wgt.ID.values, "ID2": wgt.ID.values})
    cov = RM.Cov
    corr = RM.Corr
    volxID = RM.Vol

    # calc
    coords = [wgt.Date, wgt.ID]
    wgt2 = xr.DataArray(wgt, dims=('Date', 'ID2'), coords=coords)

    var_contr = (wgt * cov * wgt2).sum(dim='ID2')
    var = var_contr.sum(dim='ID')
    vol = var**0.5
    var_contr_pct = var_contr / var
    vol_contr = var_contr / vol

    wgt_abs = np.abs(wgt)
    lev = wgt_abs.sum(dim='ID')
    vol_wgt = (volxID * wgt_abs).sum(axis=1)
    corr_wgt = (wgt * corr * wgt2).sum(dim=('ID', 'ID2'))/lev
    dummy = corr_wgt*np.nan

    S = xr.Dataset({
        'VolExante' : vol,
        'VolContribution' : vol_contr,
        'Vol' : volxID,
        'CorrToStrategy' : vol_contr / volxID / wgt,
        'Cov' : cov,
        'Corr' : corr,

        'VolWeighted': vol_wgt/lev,
        'VolAverage': volxID.mean(dim='ID'),
        'VolDispersion': volxID.std(dim='ID'),
      
        'DiversificationRatio': vol / vol_wgt * lev,
        'Herfindal': Herfindal(wgt_abs/lev),
      
        'CorrWeighted': corr_wgt,
        'CorrAverage': corr.mean(dim=('ID', 'ID2')),
        'CorrDispersion': corr.std(dim=('ID', 'ID2')),

        'VolCrowding': dummy,
        'CorrCrowding': dummy
        })

    S['VolCrowding'].values = S.VolWeighted / S.VolAverage - 1 # ratio easier to interprete since units vary  
    S['CorrCrowding'].values = S.CorrWeighted - S.CorrAverage # diff easier to interprete since corr already [-1 1]

    return S

  @staticmethod
  def SimpleRiskModel(ret, wgt, w_roll=250, AF=None):
    if AF is None:
      AF = w_roll

    coords = [ret.Date, ret.ID]
    ret2 = xr.DataArray(ret, dims=('Date', 'ID2'), coords=coords)
    ret12 = ret * ret2

    varxID = (ret**2).rolling(Date=w_roll).mean() * AF
    volxID = varxID**0.5
    volxID2 = xr.DataArray(volxID, dims=('Date', 'ID2'), coords=coords)
    cov = ret12.rolling(Date=w_roll).mean() * AF

    corr = cov/volxID/volxID2

    wgt2 = xr.DataArray(wgt, dims=('Date', 'ID2'), coords=coords)

    var_contr = (wgt * cov * wgt2).sum(dim='ID2')
    var = var_contr.sum(dim='ID')
    vol = var**0.5
    var_contr_pct = var_contr / var
    vol_contr = var_contr / vol

    wgt_abs = np.abs(wgt)
    lev = wgt_abs.sum(dim='ID')
    vol_wgt = (volxID * wgt_abs).sum(axis=1)
    corr_wgt = (wgt * corr * wgt2).sum(dim=('ID', 'ID2'))/lev
    dummy = corr_wgt*np.nan

    S = xr.Dataset({
        'VolExante' : vol,
        'VolContribution' : vol_contr,
        'Vol' : volxID,
        'CorrToStrategy' : vol_contr / volxID / wgt,
        'Cov' : cov,
        'Corr' : corr,

        'VolWeighted': vol_wgt/lev,
        'VolAverage': volxID.mean(dim='ID'),
        'VolDispersion': volxID.std(dim='ID'),
      
        'DiversificationRatio': vol / vol_wgt * lev,
        'Herfindal': Herfindal(wgt_abs/lev),
      
        'CorrWeighted': corr_wgt,
        'CorrAverage': corr.mean(dim=('ID', 'ID2')),
        'CorrDispersion': corr.std(dim=('ID', 'ID2')),

        'VolCrowding': dummy,
        'CorrCrowding': dummy
        })

    S['VolCrowding'].values = S.VolWeighted / S.VolAverage - 1 # ratio easier to interprete since units vary  
    S['CorrCrowding'].values = S.CorrWeighted - S.CorrAverage # diff easier to interprete since corr already [-1 1]

    return S

  @staticmethod
  def TailImpliedVol(VaR, q):
    from scipy.stats import norm
    iVol = VaR/norm.ppf(q)
    return iVol

  @staticmethod
  def PFVolOfSameCorrAssets(n, rho, AssetVol=1):
    PFVol = math.sqrt(n+rho*(n-1)*n) * AssetVol
    return PFVol

  @staticmethod
  def PFVolOfSameCorrAssets_inv(n, PFVol, AssetVol=1):
    rho = ((PFVol/AssetVol)**2 - n)/n/(n-1)
    return rho

def datetime64_to_datetime(x):
    return (x.values.
        astype('datetime64[D]'). # need the 2 conversions to get to datetime
        astype(datetime.datetime))

def Herfindal(w):
    Herf = (w**2).to_pandas().sum(axis=1)
    N1 = 1/w.shape[1]
    Herf = Herf - N1
    Herf = Herf / (1-N1)

    return Herf


def xr_corr(x, y, axis=0):
  x = (x - x.mean(axis=axis))/x.std(axis=axis)
  y = (y - y.mean(axis=axis))/y.std(axis=axis)
  c2 = (x*y).mean(axis=axis)

  return c2

def SharpeRatio(x, AF=250):
  return x.mean(axis=0)/x.std(axis=0)*np.sqrt(AF)

def xr_concat(ds):
  ps = []
  for d in ds:
    p = d.to_pandas()
    p.name = d.name
    ps.append(p)

  return pd.concat(ps, axis=1)


class plotHelp:
  @staticmethod
  def pctFormat(vals):
    dval = vals[1]-vals[0]

    if dval > 0.99:
        valsfmt = vals
        pass
    elif dval > 0.0099:
        valsfmt = ['{0:0.0%}'.format(x) for x in vals]
    elif dval > 0.00099:
        valsfmt = ['{0:0.1%}'.format(x) for x in vals]
    elif dval > 0.000099:
        valsfmt = ['{0:0.2%}'.format(x) for x in vals]
    else:
        valsfmt = ['{0:0.3%}'.format(x) for x in vals]    
        
    return valsfmt