from StrategyUtils import *
from StrategyModule import *

Implemented_StrategyNames = ['DBPPPUSF', 'DBHVG10U', 'DBHVGUSI', 'DBMOMUSF', 'SGICCMX', 
              'UISMT6UE', 'UISRT6UE', 'UISET6UE', 'UISKT6UE', 'UISST6UE']
Implemented_DisplayNames = ['FX-Value', 'FX-Carry', 'FX-Carry-G10EM', 'FX-Mom', 'Comm-Carry',
                'Rates-Mom-MM', 'Rates-Mom', 'Equity-Mom', 'Comm-Mom', 'Equity-Mom-Sectors']
DisplayNames2StrategyNames = dict(zip(Implemented_DisplayNames, Implemented_StrategyNames))


class Strategy_DBPPPUSF(Strategy_FXValue):
  pass
   
class Strategy_DBMOMUSF(Strategy_Momentum):
  pass

class Strategy_DBHVG10U(Strategy):
  pass

class Strategy_DBHVGUSI(Strategy):
  pass

class Strategy_SGICCMX(Strategy_Carry):
  pass
  
class Strategy_UBSTrend(Strategy_Momentum):
  pass

class Strategy_UISRT6UE(Strategy_Momentum):
  pass

class Strategy_UISET6UE(Strategy_Momentum):
  pass

class Strategy_UISKT6UE(Strategy_Momentum):
  pass

class Strategy_UISFT6UE(Strategy_Momentum):
  pass

class Strategy_UISST6UE(Strategy_Momentum):
  pass

class Strategy_UISMT6UE(Strategy_Momentum):
  pass
