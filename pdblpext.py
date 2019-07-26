# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 16:35:56 2018

@author: Andreas
"""

import os
import re
import numpy as np
import pandas as pd
import pdblp

class PDBLP2():
    def __init__(self, Tickers, cBatches = 50):
        self.Tickers = Tickers
        self.cBatches = cBatches
        
        self.con = pdblp.BCon(debug=False, port=8194, timeout=5000)
        self.con.start()
        
    def __del__(self):
        self.con.stop()
	
    def CheckTickers(self, Tickers):
        iValid = np.full((len(Tickers)), False)

        for iT, iTicker in enumerate(Tickers):
            try:
                self.con.ref(iTicker, "NAME")
                iValid[iT] = True
            except:
                print(iT)	

        return iValid
	
    @staticmethod
    def GetFileSetsInDir(FileDir = './parts/'):
        FileNames = os.listdir(FileDir)
        FileSets = set([re.sub('[0-9]+\.pkl', '', fn) for fn in FileNames])
        return list(FileSets)
	
    @staticmethod	
    def CombineParts(FileSet):
        pat = '{}[0-9]+\.pkl'.format(FileSet)

        files = [fn for fn in os.listdir('./parts/') if re.match(pat, fn)]
        len(files)
		
        parts = []
        for file in files:
            part = pd.read_pickle('./parts/' + file)
            parts.append(part)
			
        parts = pd.concat(parts, axis=1)
        parts.to_pickle('./{}.pkl'.format(FileSet))
		
        return parts
		
    @staticmethod	
    def CleanTickers(Tickers):
        iVal = [type(t)==str for t in Tickers]
        Tickers = Tickers[iVal]
        return Tickers
		
    @staticmethod
    def SplitIntoFileSets(Tickers):
        bTickers = [Tickers[i:i + cTickers] for i in range(0, len(Tickers), cTickers)]        
        return Tickers