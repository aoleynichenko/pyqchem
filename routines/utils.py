#######################################
#   MODULE UTILS
#   Some utilities for pyqchem
#######################################

from __future__ import division
import sys
import math
import time
import numpy as np

def printmat(title, M):
  vsize = M.shape[0]
  hsize = M.shape[1]
  print "\t\t%s   %s" % (title, str(M.shape))
  for i in range(0,vsize):
    for j in range(0,hsize):
      print "%8.4f" % M[i,j],
    print
  print "\n"

