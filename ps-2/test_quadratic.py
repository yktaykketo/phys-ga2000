#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 18:23:50 2023

@author: wangziyao
"""

import quadratic
from decimal import Decimal, getcontext
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
def test_quadratic():
      # Check the case from the problem
      x1, x2 = quadratic.quadratic(0.001, 1000., 0.001)
      assert (np.abs(x1 - Decimal(- 1.e-6)) < 1.e-10)
      assert (np.abs(x2 - Decimal(- 0.999999999999e+6)) < 1.e-10)
      # Check a related case to the problem
      x1, x2 = quadratic.quadratic(0.001, -1000., 0.001)
      assert (np.abs(x1 - Decimal(0.999999999999e+6)) < 1.e-10)
      assert (np.abs(x2 - Decimal(1.e-6)) < 1.e-10)
      # Check a simpler case (note it requires the + solution first)
      x1, x2 = quadratic.quadratic(1., 8., 12.)
      assert (np.abs(x1 - Decimal(- 2.)) < 1.e-10)
      assert (np.abs(x2 - Decimal(- 6)) < 1.e-10)
     
     
