# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 11:04:42 2019

@author: bdai729
"""

# <codecell> introduction
# this program does not work yet, I suspect this is because I have not
# successfully installed circuitikz package
from lcapy import Circuit
cct = Circuit()
cct.add('V 1 0 {V(s)}; down')
cct.add('R 1 2; right')
cct.add('C 2 _0_2; down')
cct.add('W 0 _0_2; right')
cct.draw('schematic.pdf')