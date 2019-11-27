# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 17:19:06 2019

@author: bdai729
"""

# reference: http://lcapy.elec.canterbury.ac.nz/tutorial.html#transfer-functions

from lcapy import Circuit
from lcapy import s
from sympy import *
from IPython.display import display # to display math formula image, print function only prints text
init_printing(use_unicode = True,wrap_line=False, no_global=True)

cct = Circuit()
cct.add('V1 1 0 {u(t)}')
cct.add('R1 1 2')
cct.add('L1 2 0')
# <codecell> circuit analysis
import numpy as np
t = np.linspace(0, 0.01, 1000)
vc = cct.C.v.evaluate(t)

from matplotlib.pyplot import figure, savefig
fig = figure()
ax = fig.add_subplot(111)
ax.plot(t, vc, linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Capacitor voltage (V)')
ax.grid(True)

savefig('circuit-VRC1-vc.png')

# <codecell> transfer function (s domain)
cct = Circuit()
cct.add('V1 1 0 step Vs')
cct.add('R1 1 2')
cct.add('C1 2 0')
cct[2].V(s)
cct[1].V(s)
Gs = cct[2].V(s) / cct[1].V(s)
display(Gs)

# <codecell> transfer function (j*omega)
from lcapy import s, j, omega
from IPython.display import display # to display math formula image, print function only prints text
init_printing(use_unicode = True,wrap_line=False, no_global=True)
H = (s + 3) / (s - 4)
A = H(j * omega)
display(A)

# <codecell> CPT LC transfer function
cct = Circuit()
cct.add('Vin 1 0 step Vs')
cct.add('L1 1 2')
cct.add('Cc 2 3')
cct.add('Rload 3 0')
expr1 = cct[1].V(s)
expr2 = cct[2].V(s)
Gs = expr2 / expr1
display(Gs)
display(simplify(Gs))

# <codecell> CPT double sided LC transfer function
cct = Circuit()
cct.add('Vi 1 0 step Vs')
cct.add('L1 1 2')
cct.add('Cc 2 3')
cct.add('C1 2 0')
cct.add('C2 3 0')
cct.add('L2 3 4')
cct.add('Rac 4 0')
expr1 = cct[1].V(s)
expr2 = cct[4].V(s)
Gs = expr2 / expr1
display(Gs)
display(simplify(Gs))
print(simplify(Gs))
print(latex(simplify(Gs)))

# <codecell> schematic
from lcapy import Circuit
cct = Circuit()
cct.add('Vin 1 0 {V(s)}; down')
cct.add('L1 1 2; right')
cct.add('Cc 2 3; right')
cct.add('Rload 3 0_2; down') # Nodes start with underscore '_' will not be implemented
cct.add('W 0 0_2; right') # W: Wire object
cct.draw('schematic.pdf')

# <codecell> state space equations
from lcapy import Circuit
from IPython.display import display 
# to display math formula image, print function only prints text
from sympy.printing.mathml import print_mathml
init_printing(use_unicode = True,wrap_line=False, no_global=True)
a = Circuit("""
V 1 0 {v(t)}; down
R1 1 2; right
L 2 3; right=1.5, i={i_L}
R2 3 0_3; down=1.5, i={i_{R2}}, v={v_{R2}}
W 0 0_3; right
W 3 3_a; right
C 3_a 0_4; down, i={i_C}, v={v_C}
W 0_3 0_4; right""")
stateSpace = a.ss
display(stateSpace.x) # The state variable vector
display(stateSpace.x0) #The initial values of the state variable vector
# The output vector can either be the nodal voltages, the branch currents, 
# or both. By default the nodal voltages are chosen. 
# This vector is shown using the y attribute
display(stateSpace.y)
display(stateSpace.state_equations())

