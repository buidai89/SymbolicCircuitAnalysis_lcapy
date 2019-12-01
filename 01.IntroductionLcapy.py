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
from lcapy import Circuit
cct = Circuit("""
V 1 0 step 10; down
L 1 2 1e-3; right, size=1.2
C 2 3 1e-4; right, size=1.2
R 3 0_1 10; down
W 0 0_1; right
""")

import numpy as np
t = np.linspace(0, 0.01, 1000)
vc = cct.C.v.evaluate(t)

from matplotlib.pyplot import figure, savefig
import matplotlib.pyplot as plt
fig = figure()
ax = fig.add_subplot(111)
ax.plot(t, vc, linewidth=2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Capacitor voltage (V)')
ax.grid(True)
plt.tight_layout()
ax.show()


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

from lcapy import Circuit
cct = Circuit()
cct.add('R1 1 2')
cct.add('C1 2 0')
Hs = cct.transfer(1, 0, 2, 0)
display(Hs)
display(simplify(Hs))
# In this example, the transfer method computes (V[1] - V[0]) / (V[2] - V[0])
# there is no voltage input source in this example
# I don't understand why Hs = cct.transfer(2, 0, 1, 0) seems wrong

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

# <codecell> Transfer function transfomer non-ideal
from lcapy import Circuit, s, j, omega
from matplotlib.pyplot import savefig
from numpy import logspace
from sympy import *
import matplotlib.pyplot as plt

# Vi 1 0 step Vs; down
cct = Circuit("""
Ri 1 2 0.01; right
L1 2 0_1 12.5e-6; down
W1 0 0_1; right
W2 0_1 0_2; right
L2 3 0_2 12.5e-6; down=1.5
R2 3 4 0.01; right
Rload 4 0_3 10; down
W3 0_2 0_3; right
K1 L1 L2 0.5; size=1.5
;label_ids=false
""")
# Gs = cct[4].V(s) / cct[1].V(s)
# if this is used, need to put Vi 1 0 step Vs; down to the netlist
Gs = cct.transfer(1, 0, 4, 0)
display(Gs)

Hs = simplify(Gs)
display(Hs)
print(Hs)
from scipy import signal
sys = signal.TransferFunction([1600000, 0], [3,  3206400, 2562560000])
# get these numbers by print(Hs)

w, mag, phase = signal.bode(sys)
plt.figure(figsize=(9,6))
ax1 = plt.subplot(211)
plt.semilogx(w/2/3.14, mag)    # Bode magnitude plot
plt.ylabel('|V_Out/ V_in| (dB)')
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')

ax2 = plt.subplot(212, sharex = ax1)
plt.semilogx(w/2/3.14, phase)  # Bode phase plot
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (V_out/ V_in) (deg)')
plt.xlim(10, )
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')
plt.tight_layout()
plt.show()

# <codecell> Bodeplot & transfer function low-pass
from lcapy import Circuit, s, j, omega
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
from numpy import logspace
from sympy import *

# Vi 1 0 step Vs; down
cct = Circuit("""
Ri 1 2 10; right
C1 2 0_1 10e-6; down
W1 0 0_1; right
W2 0_1 0_2; right
W3 2 2_0; right
Rload 2_0 0_2 10; down
""")

# Gs = cct[2].V(s) / cct[1].V(s)
# if this is used, need to put Vi 1 0 step Vs; down to the netlist
Gs = cct.transfer(1, 0, 2, 0)
display(Gs)

Hs = simplify(Gs)
display(Hs)
print(Hs)

from scipy import signal
sys = signal.TransferFunction([10000], [1,  20000])
# get these numbers by print(Hs)
w, mag, phase = signal.bode(sys)

plt.figure(figsize=(9,6))
ax1 = plt.subplot(211)
plt.semilogx(w/2/3.14, mag)    # Bode magnitude plot
plt.ylabel('|V_Out/ V_in| (dB)')
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')

ax2 = plt.subplot(212, sharex = ax1)
plt.semilogx(w/2/3.14, phase)  # Bode phase plot
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (V_out/ V_in) (deg)')
plt.xlim(10, )
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')
plt.tight_layout()
plt.show()

# <codecell> Bodeplot & transfer function high-pass
from lcapy import Circuit, s, j, omega
from matplotlib.pyplot import savefig
import matplotlib.pyplot as plt
from numpy import logspace
from sympy import *

# Vi 1 0 step Vs; down
cct = Circuit("""
Vi 1 0 step Vs; down
C1 1 2 10e-6; right
W1 0 0_1; right
Rload 2 0_1 10; down
""")

Gs = cct[2].V(s) / cct[1].V(s)
# if this is used, need to put Vi 1 0 step Vs; down to the netlist
# Gs = cct.transfer(1, 0, 2, 0)
display(Gs)

Hs = simplify(Gs)
display(Hs)
print(Hs)

from scipy import signal
sys = signal.TransferFunction([1, 0], [1,  10000])
# get these numbers by print(Hs)
w, mag, phase = signal.bode(sys)

plt.figure(figsize=(9,6))

ax1 = plt.subplot(211)
plt.semilogx(w/2/3.14, mag)    # Bode magnitude plot
plt.ylabel('|V_Out/ V_in| (dB)')
plt.ylim(-50, 1)
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')

ax2 = plt.subplot(212, sharex = ax1)
plt.semilogx(w/2/3.14, phase)  # Bode phase plot
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (V_out/ V_in) (deg)')
plt.xlim(10, 2e4)
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')
plt.tight_layout()
plt.show()

# <codecell> Bodeplot Power Repeater
from lcapy import Circuit, s, j, omega
from matplotlib.pyplot import savefig
from numpy import logspace
from sympy import *
import matplotlib.pyplot as plt

# Vi 1 0 step Vs; down
cct = Circuit("""
Ri 1 2 0.01; right
L1 2 0_1 12.5e-6; down
W1 0 0_1; right
W2 0_1 0_2; right
L2 3 0_2 12.5e-6; down=1.5
R2 3 4 0.01; right
Rload 4 0_3 10; down
W3 0_2 0_3; right
K1 L1 L2 0.5; size=1.5
;label_ids=false
""")
# Gs = cct[4].V(s) / cct[1].V(s)
# if this is used, need to put Vi 1 0 step Vs; down to the netlist
Gs = cct.transfer(1, 0, 4, 0)
display(Gs)

Hs = simplify(Gs)
display(Hs)
print(Hs)
from scipy import signal
sys = signal.TransferFunction([1600000, 0], [3,  3206400, 2562560000])
# get these numbers by print(Hs)

w, mag, phase = signal.bode(sys)
plt.figure(figsize=(9,6))
ax1 = plt.subplot(211)
plt.semilogx(w/2/3.14, mag)    # Bode magnitude plot
plt.ylabel('|V_Out/ V_in| (dB)')
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')

ax2 = plt.subplot(212, sharex = ax1)
plt.semilogx(w/2/3.14, phase)  # Bode phase plot
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase (V_out/ V_in) (deg)')
plt.xlim(10, )
plt.minorticks_on()
plt.grid(b=True, which='major',axis='both', color='grey')
plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')
plt.tight_layout()
plt.show()

# <codecell> Bodeplot function
from lcapy import Circuit, s, j, omega
from matplotlib.pyplot import savefig
from numpy import logspace
from sympy import *
import matplotlib.pyplot as plt
from scipy import signal

def bodeplot(sys): 
    w, mag, phase = signal.bode(sys)
    plt.figure(figsize=(9,6))
    ax1 = plt.subplot(211)
    plt.semilogx(w/2/3.14, mag)    # Bode magnitude plot
    plt.ylabel('|V_Out/ V_in| (dB)')
    plt.minorticks_on()
    plt.grid(b=True, which='major',axis='both', color='grey')
    plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')
    
    ax2 = plt.subplot(212, sharex = ax1)
    plt.semilogx(w/2/3.14, phase)  # Bode phase plot
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (V_out/ V_in) (deg)')
    plt.xlim(10, )
    plt.minorticks_on()
    plt.grid(b=True, which='major',axis='both', color='grey')
    plt.grid(b=True, which='minor',axis='x',linestyle='--', color='grey')
    plt.tight_layout()
    plt.show()

def transferfunction(cct, N1p, N1m, N2p, N2m):
    Gs = cct.transfer(N1p, N1m, N2p, N2m)
    display(Gs)
    Hs = simplify(Gs)
    display(Hs)
    print(Hs)
    
cct = Circuit("""
Ri 1 2 0.01; right
L1 2 0_1 12.5e-6; down
W1 0 0_1; right
W2 0_1 0_2; right
L2 3 0_2 12.5e-6; down=1.5
R2 3 4 0.01; right
Rload 4 0_3 10; down
W3 0_2 0_3; right
K1 L1 L2 0.5; size=1.5
;label_ids=false
""")

transferfunction(cct, 1, 0, 4, 0)
sys1 = signal.TransferFunction([1600000, 0], [3,  3206400, 2562560000])
bodeplot(sys1)

    
    