# PyDesign

## Description

PyDesign is a Python library for designing and analyzing human-powered-aircraft (HPA). It is a collection of modules that can be used to design and analyze the performance of HPA. The library is designed to be modular and extensible. The library is currently in development and is not ready for use.

## Installation

First, clone the repository:

```bash
git clone https://github.com/xsuz/PYDesign.git
```

Then, install the package:

```bash
pip install -r requirements.txt
pip install xfoil-py
```

## Usage

### Beam Calculation Module

```python

from obj.Beam import Sheet,PLY,BeamInfo,Beam
import utils.unit as u
from astropy.units import Quantity
import pandas as pd
import numpy as np
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

HRX350C125S=Sheet(name="HRX350C125S",E=Quantity(40,"t/mm2"),thickness=Quantity(0.096,"mm"),density=Quantity(155,"g/(m2)"),compound_rate=0.664,use="0&45&90&802")
TR3110_381GMX=Sheet(name="TR3110 381GMX",E=Quantity(2,"t/mm2"),thickness=Quantity(0.223,"mm"),density=Quantity(333,"g/(m2)"),compound_rate=0.1,use="クロス",isotropy=True)
spec={1:HRX350C125S,5:TR3110_381GMX}

df=pd.read_excel("data/ply.xlsx",sheet_name=None)

beam=Beam([
    BeamInfo.read(df["A"],spec),
    BeamInfo.read(df["B"],spec),
    BeamInfo.read(df["C"],spec,np.array([0,-3,0])*u.deg),
    BeamInfo.read(df["D"],spec)],160)


with open("data/load.npz","rb") as file:
    f=np.load(file)

plt.style.use('dark_background')
ax:Axes=plt.axes(projection='3d')

# right wing
r,U_xi,euler=beam.deflect(f*u.N,0*np.ones((320,3))*u.m,
                    np.array([[0,1,0], # e_xi
                              [1,0,0], # e_eta
                              [0,0,-1] # e_zeta
                              ]).T)

ax.plot(r[:,0],r[:,1],r[:,2],color='white',linewidth=3)
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,0,1],U_xi[:,1,1],U_xi[:,2,1],length=3,color='green',alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\eta_r$")
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,0,2],U_xi[:,1,2],U_xi[:,2,2],length=3,color='blue' ,alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\zeta_r$")

# left wing
r,U_xi,euler=beam.deflect(f*u.N,0*np.ones((320,3))*u.m,
                    np.array([[0,-1,0], # e_xi
                              [1, 0,0], # e_eta
                              [0, 0,-1] # e_zeta
                              ]).T)

ax.plot(r[:,0],r[:,1],r[:,2],color='white',linewidth=3)
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,0,1],U_xi[:,1,1],U_xi[:,2,1],length=3,color='orange',alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\eta_l$")
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,0,2],U_xi[:,1,2],U_xi[:,2,2],length=3,color='red' ,alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\zeta_l$")

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
ax.set_title('Deflection')
# Get rid of the panes
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.axis('equal')

ax.view_init(elev=-135, azim=30)
plt.show()
```

### Airfoil Analysis Module

```python
from obj.Wing import WingInfo,WingElement
import pandas as pd
from utils.unit import *

if __name__=="__main__":
    wing_elements:list[WingElement]=[
        WingElement(name="A",inner_foil="./data/foils/Red.dat",outer_foil="data/foils/Red.dat",length=2.5*m,taper_ratio=1,diam=130*mm),
        WingElement(name="B",inner_foil="./data/foils/Red.dat",outer_foil="data/foils/Red.dat",length=5*m,taper_ratio=1/1,diam=130*mm),
        WingElement(name="C",inner_foil="./data/foils/Red.dat",outer_foil="data/foils/Red70.dat",length=4.4*m,taper_ratio=.8754/1,diam=130*mm),
        WingElement(name="D1",inner_foil="./data/foils/Red70.dat",outer_foil="data/foils/Red50.dat",length=2.504*m,taper_ratio=.7384/.8754,diam=130*mm),
        WingElement(name="D2",inner_foil="./data/foils/Red50.dat",outer_foil="data/foils/DAE41.dat",length=2.496*m,taper_ratio=.4692/.7384,diam=130*mm)
    ]
    wing=WingInfo(wing_elements=wing_elements,alpha_set=2*deg,root_chord=1.05*m)
    df=pd.read_csv("./data/csv/D.csv")
    for label in df.values:
        wing.save_foil((2.5+5+4.4)*u.m+label[0]*mm,0,f"{label[0]}","./out")
```