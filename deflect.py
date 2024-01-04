
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

left=Beam([
    BeamInfo.read(df["A"],spec),
    BeamInfo.read(df["B"],spec),
    BeamInfo.read(df["C"],spec,np.array([0,-3,0])*u.deg),
    BeamInfo.read(df["D"],spec)],320)


right=Beam([
    BeamInfo.read(df["A"],spec),
    BeamInfo.read(df["B"],spec),
    BeamInfo.read(df["C"],spec,np.array([0,3,0])*u.deg),
    BeamInfo.read(df["D"],spec)],320)

with open("data/load.npz","rb") as file:
    f=np.load(file)

plt.style.use('dark_background')
ax:Axes=plt.axes(projection='3d')

# right wing
r,U_xi=right.deflect(f*u.N,0*np.ones((320,3))*u.m,
                    np.array([[0,1,0], # e_xi
                              [1,0,0], # e_eta
                              [0,0,-1] # e_zeta
                              ]).T)

ax.plot(r[:,0],r[:,1],r[:,2],color='white',linewidth=3)
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,1,0],U_xi[:,1,1],U_xi[:,1,2],length=3,color='green',alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\eta_r$")
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,2,0],U_xi[:,2,1],U_xi[:,2,2],length=3,color='blue' ,alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\zeta_r$")

# left wing
r,U_xi=left.deflect(-f*u.N,0*np.ones((320,3))*u.m,
                    np.array([[0,-1,0], # e_xi
                              [-1,0,0], # e_eta
                              [0, 0,1] # e_zeta
                              ]).T)

ax.plot(r[:,0],r[:,1],r[:,2],color='white',linewidth=3)
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,1,0],U_xi[:,1,1],U_xi[:,1,2],length=3,color='orange',alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\eta_l$")
ax.quiver3D(r[:,0],r[:,1],r[:,2],U_xi[:,2,0],U_xi[:,2,1],U_xi[:,2,2],length=3,color='red' ,alpha=0.5,arrow_length_ratio=0,linewidths=0.5,normalize=True,label=r"$\zeta_l$")

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
plt.show()