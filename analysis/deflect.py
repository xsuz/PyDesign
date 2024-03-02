from obj.Beam import Beam
from astropy.units import Quantity
import numpy as np
import scipy.constants as consts
from utils import unit  as u
from utils.env import getEnvironment


def deflect(beam:Beam,f_local:np.ndarray[Quantity],r_local:np.ndarray[Quantity],U_base=np.eye(3),epochs=10,ALPHA=0.8,eps=1e-3)->dict[str,np.ndarray[np.float64]]:
    """桁の変形を計算する
    Args:
        beam (Beam): 桁
        f_local (np.ndarray[Quantity]): 局所座標系ξηζ系で与えられた力
        r_local (np.ndarray[Quantity]): 局所座標系ξηζ系で与えられた座標
        U_base (np.ndarray): 固定端の座標系
        epochs (int): 繰り返し計算の回数
        ALPHA (float): 更新率
        eps (float): 収束条件
    
    Returns:
        r (np.ndarray[Quantity]): 変形後の座標
        U_xi (np.ndarray): 変形後の座標系
        euler (np.ndarray[Quantity]): 変形後のEuler角
        M (np.ndarray[Quantity]): 変形後の曲げモーメント
    """
    if getEnvironment()=='Interpreter':
        from tqdm import tqdm
    else:
        from tqdm.notebook import tqdm
    
    # 回転行列

    Rx=lambda phi:np.array(
        [[1         ,0      ,0],
         [0         ,np.cos(phi) ,np.sin(phi)],
         [0         ,-np.sin(phi) ,np.cos(phi)]]
    )
    Ry=lambda theta:np.array(
        [[np.cos(theta)    ,0      ,-np.sin(theta)],
         [0                ,1      ,0],
         [np.sin(theta)    ,0      ,np.cos(theta)]]
    )
    Rz=lambda psi:np.array(
        [[np.cos(psi)    ,np.sin(psi) ,0],
         [-np.sin(psi)   ,np.cos(psi) ,0],
         [0         ,0      ,1]]
    )

    U_xi=np.zeros((beam.N,3,3))
    # 初期上反を考慮した回転行列の設定
    euler=np.zeros((beam.N,3))*u.rad
    for i in range(beam.N):
        euler[i:]+=beam.dihedral[i].to("rad")
        U_xi[i]=(Rx(euler[i,0])@Ry(euler[i,1])@Rz(euler[i,2])).T
    r=np.cumsum(U_xi[:,:,0]*beam.ds,axis=0)
    
    F_sum=sum(f_local)
    F_local=np.array([np.interp(np.arange(beam.N)*len(f_local_i)/beam.N,np.arange(len(f_local_i)),f_local_i) for f_local_i in f_local.T]).T
    F_local=f_local*F_sum/(sum(f_local)+1e-20*u.N)
    r_local=np.array([np.interp(np.arange(beam.N),np.arange(len(r_local_i)),r_local_i) for r_local_i in r_local.T]).T*u.m
    M=np.zeros((beam.N,3))*u.N*u.m
    # 繰り返し計算
    with tqdm(range(epochs)) as pbar:
        for _ in pbar:
            M=np.zeros((beam.N,3))*u.N*u.m
            f=np.zeros((beam.N,3))*u.N
            euler=np.zeros((beam.N,3))*u.rad
            for i in range(beam.N):
                f[i]=U_xi[i]@F_local[i] # 局所座標系ξηζ系で与えられた力をxyz系に変換
                f[i]+=U_base[2]*beam.weight[i]*consts.g *u.m/u.s**2 # 桁の重量を考慮
            f.decompose()
            for i in range(beam.N):
                kappa=np.zeros((beam.N,3))/u.m
                for j in range(i,beam.N):
                    T=np.cross(r[j]+U_xi[j]@r_local[j]-r[i],f[j]) # 曲げモーメント
                    # 力のつり合いの方程式は
                    # \mathbf{EI}\mathbf{\kappa}=\mathbf{M}
                    kappa+=beam.G[i]@(T@U_xi[i])
                    M[i]+=T@U_xi[i]
                # 回転ベクトル→Euler角の変換
                deuler=np.array([
                    [1,     np.sin(euler[i,0])*np.tan(euler[i,1]),   np.cos(euler[i,0])*np.tan(euler[i,1])],
                    [0,     np.cos(euler[i,0]),                    -np.sin(euler[i,0])],
                    [0,     np.sin(euler[i,0])/np.cos(euler[i,1]),   np.cos(euler[i,0])/np.cos(euler[i,1])]
                ])@kappa[i]*beam.ds*u.rad
                euler[i:]+=deuler+beam.dihedral[i]
                # zyxEuler角で基底ベクトルを構成
                U_xi[i+1:]=(Rx(euler[i,0])@Ry(euler[i,1])@Rz(euler[i,2])).T
            tip_error=np.linalg.norm(r[-1]-np.cumsum(U_xi[:,:,0]*beam.ds,axis=0)[-1])
            pbar.set_postfix({"error":tip_error})
            # 収束条件を満たしたら終了
            if tip_error<eps*u.m:
                break
            # 更新率の応じて変化量を設定
            r=r*(1-ALPHA)+ALPHA*np.cumsum(U_xi[:,:,0]*beam.ds,axis=0)
    # 固定端の座標系を基準に戻す
    for i in range(beam.N):
        r[i]=U_base@r[i]
        U_xi[i]=U_base@U_xi[i]
    return {"r":r,"U_xi":U_xi,"euler":euler,"M":M}