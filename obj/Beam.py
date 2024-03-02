#!/usr/bin/python3
# -*- coding utf-8 -*-
"""Beamモジュール
桁計算を行うためのモジュール
"""

from astropy.units import Quantity
import numpy as np
import scipy.constants as consts
import pandas as pd
from utils import unit  as u
from utils.env import getEnvironment

class Sheet:
    """シートを表すクラス

    シートは材料の種類を表すクラスです。

    Attributes:
        name (str): 材料の名前
        E (Quantity): ヤング率(GPa)
        thickness (Quantity): 厚さ(mm)
        density (Quantity): 密度
        compound_rate (float): 積層率
        use (str): 使用方法
    """
    def __init__(self,name:str,E:Quantity,thickness:Quantity,density:Quantity,compound_rate:float,use:str,isotropy:bool=False):
        self.name = name
        self.E = E.decompose()
        self.E0 = E*Quantity(consts.g,"m/(s2)") * 0.85
        if isotropy:
            self.E90 = self.E0
            self.E45 = self.E0
        else:
            self.E90 = 0
            self.E45 = self.E0 * 0.2
        self.thickness = thickness.decompose()
        self.density:Quantity = density.decompose()
        self.compound_rate = compound_rate
        self.use = use

class PLY:
    """PLYクラス

    PLYクラスは積層を表すクラスです。

    Attributes:
        sheet (Sheet): プリプレグ
        angle (Quantity): 積層角度
        start (Quantity): 積層開始位置
        end (Quantity): 積層終了位置
        theta (Quantity): 積層角度
        sheet_number (int): シートの種類
    """
    def __init__(self,sheet:Sheet,angle:Quantity,start:Quantity,end:Quantity,theta:Quantity):
        self.sheet=sheet
        self.angle = angle.decompose()
        self.start = start.decompose()
        self.end = end.decompose()
        self.theta = theta.decompose()

class BeamInfo:
    """桁1本の情報を表すクラス

    Attributes:
        plys (list[PLY]): 積層のリスト
        diamiter (list[Quantity]): 外径
        length (Quantity): 長さ
        kanzashi (Quantity): かんざし長さ
    
    Methods:
        read: データフレームから読み込む
    """
    def __init__(self,plys:list[PLY],diamiter:list[Quantity],length:Quantity,kanzashi:Quantity,dihedral=np.zeros(3)*Quantity(0,"rad")):
        """ Initializes the instance from a list of PLYs.
        Args:
            plys:積層のリスト
            length:長さ
            kanzashi:かんざし長さ
        """
        self.plys=plys
        """積層のリスト"""
        self.diamiter=[diam.decompose() for diam in diamiter]
        """外径"""
        self.length=length.decompose()
        """長さ"""
        self.kanzashi=kanzashi.decompose()
        """かんざし長さ"""
        self.diheral=dihedral.decompose()
        """初期上反"""
    
    @staticmethod
    def read(dataframe:pd.DataFrame,spec:dict[int,Sheet],dihedral=Quantity(np.zeros(3),"rad")):
        """データフレームから読み込む

        Args:
            dataframe (pd.DataFrame): データフレーム
            spec (dict[int,Sheet]): Sheetの情報
            dihedral (Quantity): 初期上反
        """
        length=Quantity(dataframe.iloc[32,1],"m")
        diamiter=str(dataframe.iloc[32,0])
        if "~" in diamiter:
            diamiter=[Quantity(float(diam),"m") for diam in diamiter.split("~")]
        else:
            diamiter=[Quantity(float(diamiter),"m")]
        kanzashi=Quantity(dataframe.iloc[32,2],"m")
        angle=Quantity(dataframe['angle'].to_numpy()[:30],"deg")
        sh_num=dataframe['sheet_number'].to_numpy()[:30]
        ply_start=Quantity(dataframe['start[m]'].to_numpy()[:30],"m")
        ply_end=Quantity(dataframe['end[m]'].to_numpy()[:30],"m")
        theta=Quantity(dataframe['θ[deg]'].to_numpy()[:30],"deg")
        plys=[]
        for i in range(30):
            if np.isnan(angle[i]):
                break
            if np.isnan(ply_end[i]):
                ply_end[i]=length
            sheet=spec[sh_num[i]]
            plys.append(PLY(sheet,angle[i],ply_start[i],ply_end[i],theta[i]))
        return BeamInfo(plys,diamiter,length,kanzashi,dihedral)

class Beam:
    """桁全体を表すクラス
    連続的なデータとして記述されたBeamInfoを離散化

    Attributes:
        EI_xi (np.ndarray): ねじり剛性
        EI_eta (np.ndarray): 曲げ剛性(上下)
        EI_zeta (np.ndarray): 曲げ剛性(左右)
        EI (np.ndarray): 剛性行列
        G (np.ndarray): コンプライアンス行列
        weight (np.ndarray): 重量
        length (Quantity): 長さ
        span (Quantity): 全長
        s (np.ndarray): 分割点
        ds (Quantity): 微小区間の幅
        dihedral (np.ndarray): 初期上反
    
    Methods:
        read: 桁情報を読み込む
        deflect: 変形を計算する
    """
    def __init__(self,info:list[BeamInfo],N:int=100):
        """

        Args:
            info(BeamInfo):桁の情報
            N(int):分割数
        """
        self.N=N
        self.EI_xi=np.zeros(N)*Quantity(0,"m3 kg / s2") # ねじり剛性
        self.EI_eta=np.zeros(N)*Quantity(0,"m3 kg / s2") # 曲げ剛性(上下)
        self.EI_zeta=np.zeros(N)*Quantity(0,"m3 kg / s2") # 曲げ剛性(左右)
        self.EI=np.zeros((N,3,3))*Quantity(0,"m3 kg / s2")  # 剛性行列
        self.G=np.zeros((N,3,3))*Quantity(0,"s2/m3 kg") # コンプライアンス行列
        self.weight=np.zeros(N)*Quantity(0,"kg") # 重量
        self.length=sum([beam_info.length for beam_info in info]).decompose()
        self.s=np.linspace(0,self.length,N)
        self.dihedral=np.zeros((N,3))*Quantity(0,"rad") # 初期上反
        self.diamiter=Quantity(np.zeros(self.N),"m")
        self.read(info,N)
        
    def read(self,info:list[BeamInfo],N:int=100):

        """桁の情報を読み込む
        
        Args:
            info(BeamInfo):桁の情報
            N(int):分割数
        """

        down_rate = 0.96 # 0deg剛性ダウン率

        EI_xi=np.zeros(N)*Quantity(0,"m3 kg / s2") # ねじり剛性
        EI_eta=np.zeros(N)*Quantity(0,"m3 kg / s2") # 曲げ剛性(上下)
        EI_zeta=np.zeros(N)*Quantity(0,"m3 kg / s2") # 曲げ剛性(左右)
        weight=np.zeros(N)*Quantity(0,"kg") # 重量
        span=sum([beam_info.length for beam_info in info]).decompose()
        s=np.linspace(0,span,N)
        diamiter=np.zeros(N)*Quantity(0,"m") # 外径
        wide=np.ones(N)/N*span # 微小区間の幅
        beam_start=0*Quantity(0,"m")
        dihedral=np.zeros((N,3))*Quantity(0,"rad") # 初期上反
        for beam_info in info:
            idx=np.where((beam_start<=s)&(s<=beam_start+beam_info.length))
            if len(beam_info.diamiter)==1:
                diamiter[idx]=beam_info.diamiter[0]
            else:
                diamiter[idx]=np.interp(s[idx],np.linspace(beam_start,beam_start+beam_info.length,100),np.linspace(beam_info.diamiter[0],beam_info.diamiter[1],100))
            dihedral[np.where(abs(s-beam_start)<wide)[0][0]]=beam_info.diheral # 上反角の設定
            for ply in beam_info.plys:
                idx=np.where((beam_start+ply.start<=s)&(s<=beam_start+ply.end))
                diamiter[idx]+=ply.sheet.thickness*2
                if np.isnan(ply.end):
                    ply.end=beam_info.length
                if np.isnan(ply.theta):
                    ply.theta=Quantity(np.pi,"rad")
                insidx=np.where((beam_start-beam_info.kanzashi<s)&(s<=beam_start))
                if len(insidx[0])!=0:
                    weight[insidx]+=ply.sheet.density*diamiter[insidx]*wide[insidx]*np.pi
                match ply.angle.to("deg").value:
                    case 0:
                        EI_eta[idx]+=1/8*(ply.theta.value+np.sin(ply.theta))*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E0 * down_rate
                        EI_zeta[idx]+=1/8*(ply.theta.value+np.sin(ply.theta))*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E0 * down_rate
                        EI_xi[idx]+=2*ply.theta.value*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E45 * down_rate
                        weight[idx]+=ply.sheet.density*diamiter[idx]*wide[idx]*ply.theta.value
                    case 45:
                        EI_eta[idx]+=1/8*(ply.theta.value+np.sin(ply.theta))*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E45 * down_rate
                        EI_zeta[idx]+=1/8*(ply.theta.value+np.sin(ply.theta))*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E45 * down_rate
                        EI_xi[idx]+=ply.theta.value*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E0 * down_rate # 2枚あって初めて両方向のねじれ剛性が出る
                        weight[idx]+=ply.sheet.density*diamiter[idx]*wide[idx]*np.pi
                    case 90:
                        weight[idx]+=ply.sheet.density*diamiter[idx]*wide[idx]*np.pi
                    case 99:
                        EI_eta[idx]+=1/8*(ply.theta.value+np.sin(ply.theta))*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E0
                        EI_zeta[idx]+=1/8*(ply.theta.value+np.sin(ply.theta))*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E0
                        EI_xi[idx]+=2*ply.theta.value*diamiter[idx]**3*ply.sheet.thickness*ply.sheet.compound_rate*ply.sheet.E45
                        weight[idx]+=ply.sheet.density*diamiter[idx]*wide[idx]*np.pi
                    case _:
                        print(f"error: unknown format ({ply.angle})")
                        pass
            beam_start+=beam_info.length
        self.span=span
        self.EI_xi=EI_xi
        self.EI_eta=EI_eta
        self.EI_zeta=EI_zeta
        self.EI[:,0,0]=EI_xi
        self.EI[:,1,1]=EI_eta
        self.EI[:,2,2]=EI_zeta
        self.G[:,0,0]=1/EI_xi
        self.G[:,1,1]=1/EI_eta
        self.G[:,2,2]=1/EI_zeta
        self.weight=weight
        self.s=s
        self.ds=span/N
        self.dihedral=dihedral
        self.diamiter=diamiter