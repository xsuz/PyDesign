#!/usr/bin/python3
# -*- coding utf-8 -*-
"""Wingモジュール
翼型の情報を格納するクラスを定義するモジュール
"""
import xfoil
import numpy as np
from astropy import units as u

class WingElement:
    def __init__(self,name:str,inner_foil:str,outer_foil:str,length:u.Quantity,diam:u.Quantity,taper_ratio:float=1,alpha_twist:u.Quantity=u.Quantity(0,"deg"),twist_length:u.Quantity=u.Quantity(0,"m")):
        """
        Args:
            name (str): 区間の名前
            inner_foil (str): 両端の翼型
            outer_foil (str): 両端の翼型
            length (Quantity): 翼要素の長さ
            diam (Quantity): 桁半径
            taper_ratio (float, optional): テーパー比. Defaults to 1.
            alpha_twist (Quantity, optional): ねじり下げ角. Defaults to u.Quantity(0,"deg").
            twist_length (Quantity, optional): ねじり下げ区間の長さ. Defaults to u.Quantity(0,"m").
            
        """
        self.name=name
        """区間の名前"""
        self.inner_foil=inner_foil;self.outer_foil=outer_foil
        """両端の翼型"""
        self.length=length
        """翼要素の長さ"""
        self.diam=diam
        """桁半径"""
        self.taper_ratio=taper_ratio
        """テーパー比"""
        self.alpha_twist=alpha_twist
        """ねじり下げ角"""
        self.twist_length=twist_length
        """ねじり下げ区間の長さ"""
        self.xf=xfoil.XFoilTask()
        self.xf.load(inner_foil,True)
        self.xf.load(outer_foil,False)
    def C_L(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから揚力係数を計算

        Args:
            alpha (Quantity): 迎角(deg)
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置(m)

        Returns:
            float: 揚力係数
        """
        self.xf.interpolate(1-x.to("m")/self.length.to("m"))
        result=self.xf.calc(alpha.to_value("deg"),Re)
        return result["cl"]
    def C_D(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから抗力係数を計算

        Args:
            alpha (Quantity): 迎角(deg)
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置(m)

        Returns:
        
        """
        self.xf.interpolate(1-x.to("m")/self.length.to("m"))
        result=self.xf.calc(alpha.to_value("deg"),Re)
        return result["cd"]
    def C_M(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから縦揺れモーメント係数を計算

        Args:
            alpha (Quantity): 迎角(deg)
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置(m)
        
        Returns:
            float: 縦揺れモーメント係数
        """
        self.xf.interpolate(1-x.to("m")/self.length.to("m"))
        result=self.xf.calc(alpha.to_value("deg"),Re)
        return result["cm"]
    def X_Cp(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから圧力中心を計算

        Args:
            alpha (Quantity): 迎角(deg)
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置(m)
        
        Returns:
            float: 圧力中心
        """
        self.xf.interpolate(1-x.to("m")/self.length.to("m"))
        result=self.xf.calc(alpha.to_value("deg"),Re)
        return result["xcp"]
    def calc(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->dict:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから揚力係数,抗力係数,縦揺れモーメント係数,圧力中心を計算

        Args:
            alpha (Quantity): 迎角(deg)
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置(m)
        
        Returns:
            dict of float: clは揚力係数,cdは抗力係数,cmは縦揺れモーメント係数,xcpは圧力中心
        """
        self.xf.interpolate(1-x.to("m")/self.length.to("m"))
        result=self.xf.calc(alpha.to_value("deg"),Re)
        return result
    def CpV(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->tuple[np.ndarray,np.ndarray]:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから圧力係数分布を計算

        Args:
            alpha (Quantity): 迎角(deg)
            Re (float): レイノルズ数
            x (float): 翼根からの位置(m)
        
        Returns:
            tuple of np.array,np.array: 圧力分布
        """
        self.xf.interpolate(1-x.to("m")/self.length.to("m"))
        _x,_cpv=self.xf.cpv(alpha.to_value("deg"),Re)
        return np.array(_x),np.array(_cpv)
    def save(self,x:u.Quantity,tegap_rate:float,filename:str,foilname:str):
        """翼根側からx(m)の翼型foilnameをfilenameに保存する

        Args:
            x (float): 翼根からの位置(m)
            tegap_rate (float): 80%の位置にかけるtegap(%)
            filename (str): ファイル名
            foilname (str): 翼型名
        
        Returns:
            None
        """
        self.xf.interpolate(1-x.to("m")/self.length.to("m"))
        self.xf.tegap(tegap_rate/100,0.8)
        self.xf.save(foilname,filename)

class WingInfo:
    """
    静的な翼の情報を格納

    Attributes
    ----------    
    wing_elements : list[WingElement]
        翼要素のリスト（翼根から翼端へ）
    delimiter : list[float]
        翼要素の区切り
    xf : xf
        XFoilのインスタンス
    default_phi : float
        上反角(deg)
    alpha_set : float
        取り付け角(deg)
    """

    def __init__(self,wing_elements:list[WingElement],alpha_set:u.Quantity,root_chord:u.Quantity):
        """
        Args:
            wing_elements (list[WingElement]): 翼要素のリスト（翼根から翼端へ）
            default_phi (Quantity): 取り付け時の上反角
            alpha_set (Quantity): 取り付け角
            root_chord (Quantity): 翼根の翼弦長
        """
        self.wing_elements=wing_elements
        """翼要素のリスト（翼根から翼端へ）"""
        self._delimiter=[u.Quantity(0,"m")]
        """翼要素の区切り"""
        self.alpha_set=alpha_set.to("rad")
        """取り付け角"""
        self.root_chord=root_chord
        x=u.Quantity(0,"m")
        for we in self.wing_elements:
            x+=we.length
            self._delimiter.append(x.to("m"))
    def C_L(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α、レイノルズ数Re、翼根からの位置xから揚力係数を計算

        Args:
            alpha (Quantity): 迎角
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置
        
        Returns:
            float: 揚力係数
        """
        i,x=self.__offset(x)
        return self.wing_elements[i].C_L(alpha,Re,x)
    def C_D(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから抗力係数を計算

        Args:
            alpha (Quantity): 迎角
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置

        Returns:
            float: 抗力係数
        """
        i,x=self.__offset(x)
        return self.wing_elements[i].C_D(alpha,Re,x)["cd"]
    def C_M(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α、レイノルズ数Re、翼根からの位置xから縦揺れモーメント係数を計算

        Args:
            alpha (Quantity): 迎角
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置

        Returns:
            float: 縦揺れモーメント係数
        """
        i,x=self.__offset(x)
        return self.wing_elements[i].C_M(alpha,Re,x)["cm"]
    def X_Cp(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->float:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから圧力中心を計算

        Args:
            alpha (Quantity): 迎角
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置
        
        Returns:
            float: 圧力中心
        """
        i,x=self.__offset(x)
        return self.wing_elements[i].X_Cp(alpha,Re,x)["xcp"]
    def calc(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->dict:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから揚力係数,抗力係数,縦揺れモーメント係数,圧力中心を計算

        Args:
            alpha (Quantity): 迎角
            Re (Quantity): レイノルズ数
            x (Quantity): 翼根からの位置

        Returns:
            dict of float: clは揚力係数,cdは抗力係数,cmは縦揺れモーメント係数,xcpは圧力中心
        """
        i,x=self.__offset(x)
        return self.wing_elements[i].calc(alpha,Re,x)
    def CpV(self,alpha:u.Quantity,Re:u.Quantity,x:u.Quantity)->tuple[np.ndarray,np.ndarray]:
        """
        迎角α(deg)、レイノルズ数Re、翼根からの位置xから圧力係数分布を計算

        Args:
            alpha (Quantity): 迎角(deg)
            Re (float): レイノルズ数
            x (float): 翼根からの位置(m)

        Returns:
            tuple of np.array,np.array: 圧力分布
        """
        _i,_x=self.__offset(x)
        return self.wing_elements[_i].CpV(alpha,Re,_x)
    def diam(self,x:u.Quantity)->u.Quantity:
        """
        翼根から位置xの桁の太さを返す
        
        Args:
            x (Quantity): 翼根からの位置(m)

        Returns:
            Quantity: 桁の太さ
        """
        root_diam=1*self.wing_elements[0].diam # 翼根側の翼型混合率（最大値1）
        for d,w in zip(self._delimiter,self.wing_elements):
            if x.to("m")>d.to("m")+w.length.to("m"):
                root_diam=1*w.diam
            else:
                return root_diam*(d+w.length-x)/w.length+w.diam*(x-d)/w.length
        return root_diam
    def save_foil(self,x:u.Quantity,tegap_rate:float,foilname:str,out_dir=""):
        """
        翼根からxの位置の翼型を80%の位置にtegapをかけて保存

        Args:
            x (Quantity): 翼根からの位置(m)
            tegap_rate (float): 80%の位置にかけるtegap(%)
            foilname (str): 翼型名
            out_dir (str, optional): 出力先ディレクトリ. Defaults to "".

        Returns:
            None

        Exceptions:
            IOError: 出力先ディレクトリが存在しない
        """
        _i,_x=self.__offset(x)
        self.wing_elements[_i].save(_x,tegap_rate,f"{out_dir}/{foilname}.dat",foilname)
        return
    def __offset(self,x:u.Quantity):
        for i,d in enumerate(self._delimiter[:-1]):
            if d.to("m")<=x.to("m") and x.to("m")< d.to("m")+self.wing_elements[i].length.to("m"):
                return (i,x-d)
        return len(self._delimiter)-2,x.to("m")-self._delimiter[len(self._delimiter)-1].to("m")
    def chord(self,x:u.Quantity)->u.Quantity:
        """
        翼根から位置xの翼弦長を返す
        
        Args:
            x (Quantity): 翼根からの位置(m)

        Returns:
            Quantity: 翼弦長
        """
        # 参考：https://yuukivel.hatenadiary.org/entry/20130913/1379072156
        root_chord=self.root_chord.to("m") # 翼根の翼弦長（最大値1）
        for d,w in zip(self._delimiter,self.wing_elements):
            if x.to("m")>d.to("m")+w.length.to("m"):
                root_chord*=w.taper_ratio
            else:
                # 多段テーパーとして線形近似する
                return root_chord*((d.to("m")+w.length.to("m")-x.to("m"))/w.length.to("m")+w.taper_ratio*(x.to("m")-d.to("m"))/w.length.to("m"))
        return root_chord