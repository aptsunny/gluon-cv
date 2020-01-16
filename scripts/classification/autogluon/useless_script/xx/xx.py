# -*- coding: utf-8 -*-
import numpy
from matplotlib import pyplot as pl
from matplotlib.font_manager import FontProperties
#font1 = FontProperties(fname=r"c:\windows\fonts\simsun.ttc",size=14) #可指定计算机内的任意字体，size为字体大小

class fitting:
    def __init__(self,X,Y):
        self.x=numpy.array(X)
        self.y=numpy.array(Y)
    def fitting(self,n):
        self.z=numpy.polyfit(self.x,self.y,n)
        self.p=numpy.poly1d(self.z)
        self.error=numpy.abs(self.y-numpy.polyval(self.p,self.x))
        self.ER2=numpy.sum(numpy.power(self.error,2))/len(self.x)
        return self.z,self.p
    def geterror(self):
        return self.error,self.ER2
    def show(self):
        figure1=pl.figure()
        pl.plot(self.x,self.y,'ro-',markersize=7,figure=figure1,label='origin data')
        pl.plot(self.x,numpy.polyval(self.p,self.x),markersize=7,figure=figure1,label='fitting data')
        pl.title(u'延滞系数a,n的标定')
        # pl.title(u'延滞系数a,n的标定',fontproperties=font1)

        #plt.title(u'流量-密度散点图',fontproperties=font1)
        pl.xlabel(u'X=ln(V)')
        # pl.xlabel(u'X=ln(V)',fontproperties=font1)
        pl.ylabel(u'Y=ln[t(V)-t0]')
        # pl.ylabel(u'Y=ln[t(V)-t0]',fontproperties=font1)
        #plt.ylabel(u'流量',fontproperties=font1)
        pl.grid()
        pl.legend()
        pl.show()
    def predict(self,x):
        return numpy.polyval(self.p,x)
'''    
X=[ 20	,40	,60	,80	,100	,120	,140, 160,180]  #仿真采集
Y=[ 0.3,	0.61,	1.06,	1.11,	1.16	,1.21	,1.22 ,1.21,1.18]#  
'''

#jiangsu#去除误差  1.94 ,	0.00 , -1.73,	2.58 ,
X=[3.40, 	4.04 ,	4.21, 	4.44 ,	4.53 ,	4.48 ,	4.61, 	4.87, 	5.17, 	5.34, 	5.40 ,	5.32, 	5.21 ,
       5.16 ,	5.16 ,	5.04 ,	3.69,  	-3.65, 	2.74 ,	4.30, 	4.89 ,	5.14 ,	5.25, 	5.43, 	5.22 ,	5.18,
       5.23, 	5.19, 	5.07, 	5.08 ,	4.95 ,	4.70 ,	4.49, 	4.26, 	3.80, 	2.03, 	-3.17 ]
Y=[ 	2.12, 	2.75, 	3.14, 	3.41, 	3.63 ,	3.81 ,	3.96, 	4.09, 	4.20, 	4.49, 	4.59, 	4.46 ,	4.31 ,
       4.13 ,	3.91, 	3.63, 	3.24, 0.60, 	1.90, 	2.75, 	3.14, 	3.41, 	3.63 ,	3.75 ,	4.02, 	4.06 ,
       4.20 ,	4.25 ,	4.39, 	4.36, 	4.31, 	4.13, 	3.91, 	3.63, 	3.24, 	2.58, 	0.00]
'''
#siping#去除误差  -3.46,0.00 ,
Y=[	0.69, 	1.10, 	2.64, 	2.77, 	3.00, 	3.00, 	2.94 ,	2.94 ,	3.00, 	2.89, 	2.94, 	2.89, 	2.94, 	
       2.89 ,	2.30, 	1.95, 	1.39, 	0.69, 	0.00, 	0.69, 	1.95, 	2.64, 	2.77 ,	3.00 ,	3.00, 	2.94, 	2.94,
       3.00 ,	2.89, 	2.94, 	2.89 ,	2.94 ,	2.89, 	2.30, 	1.39, 	0.69 ]
X=[ 	1.97 ,	3.03 ,	3.87 ,	4.05, 	4.09, 	4.16 ,	4.09 ,	4.18 ,	4.16 ,	4.02 ,	3.95, 	3.99, 	3.73 ,
       3.47, 	3.32 ,	3.23, 	2.66, 	0.80 ,	0.98 	,2.27 ,	2.94, 	3.23, 	3.39 ,	3.53 	,3.56 ,	3.71 ,	3.96, 	
       4.12 ,	4.30 ,	4.40, 	4.27, 	4.16, 	4.22, 	4.09, 	3.91, 	2.06 ]


#2-11 去除 y,x【2.686,5.280】  [1.992 ,4.437],[0.693,2.786 ]  [3.091 , 4.944 ]    ,4.449, 6.229      ,4.295,6.209   ,3.196,4.028
Y=[ 3.379 ,3.602 ,3.784 ,3.938 ,4.072 ,4.190 ,4.477 ,4.583   ,4.113 ,3.890 ,3.602  ]
X=[4.558 ,4.532 ,4.644 ,4.726 ,4.817 ,4.903 ,5.190 ,5.628  ,5.646 ,5.487 ,5.332  ]

#11-2            ,2.603,3.903       4.868 ,   6.834
Y=[3.091 ,4.500 ,4.700 ,4.431 ,4.205 ,4.078 ,4.159 ,4.069 ,3.738 ,3.912 ,4.190 ,4.324 ,4.174 ,3.664 ,3.418  ,1.609] 
X=[3.942 ,5.650  ,6.352 ,5.511 ,5.083 ,4.992 ,4.997 ,4.894 ,4.750 ,4.813 ,5.058 ,5.309 ,5.028 ,4.422 ,4.088  ,2.181] 

'''

F=fitting(X,Y)
z,p=F.fitting(1)
e,E=F.geterror()
print ('系数：',z)
print ('拟合函数：',p)
print ('最小平方误差：',E)
print ('F(140)的预测值',F.predict(4.397229237))  #3.80666249 对应Y2.62754469743
#  5.247024072  对应3.99023896973
F.show()