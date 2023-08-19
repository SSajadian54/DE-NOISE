import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import emcee
from matplotlib import rc
rc('font', **{'family':'serif','serif':['Helvetica']})
rc('text', usetex=True)
plt.rcParams.update({'font.size': 14})
import numpy as np

import pandas as pd
import pywt
from skimage.restoration import denoise_wavelet
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio
def PTRm(num, year):  
    if(num==213 and year==2015):    ptr=np.array([0.293 , 7083.431 , 5.022  , 19.921 ,  1.0 ]);  return(ptr)
    elif(num==225 and year==2015):  ptr=np.array([0.157 , 7087.233 , 20.918  , 18.618 , 0.118 ]);  return(ptr)
    elif(num==246 and year==2015):  ptr=np.array([0.634 , 7140.923 , 34.939  , 15.942 , 1.000 ]);  return(ptr)    
    elif(num==255 and year==2015):  ptr=np.array([0.414 , 7111.795 , 20.037 , 15.937 ,  0.694 ]);  return(ptr)
    elif(num==274 and year==2015):  ptr=np.array([0.920 , 7104.752 , 12.690  , 16.877 , 1.0 ]);  return(ptr)
    elif(num==298 and year==2015):  ptr=np.array([0.729 , 7161.847 , 25.584  , 17.958 , 1.0 ]);  return(ptr)  
    elif(num==410 and year==2015):  ptr=np.array([0.267 , 7120.178 , 20.182 , 16.145 ,  0.918 ]);  return(ptr)  
    elif(num==432 and year==2015):  ptr=np.array([0.081 , 7101.595 , 13.702  , 18.812 , 1.0 ]);  return(ptr)    
    elif(num==610 and year==2015):  ptr=np.array([0.061 , 7136.079 , 36.302  , 18.833 , 0.686 ]);  return(ptr)    
    elif(num==23  and year==2017):  ptr=np.array([0.219 , 7815.727 , 30.440  , 15.947 , 1.000 ]);  return(ptr)    
    elif(num==415 and year==2017):  ptr=np.array([0.725 , 7861.974 , 13.594  , 17.084 , 1.000 ]);  return(ptr)  
    elif(num==430 and year==2017):  ptr=np.array([0.339 , 7905.223 , 300.713  , 19.151, 0.318 ]);  return(ptr)
    elif(num==436 and year==2017):  ptr=np.array([1.099 , 7843.443 , 5.980  , 17.346 ,  1.000 ]);  return(ptr)     
    elif(num==463 and year==2017):  ptr=np.array([1.719 , 7852.494 , 7.766  , 16.311 ,  1.000 ]);  return(ptr)
    elif(num==540 and year==2017):  ptr=np.array([0.019 , 7964.470 ,267.549 , 20.138 ,  0.472 ]);  return(ptr)        
    elif(num==585 and year==2017):  ptr=np.array([0.869 , 7891.031 , 28.700  , 18.729 , 1.000 ]);  return(ptr)    
    elif(num==43 and year==2018):  ptr=np.array([1.212 , 8175.805 , 9.314  , 15.172 , 1.000 ]);  return(ptr)
    elif(num==54 and year==2018):  ptr=np.array([0.138 , 8204.077 , 59.856  , 18.195 , 0.657 ]);  return(ptr)
    elif(num==57 and year==2018):  ptr=np.array([1.392 , 8179.790 , 14.234  , 15.886 , 1.000 ]);  return(ptr)
    elif(num==83 and year==2018):  ptr=np.array([0.852 , 8136.139 , 32.847  , 15.907 , 1.000 ]);  return(ptr)
    elif(num==250 and year==2018):  ptr=np.array([0.206 , 8202.269 , 41.888  , 18.972 , 1.000 ]);  return(ptr)
    elif(num==318 and year==2018):  ptr=np.array([0.112 , 8188.788 , 4.725  , 18.629 , 1.000 ]);  return(ptr)
    elif(num==324 and year==2018):  ptr=np.array([0.068 , 8192.189 , 11.164  , 17.121 , 0.100 ]);  return(ptr)
    elif(num==326 and year==2018):  ptr=np.array([0.063 , 8182.982 , 9.033  , 19.840 , 1.000 ]);  return(ptr)
    elif(num==566 and year==2018):  ptr=np.array([0.929 , 8281.080 , 35.023  , 16.665 , 1.000 ]);  return(ptr)
    elif(num==613 and year==2018):  ptr=np.array([0.683 , 8288.575 , 45.370  , 16.240 , 0.656 ]);  return(ptr)
    elif(num==1040 and year==2018):  ptr=np.array([0.046 , 8282.033 , 11.738  , 19.038 , 0.339 ]);  return(ptr)
    elif(num==1041 and year==2018):  ptr=np.array([0.732 , 8284.400 , 10.695  , 18.753 , 1.000 ]);  return(ptr)
    elif(num==1050 and year==2018):  ptr=np.array([0.852 , 8290.500 , 22.592  , 18.314 , 1.000 ]);  return(ptr)
    elif(num==1061 and year==2018):  ptr=np.array([0.612 , 8286.024 , 11.670  , 19.422 , 1.000 ]);  return(ptr)
    elif(num==1064 and year==2018):  ptr=np.array([0.468 , 8311.987 , 33.670  , 17.034 , 0.337 ]);  return(ptr)
    elif(num==1074 and year==2018):  ptr=np.array([0.023 , 8314.821 , 58.069  , 17.976 , 0.926 ]);  return(ptr)
    elif(num==1107 and year==2018):  ptr=np.array([0.034 , 8320.129 , 65.985  , 18.267 , 0.135 ]);  return(ptr)
    elif(num==1119 and year==2018):  ptr=np.array([0.662 , 8314.122 , 30.187  , 18.884 , 1.000 ]);  return(ptr)
    elif(num==1126 and year==2018):  ptr=np.array([0.008 , 8298.285 , 38.562  , 18.908 , 0.096 ]);  return(ptr)
    elif(num==1155 and year==2018):  ptr=np.array([1.403 , 8295.220 , 6.696  , 17.726 , 1.000 ]);  return(ptr)
    elif(num==1181 and year==2018):  ptr=np.array([0.115 , 8331.763 , 21.965  , 17.250 , 1.000 ]);  return(ptr)
    elif(num==1198 and year==2018):  ptr=np.array([0.018 , 8316.718 , 37.221  , 19.752 , 0.564 ]);  return(ptr)
    elif(num==1199 and year==2018):  ptr=np.array([0.207 , 8312.389 , 4.508   , 15.428 , 1.000 ]);  return(ptr)
    elif(num==1214 and year==2018):  ptr=np.array([1.253 , 8307.729 , 14.424  , 17.827 , 1.000 ]);  return(ptr)
    elif(num==1235 and year==2018):  ptr=np.array([1.081 , 8308.387 , 12.140  , 18.065 , 1.000 ]);  return(ptr)
    elif(num==1245 and year==2018):  ptr=np.array([0.568 , 8322.501 , 20.543  , 18.283 , 1.000 ]);  return(ptr)
    elif(num==1254 and year==2018):  ptr=np.array([0.045 , 8339.332 , 39.121  , 18.019 , 0.591 ]);  return(ptr)
    elif(num==1261 and year==2018):  ptr=np.array([0.649 , 8344.419 , 60.927  , 18.517 , 1.000 ]);  return(ptr)
    elif(num==1296 and year==2018):  ptr=np.array([0.103 , 8316.540 , 4.618  ,  18.729 , 0.278 ]);  return(ptr)
    elif(num==1302 and year==2018):  ptr=np.array([1.226 , 8347.689 , 38.784  , 18.031 , 1.000 ]);  return(ptr)
    elif(num==1306 and year==2018):  ptr=np.array([1.704 , 8315.051 , 20.746  , 17.339 , 1.000 ]);  return(ptr)
    elif(num==1367 and year==2018):  ptr=np.array([0.031 , 8358.129 , 20.111  , 18.491 , 0.764 ]);  return(ptr)
    elif(num==1385 and year==2018):  ptr=np.array([0.227 , 8334.704 , 19.221  , 18.098 , 0.265 ]);  return(ptr)
    elif(num==1393 and year==2018):  ptr=np.array([0.0001 ,8342.997 , 19.504  , 19.222 , 0.256 ]);  return(ptr)
    elif(num==1397 and year==2018):  ptr=np.array([0.145 , 8338.126 , 18.845  , 18.792 , 1.000 ]);  return(ptr)
    elif(num==1470 and year==2018):  ptr=np.array([0.185 , 8345.498 , 15.822  , 19.101 , 0.549 ]);  return(ptr)
    elif(num==1542 and year==2018):  ptr=np.array([0.055 , 8349.204 , 39.527  , 18.232 , 0.045 ]);  return(ptr)
    elif(num==1545 and year==2018):  ptr=np.array([0.658 , 8341.712 , 12.123  , 18.810 , 1.000 ]);  return(ptr)
    elif(num==1784 and year==2018):  ptr=np.array([0.0001 ,8419.342 , 61.604  , 17.645 , 0.048 ]);  return(ptr)
    elif(num==1115 and year==2014):  ptr=np.array([0.154 , 6830.220 , 48.608  , 19.994 , 0.545 ]);  return(ptr)
    elif(num==1284 and year==2014):  ptr=np.array([0.022 , 6852.987 , 65.674  , 19.266 , 0.195 ]);  return(ptr)
    elif(num==1407 and year==2014):  ptr=np.array([0.063 , 6862.562 , 42.663  , 18.209 , 0.630 ]);  return(ptr)
    elif(num==1574 and year==2014):  ptr=np.array([0.079 , 6870.930 , 27.468  , 19.307 , 0.099 ]);  return(ptr)
    elif(num==1566 and year==2014):  ptr=np.array([0.004 , 6888.442 , 286.595  ,18.741 , 0.035 ]);  return(ptr)
    elif(num==1286 and year==2014):  ptr=np.array([0.137 , 6861.969 , 78.785  , 18.962 , 0.207 ]);  return(ptr)
    elif(num==410  and year==2018):  ptr=np.array([0.109 , 8291.744 , 53.994  , 16.567 , 1.000 ]);  return(ptr)
    elif(num==1637 and year==2015):  ptr=np.array([0.038 , 7221.868 , 38.934  , 20.319 , 0.167 ]);  return(ptr)
    elif(num==1582 and year==2015):  ptr=np.array([0.039 , 7211.715 , 80.199  , 19.680 , 0.120 ]);  return(ptr)
    elif(num==1132 and year==2013):  ptr=np.array([0.172 , 6490.147 , 42.215  , 18.341 , 0.109 ]);  return(ptr)
    elif(num== 922 and year==2013):  ptr=np.array([0.039 , 6452.741 , 62.430  , 19.687 , 0.139 ]);  return(ptr)
    elif(num== 909 and year==2013):  ptr=np.array([0.617 , 6495.703 , 45.546  , 18.363 , 1.000 ]);  return(ptr)
    elif(num==1459 and year==2012):  ptr=np.array([0.015 , 6069.934 , 101.929 , 20.181 , 0.138 ]);  return(ptr)
    elif(num== 805 and year==2013):  ptr=np.array([0.062 , 6465.443 ,  66.063 , 19.111 , 0.317 ]);  return(ptr)    
    elif(num== 518 and year==2017):  ptr=np.array([1.394 , 7860.637 ,  18.664 , 17.530 , 1.000 ]);  return(ptr)        
    elif(num== 433 and year==2017):  ptr=np.array([0.070 , 7844.099 ,   8.944 , 19.347 , 1.000 ]);  return(ptr)        
    elif(num== 317 and year==2017):  ptr=np.array([0.109 , 7824.432 ,   3.231 , 19.810 , 1.000 ]);  return(ptr)        
################################################################################
################################################################################
wwtt=('cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8','morl', 'cmor', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'shan')
################################################################################    
def get_fft_values(y_values, T, N, f_s):
    f_values = np.linspace(0.0, 1.0/(2.0*T), N//2)
    fft_values_ = np.fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values
    
################################################################################    
def get_ave_values(xvalues, yvalues, n = 5):
    signal_length = len(xvalues)
    if signal_length % n == 0:
        padding_length = 0
    else:
        padding_length = n - signal_length//n % n
    xarr = np.array(xvalues)
    yarr = np.array(yvalues)
    xarr.resize(signal_length//n, n)
    yarr.resize(signal_length//n, n)
    xarr_reshaped = xarr.reshape((-1,n))
    yarr_reshaped = yarr.reshape((-1,n))
    x_ave = xarr_reshaped[:,0]
    y_ave = np.nanmean(yarr_reshaped, axis=1)
    return( x_ave, y_ave)
################################################################################ 
'''
f0=open("./dataSim/param.txt","r")
nr=sum(1 for line in f0)
par=np.zeros(( nr , 15 ))
par=np.loadtxt("./dataSim/param.txt") 
'''
pp='./data/'
ndim=5
pt=np.zeros((ndim));
date=np.zeros((71,2)); 
date= np.loadtxt("./realevent.txt")



for ii in range(71): 
    #num=ii+130 ##64, 80, 91, 113, 141
    kk=ii+28
    year, num= int(date[kk,0]), int(date[kk,1])

    pt=PTRm(num, year) 
    u0=pt[0]; t0=pt[1]; tE=pt[2];  mbase=pt[3]; fb=pt[4]
    print ("Lensing parameters:  ", u0, t0, tE, mbase, fb )
    '''
    filnam1=path+'dat{0:d}.txt'.format(num)
    filnam2=path+'mod{0:d}.txt'.format(num)
    icon, u0, t0, tE, mbase, fb,  q, dis, ksi,  chi1, chi2, dchi, ndat, std1 , rho= par[num,:]
    print("******************************************************************")    
    print("Path,   counter:  ",  path,  num)
    print ("File_names:      ",  filnam1,   filnam2)
    if(not(filnam1) or not(filnam2) or not(path) or icon!=num): 
        print("Error icon, num: ",  icon, num) 
        print("The file does not exit!!!")
        input("Enter a number")
    '''   
    ############################################################################
    '''
    dg=pd.read_csv(filnam2, delim_whitespace=True)
    dg.columns=["tim2","mag1","mag2"]
    timm=np.array( dg["tim2"])
    mag1=np.array( dg["mag1"])
    mag2=np.array( dg["mag2"])            
    '''
    ############################################################################
    path= pp+'OGLE_'+str(kk+1)
    filnam=path+'/OGLE{0:d}_{1:d}.txt'.format(year,num)
    df=pd.read_csv(filnam, delim_whitespace=True)
    df.columns=["#HJD","I mag","magerror","seeing","Sky Level"]
    x=np.array(df["#HJD"])-2450000.0
    y=np.array((df["I mag"]))
    z=np.array(df["magerror"])
    ndat=int(len(x));
    i1=0; i2=ndat-1;  flag=0.0
    print("x:  ",  x[0], x[ndat-1],  float(-5.97*tE+t0),  float(5.97*tE+t0)   )
    for i in range(ndat):  
        if(x[i]>float(-5.97*tE+t0) and flag<0.5):  
            i1=int(i)
            print ("i1: ", i1)
            flag=1.0
        if(flag>0.5 and flag<1.5 and x[i]>float(5.97*tE+t0) ): 
            i2=int(i)
            print("i2:  ", i2)
            flag=2.0
    if(x[ndat-1]<float(5.97*tE+t0) ):
        x2=int(ndat-1)
    
                
    #print("Counters:  ",   i1,   i2)        
    #input("Enter a number ")

    
    #df=pd.read_csv(filnam1, delim_whitespace=True)
    #df.columns=["tim","Imag","mager", "magm1", "magm2"]
    #xb=    np.array( df["tim"] )
    #yb=    np.array( df["Imag"])
    #zb=    np.array( df["mager"])
    #magm1b=np.array( df["magm1"])
    #magm2b=np.array( df["magm2"])
    #std1=0.0; 
    #for i in range(len(zb)):
    #    errm=abs(magm2b[i]-yb[i]) 
    #    errm=(np.random.normal(0.0,zb[i],1)) ##RandN(sigma,5.0)
    #    yb[i]=float(magm2b[i]+errm) 
    #x=xb[::8]
    #y=yb[::8]
    #z=zb[::8]
    #magm1=magm1b[::8]
    #magm2=magm2b[::8]
    #ndat=int(len(x));
    scales = np.arange(0.9, 128)# that should be determined accoring to tE, possible periods for wavelets 

    k=0; 
    for wav in wwtt:
        #print("Wave:  ",  wav)
        coef, freq = pywt.cwt( y[i1:i2], scales, wav , float(x[i1+1]-x[i1])*0.1   )
        power = np.log10((abs(coef))**2.0)
        #print (power)
        #print (np.min(power),    np.max(power))
        period =1./freq

        #if(power.any<-3.0): 
        #    power=-3.0
        lm=-3.5; ##np.min(power)    
        dl= float(np.max(power) - lm )/8.0
        levels=[lm, lm+dl, lm+2.0*dl, lm+3.0*dl, lm+4.0*dl, lm+5.0*dl, lm+6*dl, lm+7.0*dl]##0.0, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        #levels=[0.01, 0.1, 1.0, 10.0,100.0,1000.0, np.max(power)]
        
        #print (levels)
        
        #levels = np.log10(levels)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.contourf(x[i1:i2], period, power, levels, extend='both',cmap='viridis' )# plt.cm.seismic) #
        ax.set_title(r"$\rm{OGLE}-$"+"{0:d}".format(year)+r"$-\rm{BLG}-$"+"{0:d}".format(num)+r"$~~\rm{CWT},~\rm{Wavelet:}$"+str(wav), fontsize=16)
        #+r"$;~q=~$"+str(round(q,3))+r"$;~d(R_{\rm{E}})=~$"+str(round(dis,1))+r"$;~\rho_{\star}=~$"+str(round(rho,3)),)
        ax.set_ylabel(r"$\rm{Period}(\rm{days})$", fontsize=18)
        ax.set_xlabel(r"$\rm{Time}(\rm{days})$", fontsize=18)
        plt.yscale('log')
        #yticks = np.arange(period.min(), period.max() )
        #ax.set_yticks(yticks)
        #ax.set_yticklabels(yticks)
        plt.xticks(fontsize=17, rotation=0)
        plt.yticks(fontsize=17, rotation=0)
        ax.invert_yaxis()
        #ylim = ax.get_ylim()
        #ax.set_ylim(ylim[0], -1)
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, orientation="vertical",pad=0.0)
        fig.savefig(path+"/CWT{0:d}_{1:d}.jpg".format(num, k) , dpi=250, bbox_inches='tight')
        #except:
        #    pass  
        k+=1
    input("Enter a number ")
    






    
