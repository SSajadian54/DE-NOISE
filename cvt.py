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

f0=open("./dataSim/param.txt","r")
nr=sum(1 for line in f0)
par=np.zeros(( nr , 15 ))
par=np.loadtxt("./dataSim/param.txt") 
path='./dataSim/'


for ii in range(nr): 
    num=ii+130 ##64, 80, 91, 113, 141
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
       
    ############################################################################
    dg=pd.read_csv(filnam2, delim_whitespace=True)
    dg.columns=["tim2","mag1","mag2"]
    timm=np.array( dg["tim2"])
    mag1=np.array( dg["mag1"])
    mag2=np.array( dg["mag2"])            
    ############################################################################
    df=pd.read_csv(filnam1, delim_whitespace=True)
    df.columns=["tim","Imag","mager", "magm1", "magm2"]
    xb=    np.array( df["tim"] )
    yb=    np.array( df["Imag"])
    zb=    np.array( df["mager"])
    magm1b=np.array( df["magm1"])
    magm2b=np.array( df["magm2"])
    std1=0.0; 
    for i in range(len(zb)):
        errm=abs(magm2b[i]-yb[i]) 
        errm=(np.random.normal(0.0,zb[i],1)) ##RandN(sigma,5.0)
        yb[i]=float(magm2b[i]+errm) 
    x=xb[::8]
    y=yb[::8]
    z=zb[::8]
    magm1=magm1b[::8]
    magm2=magm2b[::8]
    ndat=int(len(x));
    scales = np.arange(1.0, 128)

    k=0; 
    for wav in wwtt:
        print("Wave:  ",  wav)
        #try:
        coef, freq = pywt.cwt(y, scales, wav , float(x[1]-x[0]))
        power = np.log10((abs(coef))**2.0)
        print (power)
        print (np.min(power),    np.max(power))
        period =1./freq

        #if(power.any<-3.0): 
        #    power=-3.0
        lm=-2.5; ##np.min(power)    
        dl= float(np.max(power) - lm )/8.0
        levels=[lm, lm+dl, lm+2.0*dl, lm+3.0*dl, lm+4.0*dl, lm+5.0*dl, lm+6*dl, lm+7.0*dl]##0.0, 0.125, 0.25, 0.5, 1, 2, 4, 8]
        #levels=[0.01, 0.1, 1.0, 10.0,100.0,1000.0, np.max(power)]
        
        print (levels)
        
        #levels = np.log10(levels)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.contourf(x, period, power, levels, extend='both',cmap='viridis' )# plt.cm.seismic) #
        ax.set_title(r"$\rm{CWT},~\rm{Wavelet:}$"+str(wav), fontsize=16)
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
        fig.savefig("./cwt/CWT{0:d}_{1:d}.jpg".format(num, k) , dpi=250, bbox_inches='tight')
        #except:
        #    pass  
        k+=1
    input("Enter a number ")
    






    
