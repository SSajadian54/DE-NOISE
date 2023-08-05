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
import os
#from scipy.optimize import curve_fit
from skimage.restoration import denoise_wavelet
from skimage.metrics import peak_signal_noise_ratio

#'bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8',
#'rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5','rbio6.8','coif1',

wwtt=('bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8','cgau1','cgau2',       'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8','coif1','coif2','coif3','coif4','coif5','coif6','coif7','coif8','coif9','coif10','coif11',       'coif12','coif13','coif14','coif15','coif16','coif17','dmey','fbsp','gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',     'shan','rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5',
'rbio6.8', 'db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19',
'db20','db21','db22','db23','db24','db25','db26','db27','db28','db29','db30','db31','db32','db33','db34','db35','db36','db37','db38',
'sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20', 'beyl')
#'cgau1','cgau2',       'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8','coif2','coif3','coif4','coif5','coif6','coif7','coif8','coif9','coif10','coif11',       'coif12','coif13','coif14','coif15','coif16','coif17','dmey','fbsp','gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',     'shan','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','db21','db22','db23','db24','db25','db26','db27','db28','db29','db30','db31','db32','db33','db34','db35','db36','db37','db38','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20', 'beyl') 
################################################################################
wavel=10
impro=0.0
nimp=0.0
################################################################################ 

save3=open("./wavelS.txt","w")
save3.close()  

fif=open('./bestwaveS.txt',"w")
fif.close()

################################################################################
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
################################################################################
def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)    
################################################################################
def signaltonoise2(a, b):
    am= np.mean(a)
    bm= np.mean(b) 
    A=np.sum((a-am)*(b-bm))
    B=np.sqrt(np.sum((a-am)**2.0)*np.sum((b-bm)**2.0))
    return (abs(A/(B+0.0000000234523645)) )    
################################################################################
def RMSE2(y, yden):
    return(np.sqrt(np.sum((y-yden)**2.0)/len(y) ) ) 
################################################################################
f0=open("./dataSim/param.txt","r")
nr=sum(1 for line in f0)
par=np.zeros(( nr , 14 ))
par=np.loadtxt("./dataSim/param.txt") 
path='./dataSim/'

good=0.0
std=np.zeros((3))
for ii in range(nr): 
    num=ii
    filnam1=path+'dat{0:d}.txt'.format(num)
    filnam2=path+'mod{0:d}.txt'.format(num)
    icon, u0, t0, tE, mbase, fb,  q, dis, ksi,  chi1, chi2, dchi, ndat, std1 = par[num,:]
    #print("******************************************************************")    
    #print("Path,   counter:  ",  path,  num)
    #print ("File_names:      ",  filnam1,   filnam2)
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
    x=    np.array( df["tim"] )
    y=    np.array( df["Imag"])
    z=    np.array( df["mager"])
    magm1=np.array( df["magm1"])
    magm2=np.array( df["magm2"])
    nd=int(len(x)); 
    ############################################################################
    plt.figure(figsize=(10, 6))
    plt.plot(timm, mag1, "m--", lw=1.4)
    plt.plot(timm, mag2, "k:",  lw=1.4)
    plt.errorbar(x,y, yerr=z, fmt=".", markersize=5.2,color='b',ecolor='#97FFFF',elinewidth=0.6, capsize=0,alpha=0.75)
    plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=17)
    plt.ylabel(r"$I-\rm{band}~\rm{magnitude}$", fontsize=17)
    plt.gca().invert_yaxis()
    plt.grid(linestyle='dashed')
    plt.savefig(path+'figs/Sim{0:d}.jpg'.format(num), dpi=250, bbox_inches='tight') 
    #print(">>>>>>>>>>>>>>>>>>>  Main light curve is plotted  <<<<<<<<<<<<<<<<<")
    ############################################################################
    rmse=np.zeros((3))
    snr1=np.zeros((3))
    snr2=np.zeros((3))
    ydeno=np.zeros((nd,3))
    nmin=np.zeros((3,2))
    minsd=1000000.0;    
    maxsn1=0.0;         
    maxsn2=0.0;         
    k=int(0)
    for wav in wwtt:
        try:
            ydeno[:,0]=denoise_wavelet(y,method='VisuShrink', mode='soft',wavelet_levels=wavel, wavelet=wav, rescale_sigma='True')
            ydeno[:,1]=denoise_wavelet(y,method='VisuShrink', mode='hard',wavelet_levels=wavel, wavelet=wav, rescale_sigma='True')
            ydeno[:,0]=denoise_wavelet(y,method='BayesShrink',mode='soft',wavelet_levels=wavel, wavelet=wav, rescale_sigma='True')
        except:
            pass 
        for i in range(1):
            rmse[i]=RMSE2(y, ydeno[:,i])    
            snr1[i]=abs(signaltonoise2(y,ydeno[:,i]))
            snr2[i]=peak_signal_noise_ratio( (y-np.mean(y))/abs(np.max(y)-np.min(y)), 
                         (ydeno[:,i]-np.mean(ydeno[:,i]))/abs(np.max(ydeno[:,i])-np.min(ydeno[:,i])))          
            if(minsd>rmse[i]): 
                minsd=float(rmse[i])
                nmin[0,:]=k, i        
            if( maxsn1<snr1[i]): 
                maxsn1=float(snr1[i])
                nmin[1,:]=k, i
            if(maxsn2<snr2[i]): 
                maxsn2=float(snr2[i])
                nmin[2,:]=k, i
        k+=1       
    fif=open('./bestwaveS.txt',"a+")
    fif.write("************************************************\n")
    resu= np.array([num, nmin[0,0], nmin[0,1],  nmin[1,0],nmin[1,1], nmin[2,0],nmin[2,1], minsd, maxsn1, maxsn2])   
    np.savetxt(fif,resu.reshape((1,10)),fmt="%d    %d %d    %d %d    %d %d      %.5f    %.5f    %.5f")
    fif.close()                            
    nmin1= unique_rows(nmin)
    lnew= int(len(nmin1[:,0]))
    print ("number of independent rows:  ",  lnew,  "rows: ",  nmin1,    num) 
    ############################################################################
    std=100000.0, 100000.0, 100000.0
    for j in range(lnew):  
        if(j==0):    n=int(nmin1[0,0]);  m=int(nmin1[0,1])
        elif(j==1):  n=int(nmin1[1,0]);  m=int(nmin1[1,1])
        elif(j==2):  n=int(nmin1[2,0]);  m=int(nmin1[2,1])
        try:
            ydeno[:,0]=denoise_wavelet(y,method='VisuShrink', mode='soft', wavelet_levels=wavel, wavelet=wwtt[n],rescale_sigma='True')
            ydeno[:,1]=denoise_wavelet(y,method='VisuShrink', mode='hard', wavelet_levels=wavel, wavelet=wwtt[n],rescale_sigma='True')
            ydeno[:,0]=denoise_wavelet(y,method='BayesShrink',mode='soft', wavelet_levels=wavel, wavelet=wwtt[n],rescale_sigma='True')
        except:
            pass
        rmse[m]=RMSE2(y, ydeno[:,m])
        snr1[m]=abs(signaltonoise2(y, ydeno[:,m]));
        snr2[m]=peak_signal_noise_ratio((y-np.mean(y))/abs(np.max(y)-np.min(y)),
                  (ydeno[:,m]-np.mean(ydeno[:,m]))/abs(np.max(ydeno[:,m])-np.min(ydeno[:,m])))
        
        chia=np.sum( (ydeno[:,m]-magm1)**2.0/(z**2))
        chib=np.sum( (ydeno[:,m]-magm2)**2.0/(z**2))
        std[j]= np.sqrt(np.sum((ydeno[:,m]-magm2)**2.0)/(ndat-1.0))
        dchin=  abs(chia-chib)
        save3=open("./wavelS.txt","a+")
        resu= np.array([num, n, m, u0, t0, tE, mbase, fb, q, dis, ksi, chi1, chi2, dchi, chia, chib, dchin, std1, std[j], good])   
        np.savetxt(save3,resu.reshape((1,20)),fmt="%d  %d  %d  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.5f  %.2f  %.2f  %.2f  %.2f  %.2f %.2f %.8f %.8f %.1f") 
        save3.close()
        if(dchi>20.0):
            impro+=float(dchin-dchi)
            nimp +=1.0
            print("This light curve is planetary one:  ",    num,   dchi,   dchin)
        ########################################################################  
        #r"$;~u_{0}=~$"+str(round(u0,1))+r"$,~t_{0}=~$"+str(round(t0,1))+r"$,~t_{\rm{E}}=~$"+str(round(tE,1))
        if(std[j]<-2.79):  
        dy=float(np.max(y)-np.min(y))/20.0
        plt.cla()
        plt.clf()    
        plt.figure(figsize=(8, 6))
        plt.plot(x,ydeno[:,m],'bo',markersize=1.0,label=r"$\rm{Denoised}~\rm{Data}$", alpha=0.4)
        plt.errorbar(x,y,yerr=z,fmt=".",markersize=3.0,color='magenta',ecolor='magenta',elinewidth=0.1,capsize=0,alpha=0.06,label=r"$\rm{Raw}~\rm{Data}$")
        plt.plot(timm, mag1, "g--", lw=1.4, label=r"$\rm{Single}~\rm{microlensing}$")
        plt.plot(timm, mag2, "c:",  lw=1.4, label=r"$\rm{Planetary}~\rm{microlensing}$")
        plt.title(r"$\rm{Wavelet:~}$"+str(wwtt[n])+r"$;~q=~$"+str(round(q,3))+r"$;~d(R_{\rm{E}})=~$"+str(round(dis,2))+r"$,~\log_{10}[\rm{STD}]=$"+str(round(np.log10(std1),3))+",~~"+str(round(np.log10(std2),3)), fontsize=13.5)
        plt.text(-2.924*tE,np.min(y)+4*dy, r"$\rm{RMSE}=~$"+str(round(rmse[m],3)),     fontsize=15 )
        plt.text(-2.924*tE,np.min(y)+6*dy, r"$\rm{SNR}=~$"+str(round(snr1[m],3)),  fontsize=15 )
        plt.text(-2.924*tE,np.min(y)+8*dy, r"$\rm{SNR}_{\rm{max}}=~$"+str(round(snr2[m],1)),  fontsize=15 )
        plt.xlim([-2.97*tE, 2.97*tE])
        plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=17 )
        plt.ylabel(r"$I-\rm{band}~\rm{magnitude}$", fontsize=17)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.legend(prop={"size":13.5})
        fig=plt.gcf()
        fig.savefig(path+"figs/DNSim{0:d}_{1:d}.jpg".format(num,j) , dpi=250, bbox_inches='tight')
    std2=np.log10( np.min(std))
    if(std2<float(-2.79)):     good+=1.0
    print ("good_now:  ",  good,  float(good*100.0/nr))
        ########################################################################  
print ("Improvement:  ",   float(impro/nimp),     )      
