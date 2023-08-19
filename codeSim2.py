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
from skimage.restoration import estimate_sigma
from skimage.metrics import peak_signal_noise_ratio

#'bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8',
#'rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5','rbio6.8','coif1',

wwtt=('bior1.1', 'bior1.3','rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5','rbio6.8', 'bior1.5','bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8','cgau5','cgau6','cgau7','cgau8','coif1','coif2','coif3','coif4','coif5','coif6','coif7','coif8','coif9','coif10','coif11',       'coif12','coif13','coif14','coif15','coif16','coif17','fbsp','gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',     'shan', 'db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19',
'db20','db21','db22','db23','db24','db25','db26','db27','db28','db29','db30','db31','db32','db33','db34','db35','db36','db37','db38',
'sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20', 'beyl', 'mexh', 'cmor')


#'bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20''cgau1','cgau2',       'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8','coif2','coif3','coif4','coif5','coif6','coif7','coif8','coif9','coif10','coif11',       'coif12','coif13','coif14','coif15','coif16','coif17','dmey','fbsp','gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',     'shan','db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19','db20','db21','db22','db23','db24','db25','db26','db27','db28','db29','db30','db31','db32','db33','db34','db35','db36','db37','db38','sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20', 'beyl') 
################################################################################
wavel=10
#impro=0.0
#nimp=0.0
################################################################################ 

save3=open("./wavelSc.txt","w")
save3.close()  

fif=open('./bestwaveSc.txt',"w")
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
par=np.zeros(( nr , 15 ))
par=np.loadtxt("./dataSim/param.txt") 
path='./dataSim/'


outp=np.zeros((nr, 4))

impSD=0.0
good=0.0
ST=np.zeros((4))
dchin=np.zeros((4))
for ii in range(nr): 
    num=ii
    filnam1=path+'dat{0:d}.txt'.format(num)
    filnam2=path+'mod{0:d}.txt'.format(num)
    icon, u0, t0, tE, mbase, fb,  q, dis, ksi,  chi1, chi2, dchi, ndat, std1 , rho= par[num,:]
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
        '''
        if(i%2==0):  
            yb[i]=float(magm2b[i]-errm)       
        else:        
            yb[i]=float(magm2b[i]+errm)       
        '''
    x=xb[::16]
    y=yb[::16]
    z=zb[::16]
    magm1=magm1b[::16]
    magm2=magm2b[::16]
    ndat=int(len(x));

    '''
    maxl=[]
    for wav in wwtt:
        maxl.append( pywt.dwt_max_level(ndat,wav) )
    print("Max_level:  ",  maxl)
    '''
    ############################################################################
    '''
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
    '''
    ############################################################################
    rmse=np.zeros((3))
    snr1=np.zeros((3))
    snr2=np.zeros((3))
    ydeno=np.zeros((ndat,3))
    nmin=np.zeros((4,2))
    minsd=1000000.0;   
    minst=100000.0 
    maxsn1=0.0;         
    maxsn2=0.0;         
    k=int(0)
    for wav in wwtt:
        #print("Wave:  ",  wav)
        try:
            #SE = estimate_sigma(y, average_sigmas=True)*1.0
            #wavel= int(maxl[k]-2)
            ydeno[:,0]=denoise_wavelet(y,method='VisuShrink', mode='soft',wavelet_levels=wavel, wavelet=wav,rescale_sigma='True')
            ydeno[:,1]=denoise_wavelet(y,method='VisuShrink', mode='hard',wavelet_levels=wavel, wavelet=wav,rescale_sigma='True')
            ydeno[:,2]=denoise_wavelet(y,method='BayesShrink',mode='soft',wavelet_levels=wavel, wavelet=wav, rescale_sigma='True')
        except:
            pass 
        for i in range(3):
            rmse[i]=RMSE2(y, ydeno[:,i])    
            snr1[i]=abs(signaltonoise2(y,ydeno[:,i]))
            snr2[i]=peak_signal_noise_ratio( (y-np.mean(y))/abs(np.max(y)-np.min(y)), 
                         (ydeno[:,i]-np.mean(ydeno[:,i]))/abs(np.max(ydeno[:,i])-np.min(ydeno[:,i])))
            #std1=  np.log10(np.sqrt(np.sum((y    -     magm2)**2.0)/(ndat-1.0)))
            ST[i]= np.log10(np.sqrt(np.sum((ydeno[:,i]-magm2)**2.0)/(ndat-1.0)))                           
            if(minsd>rmse[i]): 
                minsd=float(rmse[i])
                nmin[0,:]=k, i        
            if( maxsn1<snr1[i]): 
                maxsn1=float(snr1[i])
                nmin[1,:]=k, i
            if(maxsn2<snr2[i]): 
                maxsn2=float(snr2[i])
                nmin[2,:]=k, i
            if(minst>ST[i]): 
                minst=float(ST[i])
                nmin[3,:]=k, i
            
        ########################################################################  
            '''
            if(k>43):##try: 
                dy=float(np.max(y)-np.min(y))/20.0
                plt.cla()
                plt.clf()    
                plt.figure(figsize=(8, 6))
                plt.plot(x,ydeno[:,i],'bo',markersize=1.0,label=r"$\rm{Denoised}~\rm{Data}$", alpha=0.4)
                plt.errorbar(x,y,yerr=z,fmt=".",markersize=3.0,color='magenta',ecolor='magenta',elinewidth=0.1,capsize=0,alpha=0.06,label=r"$\rm{Raw}~\rm{Data}$")
                plt.plot(timm, mag1, "g--", lw=1.4, label=r"$\rm{Single}~\rm{microlensing}$")
                plt.plot(timm, mag2, "c:",  lw=1.4, label=r"$\rm{Planetary}~\rm{microlensing}$")
                plt.title(r"$\rm{Wavelet:}$"+str(wav)+r"$;~q=~$"+str(round(q,3))+r"$;~d(R_{\rm{E}})=~$"+str(round(dis,1))+r"$;~\rho_{\star}=~$"+str(round(rho,3))+r"$,~\log_{10}[\rm{STD}]=$"+str(round(std1,3))+",~"+str(round(ST[i],3)),fontsize=14.0)
                plt.text(-2.924*tE,np.min(y)+4*dy, r"$\rm{RMSE}=~$"+str(round(rmse[i],3)),     fontsize=17 )
                plt.text(-2.924*tE,np.min(y)+6*dy, r"$\mathcal{P}=~$"+str(round(snr1[i],3)),  fontsize=17 )
                plt.text(-2.924*tE,np.min(y)+8*dy, r"$\rm{SNR}_{\rm{max}}=~$"+str(round(snr2[i],1)),  fontsize=17 )
                plt.xlim([-2.97*tE, 2.97*tE])
                plt.ylim([np.min(mag2)-np.mean(z)*1.5, np.max(mag2)+np.mean(z)*1.5])
                plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=18 )
                plt.ylabel(r"$I-\rm{band}~\rm{magnitude}$", fontsize=18)
                plt.xticks(fontsize=17, rotation=0)
                plt.yticks(fontsize=17, rotation=0)
                plt.gca().invert_yaxis()
                plt.legend()
                plt.legend(prop={"size":13.0}, loc="upper right")
                plt.subplots_adjust(hspace=.0)
                fig=plt.gcf()
                fig.tight_layout(pad=0.1)
                fig.savefig(path+"figs/1/DNSim{0:d}_{1:d}.jpg".format(k,i) , dpi=250, bbox_inches='tight')
            #except:
            #    pass 
            '''          
        ########################################################################        
        k+=1       
    fif=open('./bestwaveSc.txt',"a+")
    fif.write("************************************************\n")
    resu= np.array([num, nmin[0,0], nmin[0,1],  nmin[1,0],nmin[1,1], nmin[2,0],nmin[2,1],nmin[3,0],nmin[3,1],minsd,maxsn1,maxsn2,minst])   
    np.savetxt(fif,resu.reshape((1,13)),fmt="%d    %d %d    %d %d   %d %d   %d %d   %.5f    %.5f    %.5f   %.6f")
    fif.close()                            
    nmin1= unique_rows(nmin)
    lnew= int(len(nmin1[:,0]))
    print ("number of independent rows,  rows: ",lnew,    nmin1,      num) 
    ############################################################################
    ST[0], ST[1], ST[2], ST[3]=100000.0, 100000.0, 100000.0, 10000.0
    dchin[0], dchin[1], dchin[2], dchin[3]=0.0, 0.0, 0.0, 0.0
    for j in range(lnew):  
        n=int(nmin1[j,0]);  m=int(nmin1[j,1])
        try:
            #SE = estimate_sigma(y, average_sigmas=True)*1.0# sigma=SE, for VisuShrink
            #wavel= int(maxl[n]-2)
            ydeno[:,0]=denoise_wavelet(y,method='VisuShrink',mode='soft',wavelet_levels=wavel,wavelet=wwtt[n],rescale_sigma='True')
            ydeno[:,1]=denoise_wavelet(y,method='VisuShrink',mode='hard',wavelet_levels=wavel,wavelet=wwtt[n],rescale_sigma='True')
            ydeno[:,2]=denoise_wavelet(y,method='BayesShrink',mode='soft', wavelet_levels=wavel, wavelet=wwtt[n],rescale_sigma='True')
        except:
            pass
        rmse[m]=RMSE2(y, ydeno[:,m])
        snr1[m]=abs(signaltonoise2(y, ydeno[:,m]));
        snr2[m]=peak_signal_noise_ratio((y-np.mean(y))/abs(np.max(y)-np.min(y)),
             (ydeno[:,m]-np.mean(ydeno[:,m]))/abs(np.max(ydeno[:,m])-np.min(ydeno[:,m])))
        chia=np.sum( (ydeno[:,m]-magm1)**2.0/(z**2))
        chib=np.sum( (ydeno[:,m]-magm2)**2.0/(z**2))
        std1=  np.log10(np.sqrt(np.sum((y    -     magm2)**2.0)/(ndat-1.0)))
        ST[j]= np.log10(np.sqrt(np.sum((ydeno[:,m]-magm2)**2.0)/(ndat-1.0)))
        dchin[j]= abs(chia-chib)
        ########################################################################  
        '''
        if(True):##try: 
            dy=float(np.max(y)-np.min(y))/20.0
            plt.cla()
            plt.clf()    
            plt.figure(figsize=(8, 6))
            plt.plot(x,ydeno[:,m],'bo',markersize=1.0,label=r"$\rm{Denoised}~\rm{Data}$", alpha=0.4)
            plt.errorbar(x,y,yerr=z,fmt=".",markersize=3.0,color='magenta',ecolor='magenta',elinewidth=0.1,capsize=0,alpha=0.16,label=r"$\rm{Raw}~\rm{Data}$")
            plt.plot(timm, mag1, "g--", lw=1.4, label=r"$\rm{Single}~\rm{microlensing}$")
            plt.plot(timm, mag2, "c:",  lw=1.4, label=r"$\rm{Planetary}~\rm{microlensing}$")
            plt.title(r"$\rm{Wavelet:}$"+str(wwtt[n])+r"$;~q=~$"+str(round(q,3))+r"$;~d(R_{\rm{E}})=~$"+str(round(dis,1))+r"$;~\rho_{\star}=~$"+str(round(rho,3))+r"$,~\log_{10}[\rm{STD}]=$"+str(round(std1,3))+",~"+str(round(ST[j],3)),fontsize=14.0)
            plt.text(-2.924*tE,np.min(y)+4*dy, r"$\rm{RMSE}=~$"+str(round(rmse[m],3)),     fontsize=17 )
            plt.text(-2.924*tE,np.min(y)+6*dy, r"$\mathcal{P}=~$"+str(round(snr1[m],3)),  fontsize=17 )
            plt.text(-2.924*tE,np.min(y)+8*dy, r"$\rm{SNR}_{\rm{max}}=~$"+str(round(snr2[m],1)),  fontsize=17 )
            plt.xlim([-2.97*tE, 2.97*tE])
            plt.ylim([np.min(mag2)-np.mean(z)*1.5, np.max(mag2)+np.mean(z)*1.5])
            plt.xlabel(r"$\rm{time}(\rm{days})$", fontsize=18 )
            plt.ylabel(r"$I-\rm{band}~\rm{magnitude}$", fontsize=18)
            plt.xticks(fontsize=17, rotation=0)
            plt.yticks(fontsize=17, rotation=0)
            plt.gca().invert_yaxis()
            plt.legend()
            plt.legend(prop={"size":14.0})
            plt.subplots_adjust(hspace=.0)
            fig=plt.gcf()
            fig.tight_layout(pad=0.1)
            fig.savefig(path+"figs/DNSim{0:d}_{1:d}.jpg".format(num,j) , dpi=250, bbox_inches='tight')
        #except:
        #    pass    
        '''
    jmin=np.argmin(ST)
    impSD +=float( pow(10.0,std1)- pow(10.0,ST[jmin]) )
    if(ST[jmin]<float(-2.79)):       
        good+=1.0
    n=int(nmin1[jmin,0]);  
    m=int(nmin1[jmin,1])    
    save3=open("./wavelSc.txt","a+")
    resu= np.array([num,n,m,u0,tE,mbase,fb,q,dis,ksi,rho, dchi, dchin[jmin], std1, ST[jmin], good])   
    np.savetxt(save3,resu.reshape((1,16)),fmt="%d  %d  %d  %.5f   %.5f  %.5f %.5f %.6f %.4f %.4f   %.6f  %.2f  %.2f  %.6f  %.6f  %.1f") 
    save3.close()
    outp[num, 0]=n
    outp[num, 1]=m
    outp[num, 2]=rho
    outp[num, 3]= float( pow(10.0,std1)- pow(10.0,ST[jmin]) )
    print ("STDs:    ", std1,     ST[jmin])
    print ("good_now:  ",  good,       float(good*100.0/nr))
    print ("Improvement in STD (mag):   ",  float(impSD/nr))
    if(num>5): 
        a=np.ndarray.tolist(outp[:num,0])
        b=np.ndarray.tolist(outp[:num,1])
        mm= int( max(set(a), key=lambda x: a.count(x)))
        nn= int( max(set(b), key=lambda x: b.count(x)))
        print ("The most reapearted wavelet:     ", mm ,  wwtt[mm]  )
        print ("The most reapearted threshol:     ",  nn   )
    print ("*******************************************************")   

#################################################

plt.cla()
plt.clf()    
plt.figure(figsize=(8, 6))
plt.plot(outp[:,2], outp[:,3],"bo", markersize=4.0, alpha=1.0)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\rho_{\star}$", fontsize=18)
plt.ylabel(r"$\Delta \rm{STD}(\rm{mag})$", fontsize=18)
fig3=plt.gcf()
fig3.savefig("./impo_rho2.jpg")

#################################################




