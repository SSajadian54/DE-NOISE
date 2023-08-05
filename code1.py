import matplotlib
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
from scipy.optimize import curve_fit
from skimage.restoration import denoise_wavelet
from skimage.metrics import peak_signal_noise_ratio

wwtt=('bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8','cgau1','cgau2',       'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8','coif1','coif2','coif3','coif4','coif5','coif6','coif7','coif8','coif9','coif10','coif11',       'coif12','coif13','coif14','coif15','coif16','coif17','dmey','fbsp','gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',     'shan','rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5',
'rbio6.8', 'db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19',
'db20','db21','db22','db23','db24','db25','db26','db27','db28','db29','db30','db31','db32','db33','db34','db35','db36','db37','db38',
'sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20', 'beyl') 
################################################################################      
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
################################################################################
ndim=5
Nwalkers=15
#nstep=100      
nms=  14000; 
modelA=np.zeros((nms, 2))
modelB=np.zeros((nms, 2))
best=np.zeros((ndim*3+4))
fit= np.zeros((ndim))
Epci=np.array([0.3, 10.0,  5.0,  1.5,   0.3])
p0=np.zeros((Nwalkers,ndim))


################################################################################
save2=open("./result.txt","a+")
save2.close()    
fif=open('./bestwave.txt',"a+")
fif.close()
################################################################################
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
################################################################################
def prior(p): 
    u0=p[0]; t0=p[1]; tE=p[2];  mbase=p[3]; fb=p[4]
    if(u0>0.0 and abs(u0-pt[0])<0.4 and u0<2.0 and t0>0.0 and abs(t0-pt[1])<5.0 and tE>0.0 and abs(tE-pt[2])<5.0 and mbase>0.0 and abs(mbase-pt[3])<2.5 and fb>0.0 and abs(fb-pt[4])<0.4 and fb<1.0000645): 
        return(0); 
    return(-1);     
################################################################################
def magnification(timee, u01, t01, tE1):  
    u2= ((timee-t01)/tE1)**2.0+ u01**2.0 
    As= float(u2+2.0)/np.sqrt(u2*(u2+4.0))
    return(As)
################################################################################
def lnlike2(p,tim,Magn,errm):
    U0=p[0]; T0=p[1];  TE=p[2];  mbase=p[3];  fb=p[4]
    chi2=0.0 
    for i in range(len(tim)): 
        As=magnification(tim[i],  U0, T0, TE )
        mm= float(mbase-2.5*np.log10(As*fb + 1.0-fb) )
        chi2+=float((mm-Magn[i])**2.0/(errm[i]**2.0) )
    return float(-0.5*chi2)
################################################################################
def chi2(p,tim,Magn,errm):
    lp=prior(p)
    if(lp<0.0):
        return(-np.inf)
    return lnlike2(p, tim, Magn , errm)
################################################################################    
def func(xi, slope, offset):
    return (slope * xi + offset)
################################################################################
def FitML(tim, Magn, errm, number, year, flag):
    sampler=emcee.EnsembleSampler(Nwalkers,ndim, chi2, args=(tim,Magn,errm), threads=8)
    sampler.run_mcmc(p0, 15000, progress=True)
    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)
    for k in range(ndim):
        mcmc = np.percentile(flat_samples[:, k], [16, 50, 84])
        q = np.diff(mcmc)
        fit[k]=mcmc[1]
        best[k*3+0], best[k*3+1], best[k*3+2]= mcmc[1],   q[0], q[1]
    chi_fit=float(-2.0*chi2(fit,tim, Magn, errm))    
    best[3*ndim]=chi_fit/float(len(tim)-5.0+0.00006514635463)
    best[3*ndim+1]=number
    best[3*ndim+2]=year
    best[3*ndim+3]=flag
    save2=open("./result.txt","a+")
    np.savetxt(save2,best.reshape(-1,ndim*3+4),fmt='%.5f %.4f %.4f   %.5f %.4f %.4f  %.5f %.4f %.4f  %.5f %.4f %.4f  %.5f %.4f %.4f  %.6f %d %d %d')
    if(flag==1):      save2.write("************************************************\n")
    save2.close()
    print("Chi2, Chi2_norm:    ", chi_fit,   chi_fit/float(len(tim)-5.0) )
    print("Best fitted parameters:  ", fit)
    return(fit[0], fit[1], fit[2], fit[3], fit[4], float(chi_fit/float(len(tim)-5.0))   )
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
def RMSE(yde, xc2, co): 
    nu=int(len(co));
    yfi=np.zeros(nu);
    yc2=np.zeros(nu);  
    for jl in range(nu):
        v=int(co[jl])
        yc2[jl]=float(yde[v])
    ini = np.array([ 0.0 , np.mean(yc2) ])
    fitc, covm = curve_fit(func, xc2, yc2, ini, maxfev=1000000)
    yfi= [func(xj , float(fitc[0]), float(fitc[1]) )  for xj in xc2 ]
    rmse= np.sqrt(np.sum((yfi-yc2)**2)/float(nu-1.0));     
    return(rmse, yfi)
################################################################################
def RMSE2(y, yden):## RMSE
    return(np.sqrt(np.sum((y-yden)**2.0)/len(y) ) ) 
################################################################################
foldn=[];  nf=0
pp='./data/'
direc=os.listdir(pp)
for entry in direc:
    if 'OGLE' in entry:
        foldn.append(entry)
        nf+=1 

for d in range(nf): 
    name=str.split(foldn[d],"OGLE")[1]
    year=int(str(str.split(name,"_")[0]))
    num =int(str(str.split(name,"_")[1]))
    path= pp+'OGLE'+str(year)+'_'+str(num)
    filnam=path+'/OGLE{0:d}_{1:d}.txt'.format(year,num)
    print ("******************************************************************")    
    print("Path, year, nam:  ",  path,  year, name)
    print ("File name:  ",filnam)
    if(not(filnam) or not(path)): 
        print ("year:  ",  name,  year,  num , path  , "\t",  filnam) 
        print("The file does not exit   !!!")
        input("Enter a number")
                
    ############################################################################
    df=pd.read_csv(filnam, delim_whitespace=True)
    df.columns=["#HJD","I mag","magerror","seeing","Sky Level"]
    x=np.array(df["#HJD"])-2450000.0
    y=np.array((df["I mag"]))
    z=np.array(df["magerror"])
    nd=int(len(x)); 


    plt.figure(figsize=(10, 6))
    plt.errorbar(x,y, yerr=z, fmt=".", markersize=4.6,color='b',ecolor='#97FFFF',elinewidth=0.4, capsize=0,alpha=0.9)
    plt.title(r"$\rm{OGLE}-$"+str(year)+r"$-\rm{BLG}-$"+str(num), fontsize=17)
    plt.xlabel(r"$\rm{time}~HJD (\rm{days})$", fontsize=17)
    plt.ylabel(r"$I-\rm{band}~\rm{magnitude}$", fontsize=17 )
    plt.gca().invert_yaxis()
    plt.grid(linestyle='dashed')
    plt.savefig(path+'/OGLE{0:d}_{1:d}.jpg'.format(year, num), dpi=250, bbox_inches='tight') 
    print(">>>>>>>>>>>>>>>>>>>  Main light curve is plotted  <<<<<<<<<<<<<<<<<")
    ############################################################################
    '''
    l=0
    xc=np.zeros((nd0));      
    co=np.zeros((nd0));
    for i in range(nd0): 
        if(abs(y[i]-y[0])<float(3.0*z[i])):  
            xc[l]=x[i]
            co[l]=i
            l+=1    
    '''
    ############################################################################
    rmse=np.zeros((3))
    snr1=np.zeros((3))
    snr2=np.zeros((3))
    ydeno=np.zeros((nd,3))
    minsd=1000000.0;    nmin=np.zeros((3,2))
    maxsn1=0.0;         #nmin2=np.zeros(2)
    maxsn2=0.0;         #nmin3=np.zeros(2)
    k=int(0)
    for wav in wwtt:
        try:
            ydeno[:,0]=denoise_wavelet(y,method='VisuShrink', mode='soft', wavelet_levels=30, wavelet=wav, rescale_sigma='True')
            ydeno[:,1]=denoise_wavelet(y,method='VisuShrink', mode='hard', wavelet_levels=30, wavelet=wav, rescale_sigma='True')
            ydeno[:,2]=denoise_wavelet(y,method='BayesShrink',mode='soft', wavelet_levels=30, wavelet=wav, rescale_sigma='True')
        except:
            pass 
        for i in range(3):
            rmse[i]=RMSE2(y, ydeno[:,i])
            snr1[i]=abs(signaltonoise2(y,ydeno[:,i]))
            snr2[i]=peak_signal_noise_ratio( (y-np.mean(y))/abs(np.max(y)-np.min(y)), 
                         (ydeno[:,i]-np.mean(ydeno[:,i]))/abs(np.max(ydeno[:,i])-np.min(ydeno[:,i])) )

          
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
    fif=open('./bestwave.txt',"a+")
    resu= np.array([year, num, nmin[0,0], nmin[0,1],  nmin[1,0],nmin[1,1], nmin[2,0],nmin[2,1],  minsd,  maxsn1, maxsn2])   
    np.savetxt(fif,resu.reshape((1,11)),fmt="%d  %d   %d %d   %d %d   %d %d    %.5f    %.5lf    %.5lf ") 
    fif.write("************************************************\n")
    fif.close()                      
    nmin1= unique_rows(nmin)
    lnew= int(len(nmin1[:,0]))
    print ("number of independent rows:  ",  lnew,  "rows: ",  nmin1) 
    ############################################################################
    pt=np.zeros((ndim));
    pt=PTRm(num, year)
    for j in range(lnew):  
        if(j==0):  n=int(nmin1[0,0]);  m=int(nmin1[0,1])
        if(j==1):  n=int(nmin1[1,0]);  m=int(nmin1[1,1])
        if(j==2):  n=int(nmin1[2,0]);  m=int(nmin1[2,1])
        try:
            ydeno[:,0]=denoise_wavelet(y,method='VisuShrink', mode='soft', wavelet_levels=30, wavelet=wwtt[n],rescale_sigma='True')
            ydeno[:,1]=denoise_wavelet(y,method='VisuShrink', mode='hard', wavelet_levels=30, wavelet=wwtt[n],rescale_sigma='True')
            ydeno[:,2]=denoise_wavelet(y,method='BayesShrink',mode='soft', wavelet_levels=30, wavelet=wwtt[n],rescale_sigma='True')
        except:
            pass
        rmse[m]=RMSE2(y, ydeno[:,m])
        snr1[m]=  abs(signaltonoise2(y, ydeno[:,m]));
        snr2[m]=peak_signal_noise_ratio((y-np.mean(y))/abs(np.max(y)-np.min(y)),(ydeno[:,m]-np.mean(ydeno[:,m]))/abs(np.max(ydeno[:,m])-np.min(ydeno[:,m])))
                                                                             
        ########################################################################
        p0[:,:]=[pt+Epci*np.random.randn(ndim) for i in range(Nwalkers)]  
        u0,t0,tE, Mbase,fb, chinA= FitML(x, y, z, num, year, 0)
        dt= float((np.max(x)-x[0])/nms)
        for s in range(nms): 
            modelA[s,0]=float(x[0] + s*dt)
            modelA[s,1]=float(Mbase-2.5*np.log10(fb*magnification(modelA[s,0],u0,t0,tE) +1.0-fb)) 
        ########################################################################
        u0,t0,tE, Mbase,fb, chinB= FitML(x, ydeno[:,m], z, num, year, 1)
        dt= float((np.max(x)-x[0])/nms)
        for s in range(nms): 
            modelB[s,0]=float(x[0] + s*dt)
            modelB[s,1]=float(Mbase-2.5*np.log10(fb*magnification(modelB[s,0],u0,t0,tE) +1.0-fb)) 
        ########################################################################  
        dy=float(np.max(y)-np.min(y))/20.0
        plt.cla()
        plt.clf()    
        plt.figure(figsize=(8, 6))
        
        plt.plot(x,ydeno[:,m],'go',markersize=2.4,label=r"$\rm{Denoised}~\rm{Data}$", alpha=1.0)
        plt.errorbar(x,y,yerr=z,fmt=".",markersize=7.0,color='magenta',ecolor='magenta',elinewidth=0.1,capsize=0,alpha=0.1,label=r"$\rm{Raw}~\rm{Data}$")
        plt.plot(modelA[:,0], modelA[:,1], 'k-' , lw=1.05)
        plt.plot(modelB[:,0], modelB[:,1], 'k--', lw=1.05)
        
        plt.title(r"$\rm{Wavelet:~}$"+str(wwtt[n])+r"$;~u_{0}=~$"+str(round(u0,1))+r"$,~t_{0}=~$"+str(round(t0,1))+r"$,~t_{\rm{E}}=~$"+str(round(tE,1))+r"$,~\chi^{2}_{n}= $"+str(round(chinA,2))+",~~"+str(round(chinB,2)), fontsize=13.5)
        
        plt.text(x[10],np.min(y)+4*dy, r"$\rm{RMSE}=~$"+str(round(rmse[m],3)),     fontsize=15 )
        plt.text(x[10],np.min(y)+6*dy, r"$\rm{SNR}_{1}=~$"+str(round(snr1[m],3)),  fontsize=15 )
        plt.text(x[10],np.min(y)+8*dy, r"$\rm{SNR}_{2}=~$"+str(round(snr2[m],1)),  fontsize=15 )
        
        plt.xlabel(r"$\rm{time}~HJD (\rm{days})$", fontsize=17 )
        plt.ylabel(r"$I-\rm{band}~ \rm{magnitude}$", fontsize=17)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.legend(prop={"size":13.5})
        fig=plt.gcf()
        fig.savefig(path+"/DNOGLE{0:d}_{1:d}_{2:d}.jpg".format(year,num,j) , dpi=250, bbox_inches='tight') 
        ########################################################################  
        y1=Mbase-2.5*np.log10(fb*magnification(t0,u0,t0,tE) +1.0-fb)
        y2=Mbase
        dyy=abs(y2-y1)/20.0
        plt.cla()
        plt.clf()    
        plt.figure(figsize=(8, 6))
        plt.plot(x,ydeno[:,m],'go',markersize=2.4,label=r"$\rm{Denoised}~\rm{Data}$", alpha=1.0)
        plt.errorbar(x,y,yerr=z,fmt=".",markersize=7.9,color='magenta',ecolor='magenta',elinewidth=0.1,capsize=0,alpha=0.18,label=r"$\rm{Raw}~\rm{Data}$")
        plt.plot(modelA[:,0], modelA[:,1], 'k-' , lw=1.05)
        plt.plot(modelB[:,0], modelB[:,1], 'k--', lw=1.05)
        
        plt.xlim([-3.0*tE+t0, 3.0*tE+t0])
        plt.ylim([y1-dyy*2.0, y2+dyy*2.0])
        
        plt.title(r"$\rm{Wavelet:~}$"+str(wwtt[n])+r"$;~u_{0}=~$"+str(round(u0,1))+r"$,~t_{0}=~$"+str(round(t0,1))+r"$,~t_{\rm{E}}=~$"+str(round(tE,1))+r"$,~\chi^{2}_{n}= $"+str(round(chinA,2))+",~~"+str(round(chinB,2)), fontsize=13.5)
        
        plt.text(-2.98*tE+t0,y1+4*dyy, r"$\rm{RMSE}=~$"+str(round(rmse[m],3)),     fontsize=15 )
        plt.text(-2.98*tE+t0,y1+6*dyy, r"$\rm{SNR}_{1}=~$"+str(round(snr1[m],3)),  fontsize=15 )
        plt.text(-2.98*tE+t0,y1+8*dyy, r"$\rm{SNR}_{2}=~$"+str(round(snr2[m],1)),  fontsize=15 )
        
        plt.xlabel(r"$\rm{time}~HJD (\rm{days})$", fontsize=17 )
        plt.ylabel(r"$I-\rm{band}~ \rm{magnitude}$", fontsize=17)
        plt.gca().invert_yaxis()
        plt.legend()
        plt.legend(prop={"size":13.5})
        fig=plt.gcf()
        fig.savefig(path+"/CNOISOGLE{0:d}_{1:d}_{2:d}.jpg".format(year,num,j) , dpi=250, bbox_inches='tight')     
    
   
