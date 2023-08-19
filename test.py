import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from matplotlib.figure import Figure
import warnings
warnings.filterwarnings("ignore")
rcParams["font.size"] = 13
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
wwtt=('bior2.2','bior2.4','bior2.6','bior2.8','bior3.1','bior3.3','bior3.5','bior3.7','bior3.9','bior4.4','bior5.5','bior6.8','cgau1','cgau2',       'cgau3','cgau4','cgau5','cgau6','cgau7','cgau8','coif1','coif2','coif3','coif4','coif5','coif6','coif7','coif8','coif9','coif10','coif11',       'coif12','coif13','coif14','coif15','coif16','coif17','dmey','fbsp','gaus1','gaus2','gaus3','gaus4','gaus5','gaus6','gaus7','gaus8',     'shan','rbio1.3','rbio1.5','rbio2.2','rbio2.4','rbio2.6','rbio2.8','rbio3.3','rbio3.5','rbio3.7','rbio3.9','rbio4.4','rbio5.5',
'rbio6.8', 'db2','db3','db4','db5','db6','db7','db8','db9','db10','db11','db12','db13','db14','db15','db16','db17','db18','db19',
'db20','db21','db22','db23','db24','db25','db26','db27','db28','db29','db30','db31','db32','db33','db34','db35','db36','db37','db38',
'sym2','sym3','sym4','sym5','sym6','sym7','sym8','sym9','sym10','sym11','sym12','sym13','sym14','sym15','sym16','sym17','sym18','sym19','sym20', 'beyl', 'mexh', 'cmor') 
################################################################################
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
################################################################################

num=99

dat=np.zeros((num,16))
dat=np.loadtxt("./resultOGLE.txt") 

##year,num,n,m,u0,t0,tE,mbase,fb,chinA, chinB[jmin], std1, ST[jmin], planet, float(impChi/count),float(impST/count)]) 

print("<|Chi2-1|>_before denoising:  ",  np.mean(abs(dat[:,9 ])) )
print("<|Chi2-1|>_after  denoising:  ",  np.mean(abs(dat[:,10])) )
print("<STD>_before denoising:  ",  np.mean(pow(10.0,dat[:,11]))  ) 
print("<STD>_before denoising:  ",  np.mean(pow(10.0,dat[:,12]))  )




a=np.ndarray.tolist(dat[:,2])
b=np.ndarray.tolist(dat[:,3])
mm= int( max(set(a), key=lambda x: a.count(x)))
nn= int( max(set(b), key=lambda x: b.count(x)))
print ("The most reapearted wavelet:     ", mm ,  wwtt[mm]  )
print ("The most reapearted threshol:     ",  nn   )
print ("*******************************************************")   

###=============================================================================
plt.cla()
plt.clf()
plt.clf()
fig=plt.figure(figsize=(8,6))
ax=plt.gca()
plt.hist(np.log10(dat[:,9]),20,  histtype='bar',ec='darkgreen',facecolor='green',alpha=0.95, rwidth=1.5, label=r"$\rm{Raw}~\rm{data}$")
plt.hist(np.log10(dat[:,10]),20, histtype='bar',ec='magenta',facecolor='m',alpha=0.35, rwidth=1.5, label=r"$\rm{De-noised}~\rm{data}$")

y_vals = ax.get_yticks()
ax.set_yticklabels(['{:.2f}'.format(1.0*x*(1.0/num)) for x in y_vals]) 
y_vals = ax.get_yticks()
plt.ylim([np.min(y_vals), np.max(y_vals)])

plt.xlabel(r"$\log_{10}[\chi^{2}_{\rm{n}}]$", fontsize=18)
plt.ylabel(r"$\rm{Normalized}~\rm{Distribution}$", fontsize=18)
plt.axvline(x=np.log10(1.0), color='k', linestyle='--', lw=2.)
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.grid("True")
plt.grid(linestyle='dashed')
plt.subplots_adjust(hspace=.0)
plt.legend()
plt.legend(loc='best',fancybox=True, shadow=True)
plt.legend(prop={"size":18})
fig=plt.gcf()
fig.savefig("./chi2ns.jpg")
print(">>>>>>>>>>>>>>>>>>>>>>>> The model light curve was made <<<<<<<<<<<<<<<")



plt.cla() 
plt.clf()    
plt.figure(figsize=(8, 6))
plt.plot(dat[:,6], dat[:,9]-dat[:,10], "bo", markersize=4.0, alpha=1.0)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$t_{\rm{E}}(\rm{days})$", fontsize=18)
plt.ylabel(r"$\Delta \chi^{2}_{\rm{n}}$", fontsize=18)
fig3=plt.gcf()
fig3.savefig("./imptEchi.jpg")
########################################################################
plt.cla()
plt.clf()    
plt.figure(figsize=(8, 6))
plt.plot(dat[:,7], dat[:,9]-dat[:,10], "bo", markersize=4.0, alpha=1.0)
#plt.xscale('log')
#plt.yscale('log')
plt.xticks(fontsize=17, rotation=0)
plt.yticks(fontsize=17, rotation=0)
plt.xlabel(r"$m_{\rm{base}}(\rm{mag})$", fontsize=18)
plt.ylabel(r"$\Delta \chi^{2}_{\rm{n}}$", fontsize=18)
fig3=plt.gcf()
fig3.savefig("./impmbasechi.jpg")    





