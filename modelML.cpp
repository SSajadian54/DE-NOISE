#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <sys/timeb.h>
#include "VBBinaryLensingLibrary.h"
///##################################################
using namespace std;


double RandN(double sig, double nnd);
double RandR(double down, double up);
double Errogle(double mag); 
//###################################################
//////////////////////////////////////////////////////
    time_t _timeNow;
      unsigned int _randVal;
    unsigned int _dummyVal;
    FILE * _randStream;
///////////////////////////////////////////////////////
int main()
{
///****************************************************************************
//gettimeofday(&newTime,NULL);
//ftime(&tp);
	time(&_timeNow);
	_randStream = fopen("/dev/urandom", "r");
	_dummyVal = fread(&_randVal, sizeof(_randVal), 1, _randStream);
	srand(_randVal);
    _dummyVal = fread(&_randVal, sizeof(_randVal), 1, _randStream);
	srand(_randVal);
///****************************************************************************

    VBBinaryLensing vbb;
    vbb.Tol=1.e-4;
    vbb.LoadESPLTable("./files/ESPL.tbl");
    
    char filnam1[40];  
    char filnam2[40];  
    FILE *fil1;  
    FILE *fil2; 
    
    double tE, t0, u0, mbase, fb,  q, dis, ksi, tim, cade, timp=0.0; 
    double dt, chi1, chi2, Astar0, Astar1, magni0, magni1;
    double xlens, ylens, dchi, sigma , errm, u, STD, rho;    

    cade=double(15.0/60.0/24.0);
    dt= cade/2.0; 
    int i=0;  


    FILE * param;
    param=fopen("./param.txt","w");
 
 
 


    do{    
    sprintf(filnam1,"./%c%c%c%d.txt",'d','a','t',i);
    fil1=fopen(filnam1,"w");
    
    sprintf(filnam2,"./%c%c%c%d.txt",'m','o','d',i);
    fil2=fopen(filnam2,"w");
   
   
    tE= RandR(20.0,60.0);  
    t0= 0.0;  
    u0=RandR(0.1,0.9);  
    mbase=RandR(18.5,21.0);  
    fb=RandR(0.1,1.0);  
    q=RandR(-3.5,-1.5);  
    q=pow(10.0 , q); 
    dis=RandR(0.9 , 1.7);  
    ksi=RandR(0.0 , 2.0*M_PI); 
    rho=pow(10.0, RandR(-3.0,-1.5) );  
    timp=0.0; 
    chi1=0.0;  
    chi2=0.0;  
    cout<<"step:  "<<i<<" *************************************************"<<endl; 
    cout<<"tE:  "<<tE<<"\t t0:  "<<t0<<"\t u0:  "<<u0<<endl;
    cout<<"mbase:  "<<mbase<<"\t fb:  "<<fb<<"\t q:  "<<q<<"\t dis:  "<<dis<<endl;
    
    int ndat=0;
    STD=0.0;   
   
    for(tim=double(-3.5*tE+t0);  tim<double(3.5*tE+t0);  tim+=dt){
    timp+=dt;   
    xlens = (tim-t0)/tE * cos(ksi) - u0 * sin(ksi);
    ylens = (tim-t0)/tE * sin(ksi) + u0 * cos(ksi);
    u=sqrt(xlens*xlens+  ylens*ylens);  
    
    Astar1= vbb.BinaryMag2(dis, q, xlens, ylens, rho);
    magni1= mbase-2.5*log10( fb*Astar1+1.0-fb); 
    Astar0= vbb.ESPLMag2(u, rho);
    magni0= mbase-2.5*log10( fb*Astar0+1.0-fb); 
    
    if(Astar1<1.0  or Astar0<1.0 or u<u0){
    cout<<"Error Astar0:  "<<Astar0<<"\t Astar1:  "<<Astar1<<"\t u:  "<<u<<endl;
    int uue;  cin>>uue;  } 
    //cout<<"Astar1:  "<<Astar1<<"\t Astar0:  "<<Astar0<<"\t u:  "<<u<<endl;
    
    fprintf(fil2, "%.5lf  %.5lf   %.5lf\n", tim, magni0, magni1);
    
    
    if(timp>cade){
    timp=timp-cade; 
    sigma=Errogle(magni1); 
    errm = fabs(RandN(sigma,2.5));
    if(ndat%2==0)  errm=-1.0*errm;  
    fprintf(fil1, "%.5lf     %.6lf     %.6lf  %.5lf  %.5lf\n", tim, magni1+errm,sigma,magni0,magni1);
    chi1+= double(magni1+errm-magni0)*(magni1+errm-magni0)/(sigma*sigma); 
    chi2+= double(magni1+errm-magni1)*(magni1+errm-magni1)/(sigma*sigma);
    ndat+=1;
    STD+= double(magni1+errm-magni1)*(magni1+errm-magni1);} 
    }  
    fclose(fil2); 
    fclose(fil1);  
    dchi=fabs(chi1-chi2);  
    STD=sqrt( STD/(ndat-1.0)); 
    fprintf(param,"%d  %.5lf  %.5lf   %.5lf   %.5lf   %.5lf   %.5lf   %.5lf   %.5lf  %.1lf %.1lf  %.4lf  %d   %.7lf\n", 
    i, u0, t0, tE, mbase, fb,  q, dis, ksi,  chi1, chi2, dchi, ndat, STD);
    i+=1; 
    }while(i<500);    
  //}  
    fclose(param);  
return(0); 
}
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
double RandR(double down, double up){
    double p =(double)rand()/((double)(RAND_MAX)+(double)(1.0));
    return(p*(up-down)+down);
}
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
double Errogle(double mag){
   double error;  
   if(mag<15.0 ) error=0.003;   //8.62339038e-03 6.82867290e-03 2.27422710e-03 1.66494077e+01
   if(mag>=15.0) error=0.00862339038 + 0.00682867290*(mag-16.6494077) +  0.00227422710*(mag-16.6494077)*(mag-16.6494077);  
   if((mag<18.0 and  error>0.097) or error <0.003 or error>0.1){cout<<"Error(OGLE) error:  "<<error<<"\t mag:  "<<mag<<endl;    }
   return(error);  
}
///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
double RandN(double sig, double nnd){
   double rr,f,frand;
   do{
   rr=double(((double)rand()/(double)(RAND_MAX+1.))*2.0-1.0)*sig*nnd; ///[-N sigma:N sigma]
   f= exp(-0.5*rr*rr/sig/sig);
   frand=fabs((double)rand()/((double)(RAND_MAX+1.))*1.0);
   }while(frand>f);
   return rr;
}

///&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
