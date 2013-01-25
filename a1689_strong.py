import numpy

import matplotlib.pyplot as plt

from math import*

from pylab import*
import scipy.linalg.basic
import smooth_pbl

from numpy import dot

if __name__=="__main__":
    
    ra,dec,g1,g2,g1w,g2w,w,i,z=numpy.loadtxt('subaru_red_photoz.cat',unpack=True)

    ramax=numpy.max(ra)
    decmin=numpy.min(dec)
    decmean=numpy.mean(dec)
    #calculating x and y in arcsecs
    #This is for B-modes
   
    x=(ramax-ra)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y=(dec-decmin)*3600
   
    ra_peak=197.875
    dec_peak=-1.3417
  
    x_peak=(ramax-ra_peak)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y_peak=(dec_peak-decmin)*3600

    rsq=(x-x_peak)**2+(y-y_peak)**2

    delta=360
    id=numpy.where((x>(x_peak-delta))&(x<(x_peak+delta))&(y>(y_peak-delta))&(y<(y_peak+delta)))
    ra=ra[id[0]]
    dec=dec[id[0]]
    x1=x[id[0]]
    y1=y[id[0]]
    print x1.shape
    xmin=min(x1)
    ymin=min(y1)
    
    x1=(x1-min(x1))/(2*delta)
    y1=(y1-min(y1))/(2*delta)

    x_peak=(x_peak-xmin)/(2*delta)
    y_peak=(y_peak-ymin)/(2*delta)

    gam1=g1[id[0]]
    gam2=g2[id[0]]

    

    rsq=(x1-x_peak)**2+(y1-y_peak)**2
    h0=exp(rsq/(2))
    #sys.exit()
    sig_e=0.26
    n=x1.shape[0]
    sigma=numpy.zeros(n,dtype=double)+sig_e

    #n_boot=2

    #for l in xrange(n_boot):
    e1=gam1.copy()
    e2=gam2.copy()

    number_density=smooth_pbl.num_dens(x1,y1)
    num_dens=number_density/sum(number_density/n)

    #hsm=(2.6/numpy.sqrt(n))/num_dens**0.5+numpy.zeros(n,dtype=double)+0.013
    #hsm=(1.6/numpy.sqrt(n))/num_dens**0.5+numpy.zeros(n,dtype=double)+0.008   
    hsm=0.073+numpy.zeros(n,dtype=double)
    hsm0=numpy.mean(hsm)
    W=smooth_pbl.weight(x1,y1,hsm)

    e1_sm=dot(W,e1)
    e2_sm=dot(W,e2)
    #h=0.65/numpy.sqrt(n)+numpy.zeros(n,dtype=double) #/num_dens**0.5+numpy.zeros(n,dtype=double)
    h=0.67/numpy.sqrt(n)+numpy.zeros(n,dtype=double) #/num_dens**0.5+numpy.zeros(n,dtype=double)
    hsm0=numpy.mean(hsm)
    sigma=numpy.zeros(n,dtype=double)+sig_e

    kmat=smooth_pbl.create_matrix(x1,y1,0,h)
    g1mat=smooth_pbl.create_matrix(x1,y1,1,h)
    g2mat=smooth_pbl.create_matrix(x1,y1,2,h)

    #Initial Conditions
    #te=0.005
    #r=((x1-x_peak)**2+(y1-y_peak)**2)**0.5
    #psi=te*r

    # kappa=dot(kmat,psi)
    # nx=60
    # kap_grid=smooth_pbl.grid_field_gaussain(x1,y1,kappa,nx,nx)
    
    C=numpy.dot(W,numpy.dot(W,numpy.diag(sigma*sigma)).T)
    
    U,s,Vh=scipy.linalg.svd(C)
    eig1=s
    alpha=5.e-5*hsm0*hsm0
    e1=e1_sm.copy()
    e2=e2_sm.copy()
    s=eig1/(eig1**2+alpha**2)
    psi=numpy.zeros(n,dtype=double)
    
    C_inv=numpy.dot(numpy.dot(numpy.transpose(Vh),numpy.diag(s)),numpy.transpose(U))
    zw=(1.4-0.165)/1.4
    chi_prevmat=smooth_pbl.chisq(C_inv,psi,g1mat,g2mat,kmat,e1,e2,zw)
    chi_prev=sum(chi_prevmat)

    eps=0.1
    del_chi=1000
    n_iter=10
    eps=0.01
    
    while (del_chi>eps):
        L1,L2,dpsi=smooth_pbl.create_psi(psi,kmat,g1mat,g2mat,C_inv,e1,e2,zw)
        psi+=dpsi
        chi2mat=smooth_pbl.chisq(C_inv,psi,g1mat,g2mat,kmat,e1,e2,zw)
        chi2=sum(chi2mat)
        print "iter"
        del_chi=(chi_prev-chi2)
        chi_prev=chi2
                     
            
    kappa=numpy.dot(kmat,psi)
    nx=60
    kap_grid=smooth_pbl.grid_field_gaussian(x1,y1,kappa,nx,nx)
    gamma1=dot(g1mat,psi)
    gamma2=dot(g2mat,psi)
    C_kap=smooth_pbl.cov_kap(g1mat,g2mat,kmat,C_inv,W,sig_e,gamma1,gamma2)
    err=numpy.zeros(n,dtype=double)
    for i in xrange(n):
        err[i]=C_kap[i][i]
    
    eig1=s.copy()
    
    s1=eig1/(eig1**2+alpha**2)
    C_kap_inv=numpy.dot(numpy.dot(numpy.transpose(Vh),numpy.diag(s1)),numpy.transpose(U))
    
    kap_ave=sum(dot(C_kap_inv,kappa))/sum(C_kap_inv)
    id=where(s<5.e-8)
    s0=1/s
    s0[id]=5.e-8

    err=numpy.zeros(n,dtype=double)
    for i in xrange(n):
        err[i]=C_kap[i][i]
        
    kappa_prime=kap_ave+dot(U.T,(kappa-kap_ave))
    
    xc=sum(x1*kappa/err)/sum(kappa/err)
    yc=sum(y1*kappa/err)/sum(kappa/err)

    r2=(x1-xc)**2+(y1-yc)**2
    sigma=0.5
    w=exp(-r2/(2*sigma**2))
    
    w=w/sum(w)

    xsq=sum((x1-xc)**2*(dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)/sum((dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)
    ysq=sum((y1-yc)**2*(dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)/sum((dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)
    
    xy=sum((x1-xc)*(y1-yc)*(dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)/sum((dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)
                
    x4=sum((x1-xc)**4*(dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)/sum((dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)
    y4=sum((x1-xc)**4*(dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)/sum((dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)
    xy2=sum((x1-xc)**2*(y1-yc)**2*(dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)/sum((dot(U,(kappa_prime-kap_ave))+kap_ave)*w*s0/num_dens)
    
    print xsq,ysq,((x4-xsq**2)/n)**0.5,((y4-ysq**2)/n)**0.5
    
    filename="kappa_1689.dat"
    e1=(xsq-ysq)/(xsq+ysq+(xsq*ysq-xy**2)**2)
    e2=xy/(xsq+ysq+(xsq*ysq-xy**2)**2)
    print "e1= ",e1,"e2= ",e2
    
    f=open(filename,mode='w')
    for i in xrange(n):
        f.write('%g %g %g %g %g %g %g %g %g %g \n'%(ra[i],dec[i],x1[i],y1[i],kappa[i],psi[i],err[i],num_dens[i],gam1[i],gam2[i]))

    f.close()
    

    
    matshow(kap_grid)
    colorbar()
