import numpy

import matplotlib.pyplot as plt

from math import*

from pylab import*
import scipy.linalg.basic
import smooth_pbl

from numpy import dot
from flipper import*

if __name__=="__main__":
    
    #ra,dec,g1,g2,g1w,g2w,w,i,z=numpy.loadtxt('subaru_red_photoz.cat',unpack=True)
    ra,dec,g1,g2,z=load('Subaru_HST_fullsky_complete.dat',unpack=True)

    #id_weak,ra,dec,x,y,et,er,ee,e10,e20,de1,de2,e,de,ell,theta,sexe,elong,dist,fwhm,z,zbmin,zbmax,tb,odds,zml,tml,chisq,g,dg,r,dr,i,di,zband,dzband=numpy.loadtxt('full.cat',unpack=True)    

    ramax=numpy.max(ra)
    decmin=numpy.min(dec)
    decmean=numpy.mean(dec)
    x=(ramax-ra)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y=(dec-decmin)*3600
   
    ra_peak=197.875
    dec_peak=-1.3417

    ra_peak1=197.883
    dec_peak1=-1.3292
  
    x_peak=(ramax-ra_peak)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y_peak=(dec_peak-decmin)*3600

    x_peak1=(ramax-ra_peak1)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y_peak1=(dec_peak1-decmin)*3600

    rsq=(x-x_peak)**2+(y-y_peak)**2

    delta=240
    id=numpy.where((x>(x_peak-delta))&(x<(x_peak+delta))&(y>(y_peak-delta))&(y<(y_peak+delta)))
    ra=ra[id[0]]
    dec=dec[id[0]]
    x1=x[id[0]]
    y1=y[id[0]]
    e1=g1[id[0]]
    e2=g2[id[0]]

    id0=where((e1**2+e2**2)>1.)
    id1=where((e1**2+e2**2)<1.)
    print x1.shape
    xmin=min(x1)
    ymin=min(y1)
    n=x1.shape[0]
    number_density=smooth_pbl.num_dens(x1,y1)
    num_dens=number_density/sum(number_density/n)
   
    xmin=min(x1)
    ymin=min(y1)
    x1=(x1-xmin)/(2*delta)
    y1=(y1-ymin)/(2*delta)

    #nx=30
    #n0,e1_grid=smooth_pbl.grid_field(x1,y1,e1,nx,nx)
    #n0,e2_grid=smooth_pbl.grid_field(x1,y1,e2,nx,nx)

    #smooth_pbl.shear_map(e1_grid,e2_grid,nx)

    #utils.saveAndShow()
    #sys.exit()
    x_peak=(x_peak-xmin)/(2*delta)
    y_peak=(y_peak-ymin)/(2*delta)

    x_peak1=(x_peak1-xmin)/(2*delta)
    y_peak1=(y_peak1-ymin)/(2*delta)
   
    print x_peak,y_peak

    r=((x1-x_peak)**2+(y1-y_peak)**2)**0.5

    width=0.5

    hsm=(1./n**0.5)/num_dens**0.5+numpy.zeros(n,dtype=double)+0.03 
    #hsm=0.08*exp(r**2/(2*width**2))+numpy.zeros(n,dtype=double)
    hsm0=numpy.mean(hsm)
    W=smooth_pbl.weight(x1,y1,hsm)
    
    e1_sm=dot(W,e1)
    e2_sm=dot(W,e2)

    id,ra_s,dec_s,dum,dum,dum,zs0,dum=numpy.loadtxt('a1689_mul.cat',unpack=True)
    id_new,ra_new,dec_new=numpy.loadtxt('table.dat',unpack=True)

    id00=floor(id)
    id_new0=floor(id_new)
    n_strong=id_new.shape[0]
    zs=numpy.zeros(n_strong,dtype=double)
       
    for i in xrange(n_strong):
        idx=where(id_new0[i]==id00)
        if (len(idx[0])!=0):
            zs[i]=zs0[idx[0][0]]
        else:
            zs[i]=2.5
            
    x_st=(ramax-ra_new)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y_st=(dec_new-decmin)*3600
    x_st=(x_st-xmin)/(2*delta)
    y_st=(y_st-ymin)/(2*delta)
    id=id_new.copy()
    ra=numpy.append(ra_new,ra)
    dec=numpy.append(dec_new,dec)
    xstmin=min(x_st)
    xstmax=max(x_st)

    ystmin=min(y_st)
    ystmax=max(y_st)

    delx=xstmax-xstmin
    dely=ystmax-ystmin

    n_strong=x_st.shape[0]
    n_weak=x1.shape[0]
    x1=numpy.append(x_st,x1)
    y1=numpy.append(y_st,y1)

    ra=numpy.append(ra_s,ra)
    dec=numpy.append(dec_s,dec)

    n=x1.shape[0]

    id_image=numpy.zeros(n,dtype=double)
    id_image[0:n_strong]=id
    
    rsq=(x1-x_peak)**2+(y1-y_peak)**2
    h0=exp(rsq/(2))

    sig_e=0.3
    
    sigma=(numpy.zeros(n_weak,dtype=double)+sig_e)
    sigma[id0]=0.
    
    number_density=smooth_pbl.num_dens(x1,y1)
    num_dens=number_density/sum(number_density/n)

    #h=(0.5/(n+n_new)**0.5)/num_dens**0.5+numpy.zeros((n+n_new),dtype=double)+0.01
    h=(0.65/n**0.5)/num_dens**0.5+numpy.zeros((n),dtype=double)+0.03
    hsm0=numpy.mean(hsm)

    strong_weight=000.
    
    kmat=smooth_pbl.create_matrix(x1,y1,0,h)
    g1mat=smooth_pbl.create_matrix(x1,y1,1,h)
    g2mat=smooth_pbl.create_matrix(x1,y1,2,h)
    a1mat=smooth_pbl.create_matrix(x1,y1,3,h)
    a2mat=smooth_pbl.create_matrix(x1,y1,4,h)
    C=numpy.dot(W,numpy.dot(W,numpy.diag(sigma*sigma)).T)
    
    U,s,Vh=scipy.linalg.svd(C)
    eig1=s
    alpha=5.e-2*hsm0*hsm0
    e1=numpy.append(numpy.zeros(n_strong,dtype=double),e1_sm)
    e2=numpy.append(numpy.zeros(n_strong,dtype=double),e2_sm)
    
    #add the artificial data

    s=eig1/(eig1**2+alpha**2)

    #setting up an initial condition
    r1=((x1-x_peak)**2+(y1-y_peak)**2)**0.5
    r2=((x1-x_peak1)**2+(y1-y_peak1)**2)**0.5

    te1=0.076
    te2=0.048
    #psi=te1*r1
    #psi=te2*r2+te1*r1
    psi=numpy.zeros(n,dtype=double)
    
    kappa0=dot(kmat,psi)
    nx=120
    kap_grid0=smooth_pbl.grid_field_gaussian(x1,y1,kappa0,nx,nx)
    #sys.exit()
    C_invhat=numpy.dot(numpy.dot(numpy.transpose(Vh),numpy.diag(s)),numpy.transpose(U))
    C_inv=numpy.zeros((n,n),dtype=double)
    C_inv[n_strong:n,n_strong:n]=C_invhat
    
    zw=numpy.append((zs-0.1832)/zs,(1.4-0.1832)/1.4+numpy.zeros(n_weak,dtype=double))
    delx,dely,chi_prevmat=smooth_pbl.chisq(x1,y1,id_image,n_strong,n_weak,a1mat,a2mat,strong_weight,C_inv,psi,g1mat,g2mat,kmat,e1,e2,zw)
    chi_prev=sum(chi_prevmat)
    print "chi_start= ",chi_prev
    eps=0.1
    del_chi=100
    n_iter=10
    eps=0.01
    
    #while (del_chi>eps):
    for i in xrange(1):
        C_kap,dpsi=smooth_pbl.create_psi(x1,y1,id_image,n_strong,n_weak,a1mat,a2mat,strong_weight,psi,kmat,g1mat,g2mat,C_inv,e1,e2,zw,sig_e,W)
        psi+=dpsi
        delta_x,delta_y,chi2=smooth_pbl.chisq(x1,y1,id_image,n_strong,n_weak,a1mat,a2mat,strong_weight,C_inv,psi,g1mat,g2mat,kmat,e1,e2,zw)
        print "chi= ",chi2
        print "iter"
        del_chi=(chi_prev-chi2)
        chi_prev=chi2
    
    kappa=numpy.dot(kmat,psi)*zw
    nx=240
    kap_grid=smooth_pbl.grid_field_gaussian(x1,y1,kappa,nx,nx)
    gamma1=dot(g1mat,psi)
    gamma2=dot(g2mat,psi)
    #C_kap=smooth_pbl.cov_kap(n_strong,g1mat,g2mat,kmat,C_inv,W,sig_e,gamma1,gamma2)
    err=numpy.zeros(n,dtype=double)

    for i in xrange(n):
        err[i]=C_kap[i][i]

    cosphi=(x1-x_peak)/r1
    sinphi=(y1-y_peak)/r1
    cos2phi=(2*cosphi**2-1)
    sin2phi=2*cosphi*sinphi
    cos3phi=4*cosphi**3-3*cosphi
    sin3phi=3*sinphi-4*sinphi**3
    cos4phi=8*cosphi**4-8*cosphi**2+1
    sin4phi=4*sinphi*cosphi-8*sinphi**3*cosphi
    #R=500h^{-1} kpc
    coverH0=3000
    Dl=coverH0*0.185
    Rmax=((0.5/Dl)*206265)/(2*delta)
    id=where(r1>Rmax)
    Rmax=0.5
    w=1+numpy.zeros(n,dtype=double)#exp(-r1**2/(2*Rmax**2))

    w[id]=0
    
    a0=sum(kappa*w/(err*num_dens))/sum(w/(err*num_dens))
    a2=sum(kappa*r1**2*cos2phi*w/(err*num_dens))/sum(w/(err*num_dens))
    b2=sum(kappa*r1**2*sin2phi*w/(err*num_dens))/sum(w/(err*num_dens))
    a3=sum(kappa*r1**3*cos3phi*w/(err*num_dens))/sum(w/(err*num_dens))
    b3=sum(kappa*r1**3*sin3phi*w/(err*num_dens))/sum(w/(err*num_dens))
    a4=sum(kappa*r1**4*cos4phi*w/(err*num_dens))/sum(w/(err*num_dens))
    b4=sum(kappa*r1**4*sin4phi*w/(err*num_dens))/sum(w/(err*num_dens))
    
    P0=(a0*log(Rmax))**2
    P2=1./(2*2**2*Rmax**4) *(a2**2+b2**2)
    P3=1./(2*3**2*Rmax**6) *(a3**2+b3**2)
    P4=1./(2*4**2*Rmax**8) *(a4**2+b4**2)
    print "P2/P0= ",P2/P0,"P3/P0= ",P3/P0,"P4/P0= ",P4/P0
    '''
    U,s1,Vh=scipy.linalg.svd(C_kap)
    eig1=s1.copy()
    s1=eig1/(eig1**2+alpha**2)
    C_kap_inv=numpy.dot(numpy.dot(numpy.transpose(Vh),numpy.diag(s1)),numpy.transpose(U))
    
    kap_ave=sum(dot(C_kap_inv,kappa))/sum(C_kap_inv)
    id=where(s<5.e-8)
    s0=1/s1
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
    
    
    e1=(xsq-ysq)/(xsq+ysq+(xsq*ysq-xy**2)**2)
    e2=xy/(xsq+ysq+(xsq*ysq-xy**2)**2)
    print "e1= ",e1,"e2= ",e2
    '''
    alpha1=zw*dot(a1mat,psi)
    alpha2=zw*dot(a2mat,psi)
    filename="kappa_1689.dat"
    f=open(filename,mode='w')
    for i in xrange(n):
        f.write('%g %g %g %g %g %g %g %g %g %g \n'%(ra[i],dec[i],x1[i],y1[i],kappa[i],psi[i],err[i],num_dens[i],alpha1[i],alpha2[i]))

    f.close()
    

    
    matshow(kap_grid)
    colorbar()
    #plot(x1[0:n_strong]*nx,y1[0:n_strong]*nx,'o')
    utils.saveAndShow()
