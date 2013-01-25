import numpy

import matplotlib.pyplot as plt

from math import*

from pylab import*

import scipy.linalg.basic

import math

from numpy import dot

def weight(x,y,hsm):
    
    n0=x.shape
    n=n0[0]
    w_tot=numpy.zeros(n,dtype=double)
    W=numpy.zeros((n,n),dtype=double)
    for i in xrange(n):
        dx=x-x[i]
        #id=numpy.where(dx>0.5)
        #dx[id]=dx[id]-1.
        #id=numpy.where(dx<-0.5)
        #dx[id]=dx[id]+1.
        dy=y-y[i]
        #id=numpy.where(dy>0.5)
        #dy[id]=dy[id]-1.
        #id=numpy.where(dy<-0.5)
        #dy[id]=dy[id]+1.
        drsq=(dx**2+dy**2)/(hsm[i]*hsm[i])
        dr=drsq**0.5
        id1=numpy.where(dr<3)
        
        W[i,id1]=exp(-drsq[id1]/2)
        
        w_tot[i]=sum(W[i,:])
        W[i,:]=W[i,:]/w_tot[i]
    #print "w_tot=", w_tot[20]
    return W

def create_matrix(x,y,n0,h):
#(x,y) positions of images
#n0=0 for kappa, 1 for gamma1, 2 for gamma2
#h :PBL_length scale: an array of size x.shape
#This function returns matrices(kmat or g1mat or g2mat) such that kappa=kmat.psi
    n=x.shape[0]
    pars=9
    kmat=numpy.zeros((n,n),dtype=double)
    g1mat=numpy.zeros((n,n),dtype=double)
    g2mat=numpy.zeros((n,n),dtype=double)

    D=numpy.zeros((pars,n,n),dtype=double)
    W=numpy.zeros(n,dtype=double)
    dx=numpy.zeros(n,dtype=double)
    dy=numpy.zeros(n,dtype=double)
    X_arr=numpy.zeros((pars,n),dtype=double)
    D_out=numpy.zeros((pars,n),dtype=double)
    A_arr=numpy.zeros((pars,pars),dtype=double)
    for i in xrange(n):
        dx=x-x[i]
        #id=numpy.where(dx>0.5)
        #dx[id]=dx[id]-1.
        #id=numpy.where(dx<-0.5)
        #dx[id]=dx[id]+1.
        dy=y-y[i]
        #id=numpy.where(dy>0.5)
        #dy[id]=dy[id]-1.
        #id=numpy.where(dy<-0.5)
        #dy[id]=dy[id]+1.
        drsq=(dx**2+dy**2)
        W=exp(-drsq/(2*h[i]*h[i]))
        
        X_arr[0,:]=dx
        X_arr[1,:]=dy
        X_arr[2,:]=0.5*dx*dx
        X_arr[3,:]=dx*dy
        X_arr[4,:]=0.5*dy*dy
        X_arr[5,:] = 1./6. *dx**3
        X_arr[6,:] = 1./2. *dx*dx*dy
        X_arr[7,:] = 1./2. *dx*dy*dy
        X_arr[8,:] = 1./6. *dy**3

        for alpha in xrange(pars):
            for beta in xrange(pars):
                A_arr[alpha,beta]=sum(X_arr[alpha,:]*X_arr[beta,:]*W)
                
                
        D_out=numpy.dot(inv(A_arr),X_arr)

        for alpha in xrange(pars):
            D[alpha,i,:]=D_out[alpha,:]*W
            tot=sum(D[alpha,i,:])
            D[alpha,i,i]-=tot

#create kmat,g1mat,g2mat
    a1mat=D[0,:,:]
    a2mat=D[1,:,:]
    kmat=(D[2,:,:]+D[4,:,:])/2
    g1mat=(D[2,:,:]-D[4,:,:])/2
    g2mat=D[3,:,:]

    if (n0==0):
        return kmat
    if (n0==1):
        return g1mat
    if (n0==2):
        return g2mat
    if (n0==3):
        return a1mat
    if (n0==4):
        return a2mat
        

def grid_field(x,y,field,nx,ny):
    i=numpy.floor(x*nx)
    j=numpy.floor(y*ny)
    n=x.shape[0]
    print max(x), "max"
    numcount=numpy.zeros((nx,ny),dtype=int)
    data=numpy.zeros((nx,ny),dtype=double)
    for k in xrange(n):
        data[i[k],j[k]]+=field[k]
        numcount[i[k],j[k]]+=1
    id=where(numcount>0)
    data[id]=data[id]/numcount[id]

    return(numcount,data)

def create_psi(x,y,id_image,n_strong,n_weak,a1mat,a2mat,strong_weight,psi,kmat,g1mat,g2mat,C_inv,e1_sm,e2_sm,zw,sig_e,W):
    n=x.shape[0]
    gamma1=dot(g1mat,psi)*zw
    gamma2=dot(g2mat,psi)*zw
    kappa=dot(kmat,psi)*zw
    alphax=dot(a1mat,psi)*zw
    alphay=dot(a2mat,psi)*zw

    id=where(((1-kappa)**2-gamma1**2-gamma2**2)<0)

    e1hat=numpy.zeros((n,n),dtype=double)
    e2hat=numpy.zeros((n,n),dtype=double)
    de1dpsi=numpy.zeros((n,n),dtype=double)
    de2dpsi=numpy.zeros((n,n),dtype=double)
    
    de1dgamma1=zw/(1-kappa)
    de1dgamma2=numpy.zeros(n,dtype=double)
    de1dkappa=zw*gamma1/(1-kappa)**2
    de1dkappa[id]=-gamma1[id]/(gamma1[id]**2+gamma2[id]**2)
    de1dgamma1[id]=(kappa[id]-1)*(gamma2[id]**2-gamma1[id]**2)/(gamma1[id]**2+gamma2[id]**2)**2
    de1dgamma2[id]=-2*(1-kappa[id])*gamma1[id]*gamma2[id]/(gamma1[id]**2+gamma2[id]**2)**2

    de2dgamma1=numpy.zeros(n,dtype=double)
    de2dgamma2=zw/(1-kappa)
    de2dkappa=zw*gamma2/(1-kappa)**2
    de2dkappa[id]=-gamma2[id]/(gamma1[id]**2+gamma2[id]**2)**2
    de2dgamma1[id]=-2*(1-kappa[id])*gamma1[id]*gamma2[id]/(gamma1[id]**2+gamma2[id]**2)**2
    de2dgamma2[id]=(kappa[id]-1)*(gamma1[id]**2-gamma2[id]**2)/(gamma1[id]**2+gamma2[id]**2)**2

    e1mod=gamma1/(1-kappa)
    e2mod=gamma2/(1-kappa)
    
    e1mod[id]=gamma1[id]*(1-kappa[id])/(gamma1[id]**2+gamma2[id]**2)
    e2mod[id]=gamma2[id]*(1-kappa[id])/(gamma1[id]**2+gamma2[id]**2)
    delta_e1=e1_sm-e1mod
    delta_e2=e2_sm-e2mod
    
    delx=array([])
    dely=array([])
    id1=array([])
    id2=array([])
    n_id=int(max(id_image))
    id_strong=numpy.zeros(n,dtype=int)

    for i in xrange(n_strong):
        id_strong[i]=int(id_image[i])

    for i in xrange(n_id):
        idx=where((id_strong)==(i+1))
        dx,dy,id01,id02=SL_pairs(x,y,idx)
        delx=numpy.append(delx,dx)
        dely=numpy.append(dely,dy)
        id1=numpy.append(id1,id01)
        id2=numpy.append(id2,id02)

    npairs=delx.shape[0]
    print "npairs=", npairs
    A_strongx=numpy.zeros((npairs,n),dtype=double)
    A_strongy=numpy.zeros((npairs,n),dtype=double)
    for i in xrange(npairs):
        dalphaxdalphax_a=zw
        dalphaxdalphax_b=-zw
        dalphaydalphay_a=zw
        dalphaydalphay_b=-zw
        A_strongx[i,:]=a1mat[id1[i],:]*dalphaxdalphax_a+a1mat[id2[i],:]*dalphaxdalphax_b
        A_strongy[i,:]=a2mat[id1[i],:]*dalphaydalphay_a+a2mat[id2[i],:]*dalphaydalphay_b

    A1=numpy.zeros((n,n),dtype=double)
    A2=numpy.zeros((n,n),dtype=double)

    for i in xrange(n):
        A1[i,:]=g1mat[i,:]*de1dgamma1[i]+g2mat[i,:]*de1dgamma2[i]+kmat[i,:]*de1dkappa[i]
        A2[i,:]=g1mat[i,:]*de2dgamma1[i]+g2mat[i,:]*de2dgamma2[i]+kmat[i,:]*de2dkappa[i]

    delta_delx=numpy.zeros(npairs,dtype=double)
    delta_dely=numpy.zeros(npairs,dtype=double)
    for i in xrange(npairs):
        delta_delx[i]=(delx[i]-(alphax[id1[i]]-alphax[id2[i]]))
        delta_dely[i]=(dely[i]-(alphax[id1[i]]-alphax[id2[i]]))

    delta_kappa=-kappa
    #print delta_kappa
    #kap_w=numpy.zeros(n,dtype=double) 
    #id=where(delta_kappa<0)
    kap_w=0.1
    Y=dot(dot(A1.T,C_inv),delta_e1)+dot(dot(A2.T,C_inv),delta_e2)+zw*kap_w*dot(kmat,delta_kappa)+strong_weight*dot(A_strongx.T,delta_delx)+strong_weight*dot(A_strongy.T,delta_dely)
    M=dot(dot(A1.T,C_inv),A1)+dot(dot(A2.T,C_inv),A2)+zw*zw*kap_w*dot(kmat.T,kmat)+strong_weight*dot(A_strongx.T,A_strongx)+strong_weight*dot(A_strongy.T,A_strongy)
    M_inv=scipy.linalg.pinv(M)
    delpsi=numpy.dot(M_inv,Y)
    psi=psi+0.5*delpsi
    #Calculate Covariance
    V_w=dot(M_inv,(dot(A1.T,C_inv)+dot(A2.T,C_inv)))
    V_s=dot(M_inv,strong_weight*(dot(A_strongx.T,delta_delx)+dot(A_strongy.T,delta_dely)))
    gamma1=dot(g1mat,psi)*zw
    gamma2=dot(g2mat,psi)*zw
    kappa=dot(kmat,psi)*zw
    e1mod=gamma1/(1-kappa)
    e2mod=gamma2/(1-kappa)
    C_e=Cov_e(C_inv,sig_e,e1mod,e2mod,W,n_strong)
    cov_kap=dot(dot(V_w,C_e),V_w.T)+strong_weight*dot(V_s,V_s.T)
        
    return(cov_kap,delpsi)
    
def chisq(x,y,id_image,n_strong,n_weak,a1mat,a2mat,strong_weight,C_inv,psi,g1mat,g2mat,kmat,e1_sm,e2_sm,zw):
    n=x.shape[0]
    kappa=dot(kmat,psi)*zw
    alphax=zw*dot(a1mat,psi)
    alphay=zw*dot(a2mat,psi)
    gamma1=zw*dot(g1mat,psi)
    gamma2=zw*dot(g2mat,psi)

    id=where(((1-kappa)**2-gamma1**2-gamma2**2)<0)
    e1mod=gamma1/(1-kappa)
    e2mod=gamma2/(1-kappa)
    e1mod[id]=gamma1[id]*(1-kappa[id])/(gamma1[id]**2+gamma2[id]**2)
    e2mod[id]=gamma2[id]*(1-kappa[id])/(gamma1[id]**2+gamma2[id]**2)
    delta_e1=e1_sm-e1mod
    delta_e2=e2_sm-e2mod

    del_e1=(e1_sm-e1mod)
    del_e2=(e2_sm-e2mod)

    chi2mat=numpy.zeros((n,n),dtype=double)
    chi2=0
    delx=array([])
    dely=array([])
    id1=array([])
    id2=array([])
    n_id=int(max(id_image))
    id_strong=numpy.zeros(n,dtype=int)
    
    for i in xrange(n_strong):
        id_strong[i]=int(id_image[i])

    #print id_strong
    for i in xrange(n_id):
        idx=where((id_strong)==(i+1))
        dx,dy,id01,id02=SL_pairs(x,y,idx)
        delx=numpy.append(delx,dx)
        dely=numpy.append(dely,dy)
        id1=numpy.append(id1,id01)
        id2=numpy.append(id2,id02)

    npairs=delx.shape[0]
    delta_x=numpy.zeros(npairs,dtype=double)
    delta_y=numpy.zeros(npairs,dtype=double)
    
    for i in xrange(npairs):
        delta_x[i]=delx[i]-(alphax[id1[i]]-alphax[id2[i]])
        delta_y[i]=dely[i]-(alphay[id1[i]]-alphay[id2[i]])
        chi2+=sum((delta_x**2+delta_y**2)*strong_weight)
    print "chi2",chi2
    for i in xrange(n):
        for j in xrange(n):
            chi2mat[i][j]=del_e1[i]*C_inv[i][j]*del_e1[j]+del_e2[i]*C_inv[i][j]*del_e2[j]

    print mean(numpy.sqrt((delta_x*480)**2)),mean(numpy.sqrt((delta_y*480)**2))
    chi2+=sum(chi2mat)

    
    return(delta_x,delta_y,chi2)
 
def num_dens(x,y): 
    n=x.shape[0]
    num_d=numpy.zeros((n),dtype=double)
    for i in xrange(n):
        dx=x-x[i]
        id=numpy.where(dx>0.5)
        dx[id]=dx[id]-1.
        id=numpy.where(dx<-0.5)
        dx[id]=dx[id]+1.
        dy=y-y[i]
        id=numpy.where(dy>0.5)
        dy[id]=dy[id]-1.
        id=numpy.where(dy<-0.5)
        dy[id]=dy[id]+1.
        r2=(dx**2+dy**2)
        id=r2.argsort()

        num_d[i]=1./r2[id[4]]
        
    rho0=sum(num_d)/n
    return(num_d)

def radial(x,y,i0):
    dx=x-x[i0]
    id=numpy.where(dx>0.5)
    dx[id]=dx[id]-1.
    id=numpy.where(dx<-0.5)
    dx[id]=dx[id]+1.
    dy=y-y[i0]
    id=numpy.where(dy>0.5)
    dy[id]=dy[id]-1.
    id=numpy.where(dy<-0.5)
    dy[id]=dy[id]+1.
    r=numpy.sqrt(dx**2+dy**2)
    return(r)

def calc_ellip(g1,g2,es1,es2):
    gsq=g1**2+g2**2
    id=where(gsq>1)
    A=es1+g1
    B=es2+g2
    C=1+g1*es1+g2*es2
    D=g1*es2-g2*es1
    '''if (id.shape>0):
        A[id]=1+g1*es1+g2*es2
        B[id]=g2*es1-g1*es2
        C[id]=es1+g1
        D[id]=-es2-g2
        '''
    e1=(A*C+B*D)/(C*C+D*D)
    e2=(B*C-A*D)/(C*C+D*D)
    return e1,e2

def grid_field_gaussian(x,y,field,nx,ny):
    n=x.shape[0]
    xmin=numpy.min(x)
    xmax=numpy.max(x)
    ymin=numpy.min(y)
    ymax=numpy.max(y)

    dx=(xmax-xmin)/nx
    dy=(ymax-ymin)/ny

    h=sqrt((xmax-xmin)*(ymax-ymin)/n)
    field_grid=numpy.zeros((nx,ny),dtype=double)
    for i in xrange(nx):
        xc=xmin+(i+0.5)*dx
        for j in xrange(ny):
            yc=ymin+(j+0.5)*dy
            W=numpy.exp((-(x-xc)**2-(y-yc)**2)/(2*h*h))
            W=W/numpy.sum(W)
            #print W.shape,field.shape
            field_grid[i,j]=sum(W*field)


    return(field_grid)


def redshift_weight(zs,zl):
#Right now I am doing cz/H_0, later I will use the right cosmology and calculate angular diameter distance

    return((zs-zl)/zs)



def shear_map(e1,e2,nx):
    
    n=e1.shape[0]
    field=numpy.zeros((nx,nx),dtype=double)
    x=double((numpy.indices([nx,nx])[0]))+0.5
    y=double((numpy.indices([nx,nx])[1]))+0.5

    fact=5.

    #matshow(field)
    eps=1.e-9
    
    etot=numpy.sqrt(e1**2+e2**2)
    
    phi=numpy.zeros((nx,nx),dtype=double)
    
    for l in xrange(nx):
        for k in xrange(nx):
            if (etot[l,k]>0):
                phi[l,k]=(math.acos(e1[l,k]/etot[l,k])*e2[l,k]/numpy.abs(e2[l,k]))/2.

    fct=5

    u=fct*etot*numpy.cos(phi)
    v=fct*etot*numpy.sin(phi)
    
    #u=1+numpy.zeros((nx,nx),dtype=double)
    #v=0.5+numpy.zeros((nx,nx),dtype=double)

    width=1
    Q=quiver(x,y,u,v,pivot='middle',units='width',headlength=0,headwidth=0,color='k')
    #qk = quiverkey(Q, 0.5, 0.92, 2, labelpos='W',
                   #fontproperties={'weight': 'bold'})
    #l,r,b,t = axis()
    #dx, dy = r-l, t-b
    #axis([l-0.05*dx, r+0.05*dx, b-0.05*dy, t+0.05*dy])


def Cov_e(C_inv,sig_e,e1,e2,W,n_strong):
    n=e1.shape[0]
    n_weak=n-n_strong
    e1_sm=dot(W.T,e1[n_strong:n])
    e2_sm=dot(W.T,e2[n_strong:n])
    
    C_e1=numpy.zeros((n_weak,n_weak),dtype=double)
    C_e2=numpy.zeros((n_weak,n_weak),dtype=double)

    for i in xrange(n_weak):
        for j in xrange(n_weak):
            C_e1[i,j]=(e1[i]-e1_sm[i])*(e1[j]-e1_sm[j])
            C_e2[i,j]=(e2[i]-e2_sm[i])*(e2[j]-e2_sm[j])

    
    
    C_ehat=C_e1+C_e2+2*dot(W,W.T)*sig_e**2
    C_e=numpy.zeros((n,n),dtype=double)
    C_e[n_strong:n,n_strong:n]=C_ehat
    
    return(C_e)
    
def cov_kap(n_strong,g1mat,g2mat,kmat,C_inv,W,sig_e,e1,e2):
    
    n=e1.shape[0]
    n_weak=n-n_strong
    M=dot(g1mat.T,dot(C_inv,g1mat))+dot(g2mat.T,dot(C_inv,g2mat))
    M_inv=scipy.linalg.pinv(M)
    #V1=dot(kmat,dot(M_inv,dot(g1mat.T,dot(C_inv,W))))+dot(kmat,dot(M_inv,dot(g2mat.T,dot(C_inv,W))))
    V1=dot(kmat,dot(M_inv,dot(g1mat.T,C_inv)))+dot(kmat,dot(M_inv,dot(g2mat.T,C_inv)))
    e1_sm=dot(W.T,e1[n_strong:n])
    e2_sm=dot(W.T,e2[n_strong:n])
    
    C_e1=numpy.zeros((n_weak,n_weak),dtype=double)
    C_e2=numpy.zeros((n_weak,n_weak),dtype=double)

    for i in xrange(n_weak):
        for j in xrange(n_weak):
            C_e1[i,j]=(e1[i]-e1_sm[i])*(e1[j]-e1_sm[j])
            C_e2[i,j]=(e2[i]-e2_sm[i])*(e2[j]-e2_sm[j])

    
    
    C_ehat=C_e1+C_e2+2*dot(W,W.T)*sig_e**2
    C_e=numpy.zeros((n,n),dtype=double)
    C_e[n_strong:n,n_strong:n]=C_ehat
    
    C_kap=dot(V1,dot(C_e,V1.T))

    return(C_kap)

def calc_moments(number_density,kappa,C_kap,field):

    U,s,Vh=scipy.linalg.svd(C_kap)
    alpha=1.e-4


    s=s/(s**2+alpha**2)
    C_kap_inv=numpy.dot(numpy.dot(numpy.transpose(Vh),numpy.diag(s)),numpy.transpose(U))

    moment=sum(dot(C_kap_inv.T,(kappa*field/number_density)))/sum(dot(C_kap_inv.T,kappa/number_density))

    return (moment)

def calc_measures(number_density,C_kap,kappa,x,y):

#This function calculates the centroid and moment of inertia for the mass map.
    n=x.shape[0]

    kap_ave=sum(dot(C_kap.T,(kappa*number_density)))/sum(dot(C_kap.T,number_density))
    kapsq_ave=sum(dot(C_kap.T,(kappa**2*number_density)))/sum(dot(C_kap.T,number_density))
    err_kap=(kapsq_ave-kap_ave**2)**0.5/n**0.5
    
    #calculating xc and yc
    xc=calc_moments(number_density,kappa,C_kap,x)
    xsq=calc_moments(number_density,kappa,C_kap,x**2)
    err_x=(xsq-xc**2)**0.5/n**0.5 

    yc=calc_moments(number_density,kappa,C_kap,y)
    ysq=calc_moments(number_density,kappa,C_kap,y**2)
    err_y=(ysq-yc**2)**0.5/n**0.5
        
    centroid=[xc,yc]
    error_centroid=[err_x,err_y]

    #calculating the four compoents of moment of inertia
    
    I_xx=xsq
    I_yy=ysq
    I_xy=calc_moments(number_density,kappa,C_kap,x*y)

    err_xx=(calc_moments(number_density,kappa,C_kap,x**4)-I_xx**2)**0.5/n**0.5
    err_yy=(calc_moments(number_density,kappa,C_kap,y**4)-I_yy**2)**0.5/n**0.5
    err_xy=(calc_moments(number_density,kappa,C_kap,y**2*x**2)-I_xy**2)**0.5/n**0.5

    I=[[I_xx,I_xy],[I_xy,I_yy]]
    err_I=[[err_xx,err_xy],[err_xy,err_yy]]

    
    return(kap_ave,err_kap,centroid,error_centroid,I,err_I)

    
        
def SL_pairs(x,y,idx):
    idx=idx[0]
    ns=idx.shape[0]
    
    npairs=(ns-1)*ns/2
    x1=numpy.zeros(npairs,dtype=double)
    x2=numpy.zeros(npairs,dtype=double)
    y1=numpy.zeros(npairs,dtype=double)
    y2=numpy.zeros(npairs,dtype=double)
    id1=numpy.zeros(npairs,dtype=int)
    id2=numpy.zeros(npairs,dtype=int)

    n0=0
    for i in xrange(ns-1):
        n=ns-1-i
        x1[n0:n+n0]=x[idx[i]]
        y1[n0:n+n0]=y[idx[i]]
        id1[n0:n+n0]=idx[i]
        #print x2[n0:n+n0].shape,x[idx[i+1:ns]].shape,i
        x2[n0:n+n0]=array([x[idx[i+1:ns]]])
        y2[n0:n+n0]=array([y[idx[i+1:ns]]])
        id2[n0:n+n0]=array([idx[i+1:ns]])
        n0+=n
        #print id1,id2,id1.shape,id2.shape

    delx=x1-x2
    dely=y1-y2
    return(delx,dely,id1,id2)

    
