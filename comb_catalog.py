import numpy
from math import*
from pylab import*
import scipy.linalg.basic
from flipper import*
import smooth_pbl


if __name__=="__main__":

    id_weak,ra_hst,dec_hst,x,y,et,er,ee,e10,e20,de1,de2,e,de,ell,theta,sexe,elong,dist,fwhm,z,zbmin,zbmax,tb,odds,zml,tml,chisq,g,dg,r,dr,i,di,zband,dzband=numpy.loadtxt('full.cat',unpack=True)
    ra,dec,g1,g2,g1w,g2w,w,i,z=numpy.loadtxt('subaru_red_photoz.cat',unpack=True) 
    
    decmean=mean(dec)
    ramax=max(ra)
    decmin=min(dec)
            

    x=(ramax-ra)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y=(dec-decmin)*3600

    x_hst=(ramax-ra_hst)*numpy.cos(decmean/(180/numpy.pi))*3600.
    y_hst=(dec_hst-decmin)*3600

    '''delta_x=max(x_hst)-min(x_hst)
    delta_y=max(y_hst)-min(y_hst)

    delta=max(array([delta_x,delta_y]))
    xmin=min(x_hst)
    ymin=min(y_hst)

    x_hst=(x_hst-xmin)/delta
    y_hst=(y_hst-ymin)/delta

    nx=16
    e1_grid=smooth_pbl.grid_field_gaussian(x_hst,y_hst,-e10,nx,nx)
    e2_grid=smooth_pbl.grid_field_gaussian(x_hst,y_hst,e20,nx,nx)

    smooth_pbl.shear_map(e1_grid,e2_grid,nx)
    utils.saveAndShow()
    sys.exit()
    '''
    n_hst=x_hst.shape[0]
    n_subaru=x.shape[0]
    
    f=open('Subaru_HST_fullsky_complete.dat',mode='w')

    for i in xrange(n_hst):
        f.write('%g %g %g %g %g \n'%(ra_hst[i],dec_hst[i],e10[i],e20[i],z[i]))

    for i in xrange(n_subaru):
        rsq=(x[i]-x_hst)**2+(y[i]-y_hst)**2
        id=where(rsq==min(rsq))
        if (rsq[id]>25):
            f.write('%g %g %g %g %g \n'%(ra[i],dec[i],g1w[i],g2w[i],z[i]))
            
    f.close()

    ra,dec,g1,g2,z=load('Subaru_HST_fullsky_complete.dat',unpack=True)
                                                
                                                    
    
