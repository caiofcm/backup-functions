# New test cubic splines
from numpy import *
from matplotlib import pyplot as plt


def spline3_coeff(ti, yi):
    """# http://www.lce.hut.fi/teaching/S-114.1100/lect_6.pdf pg 16"""
    h = empty(n)
    b = empty(n)
    for i in arange(0, n):
        h[i] = ti[i+1] - ti[i]
        b[i] = 1./h[i]*(yi[i+1] - yi[i])
        
    # Step 2:
    u = empty(n)
    v = empty(n)
    u[1] = 2.*(h[0] + h[1])
    v[1] = 6.*(b[1] - b[0])
    for i in arange(2, n):
        u[i] = 2.*(h[i]+h[i-1]) - h[i-1]**2/u[i-1]
        v[i] = 6.*(b[i]-b[i-1]) - h[i-1]*v[i-1]/u[i-1]
        
    # Step 3:
    z = empty(n + 1)
    z[n] = 0.
    z[0] = 0.
    for i in arange(n - 1, 0, -1):
        z[i] = (v[i] - h[i]*z[i+1])/u[i]
    return z
   
def spline3_eval(ti, y, z, tnow):
    """# http://www.lce.hut.fi/teaching/S-114.1100/lect_6.pdf pg 17"""
    for i in arange(n - 1, 0 - 1, -1):
        if (tnow - ti[i] >= 0.):
            break;
    h = ti[i+1] - ti[i]
    tmp = .5*(z[i]) + (tnow-ti[i])*(z[i+1]-z[i])/(6.*h)
    tmp = -(h/6.)*(z[i+1]+2.*z[i]) + (y[i+1]-y[i])/h + (tnow-ti[i])*tmp
    result = y[i] + (tnow - ti[i])*tmp
    return result
    
    
n = 10
ti = linspace(0., 10., n + 1)
yi = array([10., 9., 11., 10., 5., 4., 7., 8., 5., 9., 11.])
Nt = 100
tspan = linspace(0., 10., Nt)
    
z = spline3_coeff(ti, yi)
S = empty_like(tspan)


for i in arange(0, Nt):
    S[i] = spline3_eval(ti, yi, z, tspan[i])
            
plt.figure()
plt.plot(ti, yi, '.', ti, z, 'o', tspan, S, '-')
plt.show()



































