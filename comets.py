import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import yt
import pdb
import os
import cPickle as pickle
import copy

fn = 'jpl2.csv'
mu = 877.406516 # AU * km^2/s^2
G = yt.units.G
c = yt.units.c
msun = yt.units.msun
Moumuamua  = yt.YTQuantity(1.44e9,'kg')
rsun = yt.YTQuantity(7.0e10, 'cm')


def setval(val):
    if val == '' or val=='\n':
        return None
    else:
        try:
            return float(val)
        except:
            pdb.set_trace()

def matrixMult(A,x):
    ncols, nrows = np.shape(A)
    assert ncols==len(x)
    y = np.zeros(nrows)
    for i in range(nrows):
        y[i] = np.sum(A[i,:]*x[:])
    return y




class simulatedComet:
    def __init__(self,vx,vy,vz,b,theta):
        self.vGalactic = np.array([vx,vy,vz])
        self.b = b
        self.theta=theta
        self.rICRS, self.vICRS = computeSolarSystemCoords(vx,vy,vz,b,theta)
        semi,ecc,inc,omega,w= computeOrbitalElements(self.rICRS[0],self.rICRS[1],self.rICRS[2], self.vICRS[0], self.vICRS[1], self.vICRS[2])
        self.semi,self.ecc,self.inc,self.omega,self.w=(semi,ecc,inc*180/np.pi,omega*180/np.pi,w*180/np.pi) # convert angles to degrees
        self.peri = self.semi*(1-self.ecc)
        if self.omega<0:
            self.omega+=360
        if self.w<0:
            self.w+=360
        # one more time since this one is computed as a difference in angles that may have been outside of whatever range we were aiming for
        if self.w<0:
            self.w+=360

        assert self.inc>=0 and self.inc<=180
        assert self.omega>=0 and self.omega<=360
        assert self.w>=0 and self.w<=360


def drawComets(qmax, N, M, U, V, W, sigmau, sigmav, sigmaw):
    savefilename = 'simcomets_'+str(qmax).replace('.','p')+'_'+str(U).replace('.','p')+'_'+str(V).replace('.','p')+'_'+str(W).replace('.','p')+'_'+str(sigmau).replace('.','p')+'_'+str(sigmav).replace('.','p')+'_'+str(sigmaw).replace('.','p')+'_'+str(N)+'_'+str(M)+'.pickle'
    if os.path.isfile(savefilename):
        with open(savefilename,'rb') as theFile:
            cometList = pickle.load(theFile)
            return cometList
    vx0 = U
    vy0 = V
    vz0 = W
    # first draw a velocity from the assumed gaussian.
    # LSR velocity in the frame of the sun.
    #vx0 = -10.0
    #vy0 = -11.0
    #vz0 = -7.0
    # M dwarf velocity :shrug:

    # 'Oumuamua
    #vx0 = -11.457
    #vy0 = -22.395
    #vz0 = -7.746
    #vx0=0
    #vy0=0
    #vz0=0 # sanity check
    fac = 1.0
    #sigmax=20*fac # for a quick test
    #sigmay=20*fac
    #sigmaz=10*fac
    vx = np.random.normal(size=N)*sigmau+vx0
    vy = np.random.normal(size=N)*sigmav+vy0
    vz = np.random.normal(size=N)*sigmaw+vz0

    v = np.sqrt(vx*vx + vy*vy + vz*vz) # array-like plox
    #print "random velocitiies"
    #print vx, vy,vz, v
    w = np.pi*qmax*qmax * (1.0 + 2.0*mu/(qmax*v*v)) * v # the weight!

    nw = w/np.sum(w)
    nwc = np.cumsum(nw)
    unif = np.random.random(size=M)
    
    simCometList = []
    print "Drawing M comets",M,"from N velocity draws",N
    for j in range(M):
        k = np.argmin( np.abs(nwc-unif[j]) ) # select one of the velocities - select based on the weights.
        #pdb.set_trace()
        vThis = v[k]
        vxThis = vx[k]
        vyThis = vy[k]
        vzThis = vz[k]
        bmax = qmax * np.sqrt(1.0+ 2.0*mu/(qmax*vThis*vThis))
        b = np.sqrt(np.random.random())*bmax
        theta = np.random.random()*2.0*np.pi
        simCometList.append(simulatedComet(vxThis,vyThis,vzThis,b,theta))
        if j%10000==0:
            print "comet ",j,"/",M, "= ",float(j)/float(M) * 100 , "%"


    with open(savefilename,'wb') as theFile:
        pickle.dump(simCometList, theFile)


    return simCometList


def solarcollisionrate(sigmaU=30, sigmaV=18, sigmaW=18, U=10, V=11, W=7):
    #n0 = yt.YTQuantity( density, 'AU**-3')
    sigmax = yt.YTQuantity(sigmaU,'km/s')
    sigmay = yt.YTQuantity(sigmaV,'km/s')
    sigmaz = yt.YTQuantity(sigmaW,'km/s')
    #prefactor = np.pi* mu*mu * np.power(2.0*np.pi,-1.5) * n0* 1.0/(sigmax*sigmay*sigmaz)
    #chi = rsun *sigmax*sigmax / mu # should be dimensionless I hope
    # nondimensionalize velocities by scaling to sigmax.
    sy = sigmay/sigmax
    sz = sigmaz/sigmax
    solarMotion = yt.YTArray( [U, V, W], 'km/s') # solar motion relative to LSR quoted by Schoenrich and Aumer
    vx0 = solarMotion[0]/sigmax
    vy0 = solarMotion[1]/sigmax
    vz0 = solarMotion[2]/sigmax
    cutoffImpact = yt.YTQuantity( 0.01, 'pc')
    
    def thirdIntegrandA(vz,vy,vx):
        # vx vy and vz are dimensionless, scaled to sigmax.
        v = np.sqrt(vx*vx + vy*vy + vz*vz)
        if v<1.0e-10:
            v=1.0e-10
        ret = np.exp(-(vx-vx0)**2/(2.0)) * np.exp(-(vy-vy0)**2/(2.0*sy*sy)) * np.exp(-(vz-vz0)**2/(2.0*sz*sz)) * v
        if np.isfinite(ret):
            return ret
        else:
            pdb.set_trace()

    def secondIntegrandA(vy,vx):
        integral, error = scipy.integrate.quad( thirdIntegrandA, -10, 10, args=(vy,vx), limit=5000, limlst=5000, epsrel=0.01)
        if np.isfinite(integral):
            return integral
        else:
            pdb.set_trace()
    def firstIntegrandA(vx):
        integral, error = scipy.integrate.quad( secondIntegrandA, -10, 10, args=(vx), limit=5000, limlst=5000, epsrel=0.01)
        if np.isfinite(integral):
            return integral
        else:
            pdb.set_trace()
    integralA, error = scipy.integrate.quad( firstIntegrandA, -10, 10, limit=5000, limlst=5000, epsrel=0.01)
    print "integral,error: ",integralA, error
    vAvg = integralA*sigmax* 1.0*sigmax*sigmax*sigmax/np.power(2.0*np.pi, 1.5) * 1.0/(sigmax*sigmay*sigmaz)


    def thirdIntegrandB(vz,vy,vx):
        # vx vy and vz are dimensionless, scaled to sigmax.
        v = np.sqrt(vx*vx + vy*vy + vz*vz)
        if v<1.0e-10:
            v=1.0e-10
        ret = np.exp(-(vx-vx0)**2/(2.0)) * np.exp(-(vy-vy0)**2/(2.0*sy*sy)) * np.exp(-(vz-vz0)**2/(2.0*sz*sz)) / v
        if np.isfinite(ret):
            return ret
        else:
            pdb.set_trace()

    def secondIntegrandB(vy,vx):
        integral, error = scipy.integrate.quad( thirdIntegrandB, -10, 10, args=(vy,vx), limit=5000, limlst=5000, epsrel=0.01)
        if np.isfinite(integral):
            return integral
        else:
            pdb.set_trace()
    def firstIntegrandB(vx):
        integral, error = scipy.integrate.quad( secondIntegrandB, -10, 10, args=(vx), limit=5000, limlst=5000, epsrel=0.01)
        if np.isfinite(integral):
            return integral
        else:
            pdb.set_trace()
    integralB, error = scipy.integrate.quad( firstIntegrandB, -10, 10, limit=5000, limlst=5000, epsrel=0.01)
    print "integral,error: ",integralB, error
    vInvAvg = integralB/sigmax * 1.0* sigmax*sigmax*sigmax/np.power(2.0*np.pi, 1.5) * 1.0/(sigmax*sigmay*sigmaz)
    return vAvg, vInvAvg

def figureOne():
    # this is a plot of rate vs. q for different assumptions about the velocity distribution and density.
    n0 = yt.YTQuantity( 0.2, 'AU**-3' )
    #q = yt.YTQuantity(7e10, 'cm')
    mu = G*msun
    def dimensionalRate(vA, vInv, q):
        return n0*np.pi*q*q*vA + 2.0*n0*np.pi*mu*q*vInv

    #U,V,W,sigU,sigV,sigW = (-10,-11,-7, 35,25,25) # bland-hawthorn consensus LSR, old thin disk
    #simCometsLSR = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
    #U,V,W,sigU,sigV,sigW = (-10,-11,-7, 50,50,50) # bland-hawthorn consensus LSR, old thick disk
    #simCometsThickDisk = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
    #U,V,W,sigU,sigV,sigW = (-10.5,-18.0,-8.4, 33,24,17) #  XHIP Anderson & Francis 2012
    #simCometsXHIP = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
    #U,V,W,sigU,sigV,sigW = (-9.7,-22.4,-8.9, 37.9,26.1,20.5) # Reid 2002
    #simCometsMdwarf = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)

    U,V,W,sigU,sigV,sigW = (-10,-11,-7, 35,25,25) # bland-hawthorn consensus LSR, old thin disk
    vAvg, vInvAvg = solarcollisionrate(sigmaU=sigU, sigmaV=sigV, sigmaW=sigW, U=U, V=V, W=W)
   
    qs = yt.YTArray(np.logspace(10, 13+np.log10(5*3.0), 500), 'cm')

    ymin = 1.0e6
    ymax = -1

    fig,ax = plt.subplots()
    #vAvg = 7.35974197173 
    #vInvAvg = 5.64979944705 
    rates = dimensionalRate(vAvg, vInvAvg, qs)
    if float(np.min(rates.in_units('yr**-1'))) < ymin:
        ymin = float(np.min(rates.in_units('yr**-1')))
    if float(np.max(rates.in_units('yr**-1'))) > ymax:
        ymax = float(np.max(rates.in_units('yr**-1')))
    ax.plot(qs.in_units('AU'), rates.in_units('yr**-1'), c='k', lw=5, label='LSR, thin disk')
    print "Within 5 A.U. the rate is, LSR thin: ", dimensionalRate(vAvg, vInvAvg, yt.YTArray( np.linspace(1,9,3), 'AU') ).in_units('yr**-1')

    #vAvg = 10.2154707365 
    #vInvAvg = 7.50490266594 
    U,V,W,sigU,sigV,sigW = (-10,-11,-7, 50,50,50) # bland-hawthorn consensus LSR, old thick disk
    vAvg, vInvAvg = solarcollisionrate(sigmaU=sigU, sigmaV=sigV, sigmaW=sigW, U=U, V=V, W=W)
    #vAvg, vInvAvg = solarcollisionrate(sigmaU=50, sigmaV=35, sigmaW=35, U=10, V=11, W=7)
    rates = dimensionalRate(vAvg, vInvAvg, qs)
    ax.plot(qs.in_units('AU'), rates.in_units('yr**-1'), c='r', lw=3, label='LSR, thick disk')
    print "Within 5 A.U. the rate is, LSR thick: ", dimensionalRate(vAvg, vInvAvg, yt.YTArray( np.linspace(1,9,3), 'AU') ).in_units('yr**-1')

    #vAvg = 8.07247322659 
    #vInvAvg = 5.00554823759 
    U,V,W,sigU,sigV,sigW = (-10.5,-18.0,-8.4, 33,24,17) #  XHIP Anderson & Francis 2012
    vAvg, vInvAvg = solarcollisionrate(sigmaU=sigU, sigmaV=sigV, sigmaW=sigW, U=U, V=V, W=W)
    #vAvg, vInvAvg = solarcollisionrate(sigmaU=30, sigmaV=18, sigmaW=18, U=11.457, V=22.395, W=7.746)
    rates = dimensionalRate(vAvg, vInvAvg, qs)
    ax.plot(qs.in_units('AU'), rates.in_units('yr**-1'), c='b', lw=3, label="XHIP")
    print "Within 5 A.U. the rate is, XHIP: ", dimensionalRate(vAvg, vInvAvg, yt.YTArray( np.linspace(1,9,3), 'AU') ).in_units('yr**-1')

    U,V,W,sigU,sigV,sigW = (-9.7,-22.4,-8.9, 37.9,26.1,20.5) # Reid 2002
    vAvg, vInvAvg = solarcollisionrate(sigmaU=sigU, sigmaV=sigV, sigmaW=sigW, U=U, V=V, W=W)
    rates = dimensionalRate(vAvg, vInvAvg, qs)
    ax.plot(qs.in_units('AU'), rates.in_units('yr**-1'), c='orange', lw=3, label="M dwarfs")
    print "Within 5 A.U. the rate is, M dwarfs: ", dimensionalRate(vAvg, vInvAvg, yt.YTArray( np.linspace(1,9,3), 'AU') ).in_units('yr**-1')


    U,V,W,sigU,sigV,sigW = (0,0,0,20,20,20) # Iso 20
    vAvg, vInvAvg = solarcollisionrate(sigmaU=sigU, sigmaV=sigV, sigmaW=sigW, U=U, V=V, W=W)
    rates = dimensionalRate(vAvg, vInvAvg, qs)
    ax.plot(qs.in_units('AU'), rates.in_units('yr**-1'), c='gray', lw=3, ls=':', label="Isotropic 20 km/s")
    print "Within 5 A.U. the rate is, iso 20kps: ", dimensionalRate(vAvg, vInvAvg, yt.YTArray( np.linspace(1,9,3), 'AU') ).in_units('yr**-1')

    U,V,W,sigU,sigV,sigW = (0,0,0,1,1,1) # Iso 1
    vAvg, vInvAvg = solarcollisionrate(sigmaU=sigU, sigmaV=sigV, sigmaW=sigW, U=U, V=V, W=W)
    rates = dimensionalRate(vAvg, vInvAvg, qs)
    ax.plot(qs.in_units('AU'), rates.in_units('yr**-1'), c='gray', lw=3, label="Isotropic 1 km/s")
    print "Within 5 A.U. the rate is, iso 1kps: ", dimensionalRate(vAvg, vInvAvg, yt.YTArray( np.linspace(1,9,3), 'AU') ).in_units('yr**-1')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Maximum pericenter $q_\mathrm{max}$ (AU)') 
    ax.set_ylabel(r'$\mathcal{R}(q<q_\mathrm{max}) (\mathrm{yr}^{-1})$')

    ydr = ymax/ymin
    ytxt = ymax*10.0**(-0.1 * np.log10(ydr))
    adj = 0.7

    ax.fill_between( [np.min(qs.in_units('AU')), 0.0046], [ymin]*2, [ymax]*2, facecolor='yellow', alpha=0.2)

    ax.text(0.0046*adj,ytxt,'Sundivers', rotation=270)
    ax.plot([0.0046]*2, [ymin,ymax], c='gray', ls='--')
    ax.text(0.016*adj,ytxt,'Sungrazers', rotation=270)
    ax.plot([0.016]*2, [ymin,ymax], c='gray', ls='--') # outer boundary of sungrazing 
    ax.text(0.1537*adj,ytxt,'Sunskirting', rotation=270)
    ax.plot([0.1537]*2, [ymin,ymax], c='gray', ls='--') # outer boundary of sunskirting
    ax.text(0.307*adj,ytxt,'Near-Sun', rotation=270)
    ax.plot([0.307]*2, [ymin,ymax], c='gray', ls='--') # outer boundary of near-sun
    
    ax.set_xlim(np.min(qs.in_units('AU')), np.max(qs.in_units('AU')) )
    ax.set_ylim(ymin,ymax)

    ax.legend(loc=4)
    plt.savefig('fig1.pdf')
    plt.close(fig)


def cross(a,b):
    assert len(a)==len(b) and len(a)==3
    res = np.zeros(3)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = a[2]*b[0] - a[0]*b[2]
    res[2] = a[0]*b[1] - a[1]*b[0]
    return res

def dot(a,b):
    assert len(a)==len(b) 
    return np.sum(a*b)

def norm(vec):
    return np.sqrt(dot(vec,vec))

def computeSolarSystemCoords(vxGal, vyGal, vzGal, b, theta):
    dist = 1000000000.0 # AU-  shouldn't matter - just choose a big number
    vGal = np.array([vxGal, vyGal, vzGal])
    #v = np.sqrt(vxGal*vxGal + vyGal*vyGal + vzGal*vzGal)
    v = norm(vGal)
    # the direction defined by the velocity at some large distance dist. Need to add the impact parameter in!
    xGal = - vxGal/v * dist
    yGal = - vyGal/v * dist
    zGal = - vzGal/v * dist
    rhat = np.array( [-vxGal/v, -vyGal/v, -vzGal/v] )
    # pick a direction perpendicular to this position vector by crossing it with some arbitrary vector (choose zhat)
    rcrossz = cross(rhat, np.array([0,0,1]))
    rcrossz = np.array([ yGal * 1.0, -xGal*1.0, 0])
    rcrosszhat = rcrossz / norm(rcrossz)
    #ok and now to compute the whole coord system we can just define a vector perp to both lol
    yref = cross(rcrosszhat, rhat)
    #yref = np.array([ rcrosszhat[1]*rhat[2] - rcrosszhat[2]*rhat[1], rcrosszhat[2]*rhat[0] - rcrosszhat[0]*rhat[2], rcrosszhat[0]*rhat[1] - rcrosszhat[1]*rhat[0]] )
    yrefhat = yref/norm(yref)
    # ok, so now we can figure out where to nudge the rock!
    assert np.isclose( dot(yrefhat, rcrosszhat), 0)
    assert np.isclose( dot(yrefhat, rhat), 0)
    assert np.isclose( dot(rhat, rcrosszhat), 0)
    adj = b*( rcrosszhat* np.cos(theta) + yrefhat*np.sin(theta) )
    assert np.isclose( dot(adj, rhat), 0)

    xGal = xGal+adj[0]
    yGal = yGal+adj[1]
    zGal = zGal+adj[2]

    AgalPrime = np.zeros((3,3))
    # first row
    AgalPrime[0,0] = -0.0548755604162154
    AgalPrime[0,1] = -0.8734370902348850
    AgalPrime[0,2] = -0.4838350155487132
    # second row
    AgalPrime[1,0] = 0.4941094278755837
    AgalPrime[1,1] = -0.4448296299600112
    AgalPrime[1,2] = 0.7469822444972189
    # third row
    AgalPrime[2,0] = -0.8676661490190047
    AgalPrime[2,1] = -0.1980763734312015
    AgalPrime[2,2] = 0.4559837761750669

    # matrix mult is for y=Ax, y[0] = np.sum(A[0,:]*x[:]) etc.
    rgal = np.array([ xGal,yGal,zGal ])
    vgal = np.array([ vxGal,vyGal,vzGal ])

    rICRS = matrixMult(AgalPrime.T, rgal)

    alpha = 279.804 * np.pi/180.0
    dec = 33.997 * np.pi/180.0

    xou = np.cos(alpha)*np.cos(dec)
    you = np.sin(alpha)*np.cos(dec)
    zou = np.sin(dec)
    
    rICRSnorm = rICRS/np.sqrt(np.sum(rICRS*rICRS))
    #pdb.set_trace()


    vICRS = matrixMult(AgalPrime.T, vgal)
    return rICRS, vICRS

def atan2pos(x,y):
    val = np.arctan2(x,y)
    if val<0:
        val+=2.0*np.pi
    assert val>=0 and val<=2.0*np.pi
    return val

def computeOrbitalElements(x,y,z,vx,vy,vz):
    # solar system coordinates 
    hx = y*vz - z*vy # AU km/s
    hy = z*vx - x*vz # AU km/s
    hz = x*vy - y*vx # AU km/s
    hvec = np.array([hx,hy,hz])
    h = np.sqrt(hx*hx + hy*hy + hz*hz) # AU km/s
    v = np.sqrt(vx*vx + vy*vy + vz*vz) # km/s
    r = np.sqrt(x*x + y*y + z*z) # AU
    E = v*v/2.0 - mu/r # check units -- km^2/s^2
    a = - mu/(2.0*E) # AU
    rvec = np.array([x,y,z])
    vvec = np.array([vx,vy,vz])
    #print E,a,vx,vy,vz,hx,hy,hz,h,v,r
    e = np.sqrt(1 - h*h/(a*mu)) # dim'less
    i = np.arccos(hz/h) # radians b/c used downstream. need to convert before returning to user?

    hzsign = 1
    if hz<0:
        hzsign=-1
    if hz>0:
        RAascending = atan2pos(hx,-hy) # also called big Omega. Right ascension (or longitude I think) of the ascending node -- radians
    else:
        RAascending = atan2pos(-hx,hy) # also called big Omega. Right ascension (or longitude I think) of the ascending node -- radians



        
    if False:     
        argumentOfLatitude = atan2pos(z/np.sin(i), x*np.cos(RAascending) + y*np.sin(RAascending) )# little omeag + nu...? oic... I think nu is the true anomaly (computed next) which evolves as the orbit evolves. So the invariant orbital element is little omega (argument of periapse perhaps?) -- radians
    #    assert np.isclose( 1.0, np.power(z/np.sin(i),2.0) + np.power( x*np.cos(RAascending) + y*np.sin(RAascending), 2.0)) 
        #argumentOfLatitude = atan2pos( z/(r*np.sin(i)), (1.0/np.cos(RAascending)) * (x/r + np.sin(RAascending)*(z/(r*np.sin(i)))*np.cos(i))) 

        if not np.isclose(np.sin(RAascending), hzsign*hx/(h*np.sin(i))) :
            pdb.set_trace()
        if not np.isclose(np.cos(RAascending), hzsign*-hy/(h*np.sin(i))):
            pdb.set_trace()
        #print "Omega OK"

        #assert np.isclose( 1.0, np.power( z/(r*np.sin(i)), 2.0) + np.power(  (1.0/np.cos(RAascending)) * (x/r + np.sin(RAascending)*(z/(r*np.sin(i)))*np.cos(i))  ,2.0))


        #A = np.sin(argumentOfLatitude)
        #B = z/(r*np.sin(i))
        #if not np.isclose( A, B ):
        #    pdb.set_trace()
        #A = np.cos(argumentOfLatitude)
        #B = (1.0/np.cos(RAascending))* (x/r + np.sin(RAascending)*z/(r*np.sin(i)) * np.cos(i))
        #if not np.isclose(A ,B):
        #    pdb.set_trace()
        
        #print "argument of latitude OK"

        nu = np.arccos((a*(1-e*e) -r) /(e*r)) # radians (argument is AU/AU)
        rdotv = x*vx + y*vy + z*vz # AU km/s
        if rdotv<0:
            nu = np.pi + (np.pi-nu)
        nu3 = atan2pos(z/(r*np.sin(i)), 1/np.cos(RAascending) * (x/r + np.sin(RAascending)*z/(r*np.sin(i))*np.cos(i)) )

        p = a*(1-e*e)
        nu2 = atan2pos(np.sqrt(p/mu)*r*r, p-r)
        #if not np.isclose(nu,nu2):
            #print np.sin(RAascending), hx/(h*np.sin(i)), np.cos(RAascending), hy/(h*np.sin(i)), hz

        #    pdb.set_trace()
        omega = argumentOfLatitude - nu # argument of periapse (little omega) -- radians
        if omega<0:
            omega+=2.0*np.pi

    evec = ((v*v - mu/r) * rvec - dot(rvec,vvec)*vvec)/mu
    if not np.isclose( e, norm(evec), rtol=1.0e-2):
        pdb.set_trace()
    n = cross( np.array([0,0,1]), hvec)

    omega = np.arccos( dot(n,evec)/(norm(n)*norm(evec)) )
    if evec[2]<0:
        omega = 2.0*np.pi-omega

    RAascending2 = np.arccos( n[0]/norm(n) )
    if n[1]<0:
        RAascending2 = 2.0*np.pi - RAascending2
    #if not np.isclose(RAascending2, RAascending):
    #    pdb.set_trace()
    return a,e,i,RAascending2,omega

texlabels={}
texlabels['semi'] = r'Semi-major axis $a$ (AU)'
texlabels['ecc'] = r'Eccentricity $e$'
texlabels['peri'] = r'$\log_{10}$ Pericenter $q$ (AU)'
texlabels['inc'] = r'Inclination $I$ (deg)'
texlabels['omega'] = r'Longitude of the ascending node $\Omega$ (deg)'
texlabels['w'] = r'Argument of perihelion $\omega$ (deg)'
texlabels['tp'] = r'Time of perihelion $t_p$'
texlabels['moid'] = r'Earth minimum orbit intersection distance (AU)'
texlabels['A1'] = r'Radial non-grav acceleration $A_1$'
texlabels['rms'] = r'RMS of fit (arcsec)'
texlabels['nobs'] = r'$N_\mathrm{obs}$ used in fit'
texlabels['arc'] = r'$N_\mathrm{days}$ observed for fit'


# observational
class comet:
    def __init__(self, linelist):
        if len(linelist)<20:
            pdb.set_trace()
        self.fullname = linelist[0]
        self.pdes = linelist[1]
        self.name = linelist[2]
        self.prefix = linelist[3]
        self.peri = setval(linelist[4])
        self.ma = setval(linelist[5])
        self.ecc = setval(linelist[6])
        self.semi = setval(linelist[7])
        self.inc = setval(linelist[8])
        self.omega = setval(linelist[9]) # longitude of the ascending node (big omega)
        self.w = setval(linelist[10]) # argument of periapsis (little omega)
        self.tp = setval(linelist[11])
        self.moid = setval(linelist[12]) # earth minimum orbit intersection distance (AU)
        self.ast = linelist[13]
        self.A1 = setval(linelist[14])
        self.rms = setval(linelist[15])
        self.nobs = setval(linelist[16])
        self.semisig = setval(linelist[17]) # AU uncertainty in semi-major axis
        self.date = linelist[18] # date of first observation (UT)
        self.arc = setval(linelist[19]) # days of observations
        self.rank = None # useful later

        if not self.semi is None:
            self.velocity = np.sign(self.semi) * np.sqrt( 1.0 / np.abs(self.semi) * 887.406516)  # 1/AU -> km/s
        else:
            self.velocity = 0
    def __print__(self):
        print "designations:", self.fullname, self.pdes, self.name, self.prefix,self.A1
        print "pericenter: ", self.peri
        print "ecc,inc,omega,w: ",self.ecc,self.inc,self.omega,self.w
        print "a: ",self.semi,"+/-",self.semisig
        print "rms, date, arc, nobs, tp ", self.rms, self.date, self.arc, self.nobs, self.tp

rsun = 0.004679

kreutzgroup = comet(['Kreutz','','Kreutz','',str(rsun*1.5),'','','','143','0','80','','','','','','','','',''])
marsdengroup = comet(['Marsden','','Marsden','',str(rsun*11),'','','','27','79','24','','','','','','','','',''])
krachtgroup = comet(['Kracht','','Kracht','',str(rsun*10),'','','','13','44','59','','','','','','','','',''])
meyergroup = comet(['Meyer','','Meyer','',str(rsun*8),'','','','73','73','57','','','','','','','','',''])
comparisongroups = [kreutzgroup, marsdengroup, krachtgroup, meyergroup]
interestingComets = []
unboundComets = []
oumuamua = []

def energyHist(cometList, simComets=None, axIn=None):
    if not axIn is None:
        ax = axIn
    obsSemi = np.zeros((len(cometList),2))
    simSemi = np.zeros((len(simComets),2))

    for i in range(len(cometList)):
        if cometList[i].semi is None:
            obsSemi[i,0] = np.nan
            obsSemi[i,1] = np.nan
        else:
            obsSemi[i,0] = cometList[i].semi
            obsSemi[i,1] = cometList[i].peri
    for j in range(len(simComets)):
        simSemi[j,0] = simComets[j].semi
        simSemi[j,1] = simComets[j].peri

    lowq = simSemi[:,1] < 0.1
    highq = simSemi[:,1] >= 0.1

    velocity = np.sign(simSemi[lowq,0]) * np.sqrt( 1.0 / np.abs(simSemi[lowq,0]) * 887.406516)  # 1/AU -> km/s
    print "VELOCITY RANGE: ", np.min(velocity), np.max(velocity)
    ax.hist( velocity, bins = 31, density=True, color='green', label=r'Simulated, $q<0.1$ AU', alpha=0.5)

    velocity = np.sign(simSemi[highq,0]) * np.sqrt( 1.0 / np.abs(simSemi[highq,0]) * 887.406516)  # 1/AU -> km/s
    print "VELOCITY RANGE: ", np.min(velocity), np.max(velocity)
    ax.hist( velocity, bins = 31, density=True, color='lightgreen', label=r'Simulated, $q > 0.1$ AU', alpha=0.5)
    

    lowq = obsSemi[:,1] < 0.1
    highq = obsSemi[:,1] >= 0.1

    velocity = np.sign(obsSemi[lowq,0]) * np.sqrt( 1.0 / np.abs(obsSemi[lowq,0]) * 887.406516)  # 1/AU -> km/s
    print "VELOCITY RANGE: ", np.min(velocity), np.max(velocity)
    ax.hist( velocity, bins = 11, density=True, color='k', label=r'Observed, $q<0.1$ AU', alpha=0.5)

    velocity = np.sign(obsSemi[highq,0]) * np.sqrt( 1.0 / np.abs(obsSemi[highq,0]) * 887.406516)  # 1/AU -> km/s
    print "VELOCITY RANGE: ", np.min(velocity), np.max(velocity)
    ax.hist( velocity, bins = 11, density=True, color='gray', label=r'Observed, $q> 0.1$ AU', alpha=0.5)

    ax.set_yscale('log')
    ax.set_ylabel('Probability Density (km/s)$^{-1}$')
    ax.set_xlabel(r'$v_\infty$ (km/s)')

    ax.legend()






def plotAB(cometList, A, B, ident='jpl5', simComets=None, legend=False, axIn=None, legendLoc=None):
    Alist = []
    Blist = []
    Clist = []
    for i in range(len(cometList)):
        Athis = getattr(cometList[i], A)
        Bthis = getattr(cometList[i], B)
        Cthis = getattr(cometList[i], 'semi')
        if not Athis is None and not Bthis is None:
            Alist.append(Athis)
            Blist.append(Bthis)
            if Cthis is None:
                Clist.append('k')
            else:
                Clist.append('b')

    Alist2 = []
    Blist2 = []
    for i in range(len(comparisongroups)):
        Athis = getattr(comparisongroups[i], A)
        Bthis = getattr(comparisongroups[i], B)
        if not Athis is None and not Bthis is None:
            Alist2.append(Athis)
            Blist2.append(Bthis)

    Alist3=[]
    Blist3=[]
    Alist3.append(getattr(oumuamua[0],A))
    Blist3.append(getattr(oumuamua[0],B))


    Alist5 = []
    Blist5 = []
    for i in range(len(unboundComets)):
        Athis = getattr(unboundComets[i], A)
        Bthis = getattr(unboundComets[i], B)
        if not Athis is None and not Bthis is None:
            Alist5.append(Athis)
            Blist5.append(Bthis)

    Alist6 = []
    Blist6 = []
    sizes = np.linspace(150, 80, len(interestingComets))
    for i in range(len(interestingComets)):
        Athis = getattr(interestingComets[i], A)
        Bthis = getattr(interestingComets[i], B)
        if not Athis is None and not Bthis is None:
            Alist6.append(Athis)
            Blist6.append(Bthis)


    if not simComets is None:
        Alist4 = []
        Blist4 = []
        for i in range(len(simComets)):
            try:
                Athis = getattr(simComets[i], A)
                Bthis = getattr(simComets[i], B)
                if not Athis is None and not Bthis is None:
                    Alist4.append(Athis)
                    Blist4.append(Bthis)
            except:
                pass

    if axIn is None:
        fig,ax = plt.subplots()
    else:
        ax = axIn

    xr=None
    yr=None

    if A=='peri':
        #ax.set_xscale('log')
        ax.set_xlim(np.log10(rsun/2.0), np.log10(0.5))
        xr=[np.log10(rsun/2.0), np.log10(0.5)]
        Alist = np.log10(Alist)
        Alist2 = np.log10(Alist2)
        Alist3 = np.log10(Alist3)
        Alist4 = np.log10(Alist4)
        Alist5 = np.log10(Alist5)
        Alist6 = np.log10(Alist6)
        #ax.vlines(np.log10(rsun))


    if A=='semi':
        xr = [-3,3]
    if A=='omega':
        xr = [0, 360]
    if A=='inc':
        xr = [0, 180]
    if A=='w':
        xr = [0, 360]

    if B=='semi':
        yr = [-3,3]
    if B=='omega':
        yr = [0, 360]
    if B=='w':
        yr = [0, 360]
    if B=='inc':
        yr = [0, 180]


    if len(Alist4)>0:
        if len(Alist4)<1000:
            ax.scatter(Alist4, Blist4, s=20, lw=0, alpha=0.8, c='green', label='Simulated Interstellar Objects')
        else:
            theRange = [[np.min(Alist4),np.max(Alist4)],[np.min(Blist4),np.max(Blist4)]]
            if not xr is None:
                theRange[0] = xr[:]
            if not yr is None:
                theRange[1] = yr[:]
            ax.hist2d(Alist4, Blist4, bins=27, cmap='Greens', range=theRange, alpha=0.7 )
    ax.scatter(Alist, Blist, s=20, lw=0, alpha=0.8, c=Clist)
    ax.scatter([],[],s=20,lw=0,alpha=0.4,c='k', label='Solar System')
    ax.scatter([],[],s=20,lw=0,alpha=0.4,c='b', label=r'Solar System, known $a$')





    ax.scatter(Alist6, Blist6, s=sizes, alpha=1.0, c='lightgreen', lw=1, edgecolors='k', label=r"Highest Density Estimates")
    ax.set_xlabel(texlabels[A])
    ax.set_ylabel(texlabels[B])

    ax.scatter(Alist5, Blist5, s=60, lw=0, alpha=1.0, color='pink', marker='h', label=r"$v_\infty < 0$")
    ax.set_xlabel(texlabels[A])
    ax.set_ylabel(texlabels[B])

    ax.scatter(Alist3, Blist3, s=110, lw=1, alpha=1.0, color='orange', edgecolors='k', marker='^', label=r"'Oumuamua")
    ax.set_xlabel(texlabels[A])
    ax.set_ylabel(texlabels[B])

    if len(Alist2)>0:
        ax.scatter(Alist2, Blist2, s=200, lw=0, alpha=1.0, c='r', marker='*', label='Kreutz, Kracht, Marsden, Meyer')


    if legend:
        ax.legend(loc=legendLoc)
    if axIn is None:
        plt.savefig(A+'_'+B+'_'+ident+'.pdf')
        plt.close(fig)
        

def lowq(comets, qlim):
    newcomets = []
    for comet in comets:
        if not comet.peri is None and comet.peri<qlim:
            newcomets.append(comet)
    return newcomets


def summarystats(comets):
    print "Total number of comets: ", len(comets)
    nperi = 0
    nma = 0
    necc = 0
    nsemi = 0
    ninc = 0
    nomega = 0
    nw = 0
    ntp = 0
    nmoid = 0
    nA1 = 0
    nrms = 0
    for comet in comets:
        if not comet.peri is None:
            nperi+=1
        if not comet.ma is None:
            nma +=1
        if not comet.ecc is None:
            necc +=1
        if not comet.semi is None:
            nsemi +=1
        if not comet.inc is None:
            ninc +=1
        if not comet.omega is None:
            nomega +=1
        if not comet.w is None:
            nw +=1
        if not comet.tp is None:
            ntp +=1
        if not comet.moid is None:
            nmoid +=1
        if not comet.A1 is None:
            nA1 +=1
        if not comet.rms is None:
            nrms +=1
    print "n peri: ", nperi
    print "n ma : ", nma
    print "n ecc : ", necc
    print "n semi: ", nsemi
    print "n inc: ", ninc
    print "n omega: ", nomega
    print "n w: ", nw
    print "n tp: ", ntp
    print "n moid: ", nmoid
    print "n A1: ", nA1
    print "n rms: ", nrms

def printInfo(comets):
    print "Printing interesting comets"
    for comet in comets:
        if not comet.semi is None and not comet.peri is None:
            if comet.semi<0 and comet.peri<0.3:
                unboundComets.append(copy.deepcopy(comet))
                print "uuuuuuuuuuuuuuuuuuuuuuu"
                comet.__print__()
                print "uuuuuuuuuuuuuuuuuuuuuuu"

def findInterestingComets( cometsData, cometsSim, save=True, identifier='', isoScore=None):
    # construct a kernel density estimate of the angles associated with the simulated comets.
    from sklearn.neighbors.kde import KernelDensity
    X = np.zeros((len(cometsSim), 3))
    for i in range(len(cometsSim)):
        X[i,0] = cometsSim[i].inc
        X[i,1] = cometsSim[i].omega
        X[i,2] = cometsSim[i].w
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(X)

    Xobs = np.zeros((len(cometsData), 3))
    for j in range(len(cometsData)):
        Xobs[j,0] = cometsData[j].inc
        Xobs[j,1] = cometsData[j].omega
        Xobs[j,2] = cometsData[j].w

    #scores = kde.score_samples(Xobs)
    ### uhhh
    hist, edges = np.histogramdd(X, bins=21, range=((0,180),(0,360),(0,360)) )
    inclinationIndices = np.searchsorted( edges[0], Xobs[:,0] ) -1
    OmegaIndices = np.searchsorted( edges[1], Xobs[:,1] ) -1
    wIndices = np.searchsorted( edges[2], Xobs[:,2] ) -1 

    scores = np.zeros(len(cometsData))
    posteriors = np.zeros(len(cometsData))
    prior = 0.01
    for j in range(len(cometsData)):
        ii, jj, kk = (inclinationIndices[j], OmegaIndices[j], wIndices[j])
        if ii<0:
            ii=0
        if jj<0:
            jj=0
        if kk<0:
            kk=0
        scores[j] = (float(hist[ii,jj,kk ]) / float(np.sum(hist))) * 1.0/float((edges[2][1]-edges[2][0])*(edges[1][1]-edges[1][0])* (edges[0][1]-edges[0][0]) ) * (180.0/np.pi)**3 * 4.0*np.pi**3
        if not isoScore is None:
            posteriors[j] = scores[j]*prior / (scores[j]*prior + isoScore[j]*(1.0-prior))

        #print ii,jj,kk, hist[ii,jj,kk], scores[j]
        #pdb.set_trace()
        #### uhhhhhhhh what units do we want for these? Like.. 4 pi^3 / radian^3?
        #### so like implicitly we have fraction of comets per bin * bin/[binvolume] 
        #if inclinationIndices[j]<0 or OmegaIndices[j]<0 or wIndices[j]<0:
        #    print j, inclinationIndices[j], OmegaIndices[j], wIndices[j], np.shape(hist)
        #    pdb.set_trace()

    # pdb.set_trace()

    Xmod = copy.deepcopy(Xobs)

    def identity(Xorig):
        return Xorig

    def negInclination(Xorig):
        Xorig[:,0] = -Xorig[:,0] 
        return Xorig

    def posInclination(Xorig):
        Xorig[:,0] = 2.0*np.pi-Xorig[:,0] 
        return Xorig

    def twoPiGenerator(column, direction):
        def twoPi(Xorig):
            Xorig[:,column] = Xorig[:,column]+direction*2.0*np.pi
            return Xorig
        return twoPi

    incModifications = [negInclination, identity, posInclination]
    omegaModifications = [twoPiGenerator(1,-1), identity, twoPiGenerator(1,1)]
    wModifications = [twoPiGenerator(2,-1), identity, twoPiGenerator(2,1)]

    #scores = np.zeros(np.shape(kde.score_samples(Xobs)))

#    for i,incMod in enumerate(incModifications):
#        for j,omegaMod in enumerate(omegaModifications):
#            for k,wMod in enumerate(wModifications):
#                if not (i==1 and j==1 and k==1):
#                    Xthis = copy.deepcopy(Xobs)
#                    Xthis = incMod(Xthis)
#                    Xthis = omegaMod(Xthis)
#                    Xthis = wMod(Xthis)
#                    scores = np.logaddexp( scores, kde.score_samples(Xthis) )

    fig,ax = plt.subplots()
    plt.hist(scores, bins=51)
    ax.set_xlabel(r'Density Estimate $f(\Omega,\omega,I)$')
    ax.set_ylabel(r'Number of Observed Objects')
    ax.set_yscale('log')
    plt.savefig('fig3.pdf')
    plt.close(fig)

    idx = np.argsort( posteriors )
    sortedPosteriors= np.zeros(len(posteriors))
    counter = 0
    i=0


    oumuamuascore = None
    oumuamuarank = None
    for i in range(len(scores)):
        ii = idx[-i-1]
        #cuSortedScores[i] = scores[ii]
        sortedPosteriors[i] = posteriors[ii]
        if hasattr(cometsData[ii], 'fullname'):
            if 'umuamua' in cometsData[ii].fullname:
                oumuamuascore = posteriors[ii]
                oumuamuarank = i
        else:
            oumuamuascore = 1000.0
            oumuamuarank = 0


    idxscore = np.argsort( scores )
    sortedScores = np.zeros(len(posteriors))
    counter = 0
    i=0


    #oumuamuascore = None
    #oumuamuarank = None
    for i in range(len(scores)):
        ii = idxscore[-i-1]
        #cuSortedScores[i] = scores[ii]
        sortedScores[i] = scores[ii]




    i=0
    ### here's where we print out the table!!
    fn = 'interestingComets_'+identifier+'.txt'

    def addToLine(line, value, rnd=None, bold=False, first=False):
        if value is None:
            if first:
                line = line + ' - '
            else:
                line = line + r' & '+' - '
        else:
            if not rnd is None:
                if not first:
                    line = line + r' & ' 
                if bold:
                    line = line + r' {\bf ' + str(round(value,rnd)) + r' } '
                else:
                    line = line +  str(round(value,rnd)) 
            else:
                if not first:
                    line = line + r' & ' 
                if bold:
                    line = line + r' {\bf ' + str(value) + r' } '
                else:
                    line = line + str(value)
        return line


    with open(fn,'w') as savefile:
        savefile.write( r'Full Name & q & e & i  & $\Omega$  & $\omega$  & $v_\infty$  & date & $f(\Omega,\omega,I)$ \\'+'\n' )
        print 'oumuamua rank: ', oumuamuarank
        thisScore = 1.0e10
        #while counter<=oumuamuarank:
        #while thisScore > oumuamuascore/1.5:
        while i!=len(posteriors):
            ii = idx[-i-1]
            thisScore = scores[ii]
            thisPosterior = posteriors[ii]
            i+=1
            if hasattr(cometsData[ii],'velocity'):
                printOut = thisPosterior> oumuamuascore/1.5 or cometsData[ii].velocity<-1.0
            else:
                printOut = False
            if printOut:
                print "***********************************************"
                print "Interesting comets: ", i, ii, scores[ii]
                cometsData[ii].__print__()
                print "***********************************************"
                #assert cometsData[ii].peri<0.3
                #if cometsData[ii].peri < 0.3:
                if save:
                    interestingComets.append(copy.deepcopy(cometsData[ii]))
                    interestingComets[-1].rank = i

                smaline = ''
                if cometsData[ii].semi is None:
                    smaline += '-'
                else:
                    smaline += str(round(cometsData[ii].velocity,2))
                #smaline += r' $\pm$ '
                #if cometsData[ii].semisig is None:
                #    smaline += '-'
                #else:
                #    smaline += str(round(cometsData[ii].semisig,2))

                #line = cometsData[ii].fullname.replace('"','') + r' & ' + str(round(cometsData[ii].peri,2)) + r' &  ' + str(cometsData[ii].ecc) + r' & ' + str(round(cometsData[ii].inc,2)) + r' & ' + str(round(cometsData[ii].omega,2)) + r' & ' + str(round(cometsData[ii].w,2)) + r' & ' + smaline + r' & ' + str(cometsData[ii].date) + r' & ' + str(round(thisScore,2)) + r' \\ ' + '\n'
                line = ''
                bold=False
                #if cometsData[ii].velocity<-1:
                #    bold=True
                #if 'muamua'
                line = addToLine( line, cometsData[ii].fullname.replace('"',''), bold=bold, first=True)
                line = addToLine( line, cometsData[ii].peri, rnd=2, bold=bold)
                line = addToLine( line, cometsData[ii].ecc, bold=bold)
                line = addToLine( line, cometsData[ii].inc, rnd=2, bold=bold)
                line = addToLine( line, cometsData[ii].omega, rnd=2, bold=bold) 
                line = addToLine( line, cometsData[ii].w, rnd=2, bold=bold)
                line = addToLine( line, smaline, bold=bold) 
                #line = addToLine( line, cometsData[ii].date, bold=bold)
                ratio = 0
                if not isoScore is None:
                    ratio = thisScore/isoScore[ii]
                line = addToLine( line, ratio, rnd=2, bold=bold) 

                if isoScore is None:
                    posterior = 0
                else:
                    posterior = thisScore*prior / (thisScore*prior + isoScore[ii]*(1.0-prior))

                assert np.isclose( posterior, thisPosterior )

                line = addToLine( line, posterior, rnd=4, bold=bold) + r' \\ ' + '\n'
                savefile.write( line )
                counter += 1
    
    return sortedScores, oumuamuascore, oumuamuarank, scores


def figureThree( sortedScores, oumuamuascore, oumuamuarank, axIn=None, c='k', label=None ):
    if axIn is None:
        fig,ax = plt.subplots()
    else:
        ax = axIn
    ax.plot( sortedScores, np.arange(len(sortedScores))/float(len(sortedScores)), c=c, lw=2, label=label )
    #ax.plot( [oumuamuascore]*2, [1,len(scores)], c='gray', ls='--')
    #ax.plot( [np.min(scores),np.max(scores)], [oumuamuarank]*2 , c='gray', ls='--')
    ax.scatter( [oumuamuascore], [oumuamuarank], c=c )
    print "Oumuamua score and rank: ", oumuamuascore, oumuamuarank
    ax.set_xlim(0, 4.5)
    ax.set_ylim(1.0/len(sortedScores), 1.0)
    ax.set_yscale('log')
    ax.set_xlabel(r'$4\pi^3 f(\Omega,\omega,I) ')
    ax.set_ylabel(r'Fraction of objects with a larger score than $4\pi^3 f(\Omega, \omega, I)$')
    ax.plot( [1.0]*2, [1.0/len(sortedScores),1.0], c='gray', ls='-', lw=1)
    if axIn is None:
        plt.savefig('fig3a.pdf')
        plt.close(fig)

def figureTwo(cometsThis, simComets):
    fig, ax = plt.subplots(3,2, figsize=(10.5,10.5))
    fig.subplots_adjust( wspace=0.2, hspace=0.2, top=0.97, right=0.97, bottom=0.07, left=0.1)
    # let's try this out lol
    # ax[1,0] lower left
    # ax[0,0] upper left
    #
    # [0,0] [0,1]
    # [1,0] [1,1]
    # [2,0] [2,1]

    plotAB(cometsThis, 'omega', 'w', simComets=simComets, axIn=ax[2,0]) # lower left I think
    plotAB(cometsThis, 'omega', 'inc', simComets=simComets, axIn=ax[1,0], legend=True, legendLoc=(.05,1.05)) # upper left
    plotAB(cometsThis, 'inc', 'w', simComets=simComets, axIn=ax[2,1]) # lower right
    energyHist(cometsThis, simComets=simComets, axIn=ax[0,1])
    plotAB(cometsThis, 'peri', 'inc', simComets=simComets, axIn=ax[1,1])
    fig.delaxes(ax[0,0])
    
    plt.savefig('fig2.pdf')
    plt.close(fig)

def figureFour( cometsThis, fidScore, isoScore ):
    fig, ax = plt.subplots()
    #pESs = np.logspace(-4, 0, 100)
    pESs = np.linspace(0, 0.05, 100)
    colors = ['k', 'b', 'r', 'orange', 'purple', 'green', 'lightblue', 'pink', 'maroon']*10
    counter=0
    maxposterior=0
    for i in range(len(cometsThis)):
        if fidScore[i]>2.0:
            posterior = fidScore[i]*pESs / (fidScore[i]*pESs + isoScore[i]*(1.0-pESs))
            maxposterior = np.max([maxposterior, np.max(posterior)])
            ax.plot( pESs, posterior, c=colors[counter] )
            ax.text( 0.04, posterior[-1]*0.9, cometsThis[i].fullname.replace('"','').replace(' ','') )
            counter +=1
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlabel(r'Prior $p(ES)$')
    ax.set_ylabel(r'Posterior $p(ES|\phi_i)$')
    ax.set_xlim(0,0.05)
    ax.set_ylim(0,maxposterior)
    plt.savefig('fig4.pdf')
    plt.close(fig)

def figureFive( cometsThis, fidScore):
    qs = []
    for i in range(len(cometsThis)):
        qs.append(cometsThis[i].peri)
    fig,ax = plt.subplots()
    qs = np.array(qs)
    highScoring = fidScore>1
    ax.scatter( qs[highScoring], fidScore[highScoring], c='r', lw=1 )
    ax.set_xlabel(r'$q$ (AU)')
    ax.set_ylabel(r'$4\pi^3 f(\Omega, \omega, I)$')
    plt.savefig('fig5.pdf')
    plt.close(fig)

def main():

    comets = []

    with open(fn,'r') as f:
        line = f.readline()
        header = line.split(',')
        linelist = line.split(',')
        while len(linelist)>1:
            line = f.readline()
            linelist = line.split(',')
            if len(linelist)>1:
                comets.append(comet(linelist))
                if 'umuamua' in linelist[0]:
                    oumuamua.append(comet(linelist))
            else:
                break

    lowqcomets = lowq(comets, .31)
    cometsThis = lowqcomets

    
    summarystats(cometsThis)
    printInfo(cometsThis)
    #simComets = drawComets(0.28, 2000000, 1000000)
#    U,V,W,sigU,sigV,sigW = (-10,-11,-7, 30,18,18)
#    simCometsFiducial = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
#    U,V,W,sigU,sigV,sigW = (-10,-11,-7, 50,35,35)
#    simCometsHighDisp = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
#    U,V,W,sigU,sigV,sigW = (-11.457,-22.395,-7.746, 30,18,18)
#    simCometsOuaCentered = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)


    U,V,W,sigU,sigV,sigW = (-10,-11,-7, 35,25,25) # bland-hawthorn consensus LSR, old thin disk
    simCometsLSR = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
    U,V,W,sigU,sigV,sigW = (-10,-11,-7, 50,50,50) # bland-hawthorn consensus LSR, old thick disk
    simCometsThickDisk = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
    U,V,W,sigU,sigV,sigW = (-10.5,-18.0,-8.4, 33,24,17) #  XHIP Anderson & Francis 2012
    simCometsXHIP = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
    U,V,W,sigU,sigV,sigW = (-9.7,-22.4,-8.9, 37.9,26.1,20.5) # Reid 2002
    simCometsMdwarf = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)
    U,V,W,sigU,sigV,sigW = (0,0,0,1,1,1) # isotropic 
    simCometsIsotropic = drawComets(.31, 200000, 100000, U,V,W, sigU, sigV, sigW)


    fig,ax = plt.subplots()
    sortedScores, oumuamuascore, oumuamuarank, _ = findInterestingComets( cometsThis, simCometsLSR, identifier='lsrThin')
    figureThree( sortedScores, oumuamuascore, oumuamuarank, axIn=ax, c='k', label='LSR, thin disk' )
    sortedScores, oumuamuascore, oumuamuarank, _ = findInterestingComets( cometsThis, simCometsThickDisk, save=False, identifier='lsrThick')
    figureThree( sortedScores, oumuamuascore, oumuamuarank, axIn=ax, c='r', label=r'LSR, thick disk' )
    sortedScores, oumuamuascore, oumuamuarank, _ = findInterestingComets( cometsThis, simCometsXHIP, save=False, identifier='xhip')
    figureThree( sortedScores, oumuamuascore, oumuamuarank, axIn=ax, c='b', label=r"XHIP sample" )
    sortedScores, oumuamuascore, oumuamuarank, _ = findInterestingComets( cometsThis, simCometsMdwarf, save=False, identifier='mdwarf')
    figureThree( sortedScores, oumuamuascore, oumuamuarank, axIn=ax, c='orange', label=r"M dwarfs" )
    sortedScores, oumuamuascore, oumuamuarank, _ = findInterestingComets( simCometsIsotropic, simCometsLSR, save=False, identifier='isotropic')
    figureThree( sortedScores, oumuamuascore, oumuamuarank, axIn=ax, c='gray', label=r"Isotropic" )
    plt.legend()
    plt.savefig('fig3b.pdf')
    plt.close(fig)

    

        

    _, _, _, fidScore = findInterestingComets( cometsThis, simCometsLSR, save=False, identifier='fid')
    _, _, _, isoScore= findInterestingComets( cometsThis, simCometsIsotropic, save=False, identifier='isotropic')
    _,_,_, _ = findInterestingComets( cometsThis, simCometsLSR, identifier='lsrThin', save=False, isoScore = isoScore) # these are the comets we want to save for output in the paper
    figureFour(cometsThis, fidScore, isoScore) 

    figureFive( cometsThis, fidScore)


    simCometsFiducial=simCometsLSR
    plotAB(cometsThis, 'inc', 'w', simComets=simCometsFiducial)
    plotAB(cometsThis, 'inc', 'omega', simComets=simCometsFiducial)
    plotAB(cometsThis, 'w', 'omega', simComets=simCometsFiducial)
    plotAB(cometsThis, 'inc', 'ecc', simComets=simCometsFiducial)
    plotAB(cometsThis, 'ecc', 'omega', simComets=simCometsFiducial)
    plotAB(cometsThis, 'w', 'ecc', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'inc', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'w', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'omega', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'ecc', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'moid', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'tp', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'rms', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'semi', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'nobs', simComets=simCometsFiducial)
    plotAB(cometsThis, 'peri', 'arc', simComets=simCometsFiducial)
    
    figureTwo(cometsThis, simCometsFiducial)
    
    #figureOne()
    
    #plotAB(simComets, 'peri', 'ecc', ident='sim')
    #plotAB(simComets, 'peri', 'omega', ident='sim')
    #plotAB(simComets, 'peri', 'w', ident='sim')
    #plotAB(simComets, 'omega', 'w', ident='sim')
    #plotAB(simComets, 'peri', 'semi', ident='sim')


if __name__=='__main__':
    main()
