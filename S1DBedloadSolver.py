
# coding: utf-8

# # License
# https://raw.githubusercontent.com/computational-sediment-hyd/a-rudimentary-knowledge-of-river-bed-variation/master/LICENSE

# # import module

# In[ ]:


import numpy as np
import pandas as pd
import numba
import sys
import os
import json
import glob as glob


# # Governing Equation of River flow 

# \begin{align}
#     \frac{\partial A}{\partial t}+\frac{\partial Q}{\partial x} = 0 \\
# \end{align}
# 
# \begin{align}
#     \frac{\partial Q}{\partial t}+\frac{\partial}{\partial x}\left(\frac{ Q^2}{ A}\right) + gA\frac{\partial H}{\partial x}+gAi_e = 0
# \end{align}

# In[ ]:


@numba.jit(nopython=True, parallel=False)
def flowmodel(A, Q, Adown, Qup, dAb, zb, B, dx, dt, g, manning):
    imax = len(A)
    Anew, Qnew = np.zeros(imax), np.zeros(imax)
    
# continuous equation
    for i in numba.prange(1, imax-1) : Anew[i] = A[i] - dt * ( Q[i] - Q[i-1] ) / dx
        
    Anew[0], Anew[-1] = Anew[1], Adown
    
# moumentum equation
    for i in numba.prange(1,imax-1): 
        ip, ic, im = (i+1, i, i-1) 
        Cr1 = 0.5*( Q[ic]/A[ic] + Q[ip]/A[ip] )*dt/dx
        Cr2 = 0.5*( Q[ic]/A[ic] + Q[im]/A[im] )*dt/dx
        dHdx1 = ( Anew[ip]/B[ip] + zb[ip] + dAb[ip]/B[ip] - Anew[ic]/B[ic] - zb[ic] - dAb[ic]/B[ic] ) / dx
        dHdx2 = ( Anew[ic]/B[ic] + zb[ic] + dAb[ic]/B[ic] - Anew[im]/B[im] - zb[im] - dAb[im]/B[im] ) / dx
        dHdx = (1.0 - Cr1) * dHdx1 + Cr2 * dHdx2
        
        Qnew[ic] = Q[ic] - dt * ( Q[ic]**2/A[ic] - Q[im]**2/A[im] ) / dx                     - dt * g * Anew[ic] * dHdx                    - dt * g * A[ic] * manning**2 * Q[ic]**2 / B[ic]**2 / ( A[ic]/B[ic] )**(10.0/3.0)
                
    Qnew[0], Qnew[-1] = Qup, Qnew[-2]
            
    return Anew, Qnew 


# # Governing Equation of River Bed : Hirano model

# 
# \begin{align}
#     (1-\lambda)\frac{\partial A_{b}}{\partial t}+\frac{\partial }{\partial x} \sum_{i=1}^n ( Q_{bi}P_i)= 0 \\
# \end{align}
# 
# \begin{align}
#     \frac{\partial P_i}{\partial t} &= - \frac{1}{(1-\lambda)E_dB} \frac{\partial (Q_{bi}P_i)}{\partial x} - \frac{P_{si}}{E_d B}\frac{\partial A_b}{\partial t} \\
#     &= - \frac{1}{E_d B}\left(\frac{\partial A_{bi}}{\partial t} + P_{si}\frac{\partial A_b}{\partial t}\right) 
# \end{align}
# 
# \begin{align}
# &P_{si} = 
# \left\{ \begin{array}{ll}
#     P_i \mbox{ in exchange layer} & \left(\dfrac{\partial A_b}{\partial t} > 0 \right) \\
#     P_i \mbox{ under exchange layer} & \left(\dfrac{\partial A_b}{\partial t} < 0 \right) \\
# \end{array} \right. \\
# \end{align}
# 
# 
# $Q_{bi}$ : Ashida-Michiue Eq. and  Egiazaroff Eq.
# 
# $E_d$ : thickness of exchange layer

# \begin{align}
#     \frac{\tau_{*c i}}{\tau_{*cm}} &= 0.85 \frac{D_m}{D_i} \qquad & \left(\frac{D_i}{D_m} < 0.4 \right) \\
#     \dfrac{\tau_{*ci}}{\tau_{*cm}} &= \left( \dfrac{ \displaystyle \log_e 19 }{ \displaystyle \log_e \left(19\dfrac{D_i}{D_m} \right) } \right)^2 \qquad & \left(\frac{D_i}{D_m} \geq 0.4 \right)
# \end{align}

# In[ ]:


@numba.jit(nopython=True, parallel=False)
def sedimentmodel(dAb, dratio, A, Q, B, dx, dt, g, manning, dsize, dratioStandard, hExlayer, Qbup):
    rhosw = 1.65 # grain specific gravity in water
    porosity = 0.4
    tscAve = 0.05 # critical tractive force of average grain size
    
    dAbnew = np.zeros_like( dAb )
    drationew = np.zeros_like( dratio )
    Qbsub = np.zeros_like( dratio )
    dAbsub = np.zeros_like( dratio )
    
    imax, lmax = len(dratio), len(dratio[0])
    
    for i in numba.prange(imax):
        Ap, Qp, dr, Bp = A[i], Q[i], dratio[i], B[i]
        dAve = np.sum(dsize * dr)
        us = np.sqrt(g * Ap/Bp) * Qp/Ap * manning / (Ap/Bp)**(2/3)
        tsAve = us**2.0/rhosw/g/dAve
        use = Qp/Ap / ( 6.0 + 2.5 * np.log( Ap/Bp/dAve/( 1.0+2.0*tsAve) ) )
        Kc=1.0
        
        for l in range(lmax):
            dri, dsi = dr[l], dsize[l]
            ts  = us**2.0 /rhosw/g/dsi
            tse = use**2.0/rhosw/g/dsi
            
# Egiazaroff Eq. 
            x = dsi/dAve
            tscbytscm = (np.log(19)/np.log(19*x))**2 if x > 0.4 else 0.85/x
            tsc = Kc * tscbytscm * tscAve 
       
            if ts > tsc :
# Ashida-Michiue Eq. 
                Qbsub[i][l] = np.sqrt(rhosw * g * dsi**3.0)                         * 17.0 * tse**1.5 * ( 1.0 - tsc / ts) * ( 1.0 - np.sqrt(tsc / ts) )                         * dri * B[i]
            else:
                Qbsub[i][l] = 0.0
                
    for i in numba.prange(imax):
        Qbs = Qbsub[i]
        if i == 0:
            if np.min(Qbup) < 0.0 : # equilibrium sand supply
                dAbsub[i][:] = 0.0
            else:
                for l in range(lmax):
                    dAbsub[i][l] = - dt / (1.0 - porosity) * ( Qbsub[i][l]-Qbup[l] )/dx
                    if np.abs(dAbsub[i][l]) > hExlayer :
                        print(i) ; print('dzsub-error')
        else:
            for l in range(lmax):
                dAbsub[i][l] = - dt / (1.0 - porosity) * ( Qbsub[i][l] - Qbsub[i-1][l] )/dx
                if np.abs(dAbsub[i][l]) > hExlayer : 
                    print(i) ; print('dzsub-error')
                
    for i in numba.prange(imax):
        Qbs = Qbsub[i]
# update dAb 
        dAball = np.sum( dAbsub[i] )
        if np.abs(dAball) > hExlayer:
            print(i) ; print('dz-error')
            
        dAbnew[i] = dAb[i] + dAball
        
# update dratio
        dratioIn = dratioStandard[i][:] if dAball < 0.0 else dratio[i][:]
            
        for l in range(lmax):
            drationew[i][l] = dratio[i][l] + ( dAbsub[i][l] - dAball * dratioIn[l] ) /hExlayer/B[i]
            if drationew[i][l] < 0.0 : drationew[i][l] = 0.0
            
# correct ration so that sum of ration become 100% 
        sumdr = np.sum( drationew[i] )
        drationew[i] /= sumdr
        
    return dAbnew, drationew, Qbsub


# # main

# In[ ]:


def bedvariation(
dx,dt,manning,totalTime,outTimeStep,RunUpTime
,dsize ,dratioStandard ,dratio ,hExlayer ,A ,Q ,B ,zb ,dAb ,Qup ,Adown 
,outputfilename, screenclass
):
    
    g = 9.8
    Qbup = np.full( ( len(dsize) ), -9999.0 ) # when equilibrium sand supply, set qbup to minus value
        
# run-up calculation
    for i in range(int(RunUpTime/dt)):
        ib = ( ( dAb[-2]/B[-2] + zb[-2] ) - ( dAb[-1]/B[-1] + zb[-1] ) )/dx
        Adownp = Adown(0.0, Q[-1], dAb[-1]/B[-1], ib)
        Qupp = Qup(0.0)
        A, Q = flowmodel(A, Q, Adownp, Qupp, dAb, zb, B, dx, dt, g, manning)
        
# main calculation
    for i in range( int(totalTime/dt) ):
        
# cal bed variation
        dAb, dratio, Qbsub = sedimentmodel(dAb, dratio, A, Q, B, dx, dt, g, manning, dsize, dratioStandard, hExlayer, Qbup)
        
# cal water profile
        ib = ( ( dAb[-2]/B[-2] + zb[-2] ) - ( dAb[-1]/B[-1] + zb[-1] ) )/dx
        Adownp = Adown(i*dt, Q[-1], dAb[-1]/B[-1], ib)
        Qupp = Qup(i*dt)
        A, Q = flowmodel(A, Q, Adownp, Qupp, dAb, zb, B, dx, dt, g, manning)
        
# output 
        if( int(i*dt) % int(outTimeStep) ) == 0 :
            print( str( int(i*dt) ) + ' second') 
            profile = []
            for ii, (ap, qp, z, r, q) in enumerate(zip(A, Q, dAb, dratio, Qbsub)) : 
                profile.append({'distance' : int(ii*dx), 'A':ap, 'Q':qp, 'dAb':z, 'ratio':list(r), 'Qb':list(q)})
            json.dump( {'time':int(i*dt), 'profile' : profile}, open('%010d' % int(i*dt) + 'sec.json', 'w') )
            
# join json files
    d = [ json.load( open(f, 'r') ) for f in glob.glob('*sec.json') ]
# delete json files
    for f in glob.glob('*sec.json') : os.remove(f)
        
    d.sort(key=lambda x: x['time'])
    
    cond = {'width':list(B), 'elevation':list(zb), 'screenclass':list(screenclass), 'dsize':list(dsize) }
    json.dump( {'condition':cond, 'output':d}, open(outputfilename, 'w') )

