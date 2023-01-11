#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 21:31:21 2022

@author: lomarcus
"""

# Initalize 
# setup ghost and boundary
# calculate flux
# marching to half step
# solve possion equation
# calculate residual

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import pickle


def ghost(U,V,Nx,Ny):      #Function for ghost condition
    #BOTTOM WALL Ghost point condition
    U[1:Nx+2,0] = -U[1:Nx+2,1]  
    V[1:Nx+1,0]  = V[1:Nx+1,2] 
    
    #Left Ghost
    U[0,1:int(Ny/2)+1]  = U[2,1:int(Ny/2)+1]     
    V[0,1:int(Ny/2)+1]  = -V[1,1:int(Ny/2)+1]     
    
    #TOP WALL Ghost point condition
    U[1:Nx+2,Ny+1]  = -U[1:Nx+2,Ny]  #
    V[1:Nx+1,Ny+2] = V[1:Nx+1,Ny]   
    return(U,V)

def Flux(U,V,Nx,Ny,h,nu):    #Function for Flux calculation
## looping for F and G
    F = np.zeros((Nx,Ny)); G = np.zeros((Nx,Ny));   
    Hx = np.zeros((Nx+1,Ny+1)); Hy = np.zeros((Nx+1,Ny+1));
    for i in range(Nx):
        for j in range(Ny):
            iu=i+2; ju=j+1; iv=i+1; jv = j+2; 
            uL = U[iu-1,ju]; uLL = U[iu-2,ju]; uR = U[iu,ju]; uRR = U[iu+1,ju];
            vB = V[iv,jv-1]; vBB = V[iv,jv-2]; vT = V[iv,jv]; vTT = V[iv,jv+1];
            
            # Calculate F
            q = (uL+uR)/2;
            
            #QUICK
            if q >= 0:
                phi = (3*uR+6*uL-uLL)/8;
            else:
                phi = (3*uL+6*uR-uRR)/8;
           
            F[i,j] = q*phi-nu*(uR-uL)/h;
           
            
            # Calculate G
            q = (vT+vB)/2;
            
            #QUICK
            if q >= 0:
                phi = (3*vT+6*vB-vBB)/8;
            else:
                phi = (3*vB+6*vT-vTT)/8;
            G[i,j] = q*phi-nu*(vT-vB)/h;
        
##Looping for Hx and Hy WITH USING if statement for boundary condition
    for i in range(Nx+1):
        for j in range(Ny+1):
            if i == 0:
                iu=1; ju=j; iv=1; jv = j+1; 
                uB = U[iu,ju]; uT = U[iu,ju+1];
                vL = V[iv-1,jv]; vR = V[iv,jv]; 
                Hx[0,j] = -nu*(uT-uB)/h
                Hy[0,j] = -nu*(vR-vL)/h  
            elif i== Nx:
                #Right wall 
                iu=i+1; ju=j; iv=i+1; jv = j+1; 
                uB = U[iu,ju]; uT = U[iu,ju+1];
                vL = V[iv-1,jv]; vR = V[iv,jv]; 
                Hx[Nx,j] = -nu*(uT-uB)/h
                Hy[Nx,j] = -nu*(vR-vL)/h
            elif j ==0:
                iu=i; ju=j; iv=i; jv = j+1; 
                uB = U[iu,ju]; uT = U[iu,ju+1];
                vL = V[iv-1,jv]; vR = V[iv,jv]; 
                Hx[i,0] = -nu*(uT-uB)/h
                Hy[i,0] = -nu*(vR-vL)/h
            elif j ==Ny:
                iu=i+1; ju=j; iv=i; jv = j+1; 
                uB = U[iu,ju]; uT = U[iu,ju+1];
                vL = V[iv-1,jv]; vR = V[iv,jv]; 
                Hx[i,j] = -nu*(uT-uB)/h
                Hy[i,j] = -nu*(vR-vL)/h
            else:
                iu=i+1; ju=j+1; iv=i+1; jv = j+1; 
                uB = U[iu,ju-1]; uBB = U[iu,ju-2]; uT = U[iu,ju]; uTT = U[iu,ju+1];
                vL = V[iv-1,jv]; vLL = V[iv-2,jv]; vR = V[iv,jv]; vRR = V[iv+1,jv];
        
                #Calculate Hx
                q = (vL+vR)/2;
                
                #QUICK
                if q >= 0:
                    phi = (3*uT+6*uB-uBB)/8;  
                else:
                    phi = (3*uB+6*uT-uTT)/8;
                Hx[i,j] = q*phi-nu*(uT-uB)/h;
        
                # Calculate Hy
                q = (uB+uT)/2;
                
                #QUICK
                if q >= 0:
                    phi = (3*vR+6*vL-vLL)/8;
                else:
                    phi = (3*vL+6*vR-vRR)/8;
                Hy[i,j] = q*phi-nu*(vR-vL)/h;
    return(F,G,Hx,Hy)
        
def Halfmar(U,V,F,G,Hx,Hy,h,Nx,Ny,dt):   #march to intimidate step for U and V
    for i in range (2,Nx+1):
        for j in range (1,Ny+1):
            U[i,j] = U[i,j]-dt*(F[i-1,j-1]-F[i-2,j-1]+Hx[i-1,j]-Hx[i-1,j-1])/h
    for i in range (1,Nx+1):
        for j in range (2,Ny+1):
            V[i,j] = V[i,j]-dt*(G[i-1,j-1]-G[i-1,j-2]+Hy[i,j-1]-Hy[i-1,j-1])/h
    return(U,V)
    
# Solve PPE
def PPE(Nx,Ny,V,U,h,dt,A):
    N = (Nx)*(Ny);  #total number of unknowns
    b = np.zeros(N);
    for iy in range(Ny):
        for ix in range(Nx):
            i=iy*(Nx)+ix;
            
            if ix ==50 and iy ==3:   #pinning one point to 0 for avoiding singular
                b[i]=0
            else:
               
                b[i]= (U[ix+2,iy+1]-U[ix+1,iy+1]+V[ix+1,iy+2]-V[ix+1,iy+1])/(h*dt);
    A = A.tocsr()        
    Psol = linalg.spsolve(A,b)
    P = np.reshape(Psol,(Nx,Ny), order = 'F')# reshape into matrix
    return(P)

def GetA(Nx,Ny,V,U,h,dt):     #setting A configuration
    h2 = h*h;
    N = (Nx)*(Ny);  #total number of unknowns

    A = sparse.lil_matrix((N,N));
    
    for iy in range(Ny):
        for ix in range(Nx):
      
            i=iy*(Nx)+ix;
            iL = i-1;iR=i+1;iD = i-(Nx); iU = i+(Nx);
            
            if iy==0:   #Bottom
                if ix ==0:  #Bottom left corner
                    A[i,i]= -2/h2;
                    A[i,iR] = 1/h2;A[i,iU] = 1/h2;
                elif ix==Nx-1:  #Bottom right corner
                    A[i,i]= -2/h2;
                    A[i,iL] = 1/h2;A[i,iU] = 1/h2;
                else:  #Bottom 
                    A[i,i] = -3/h2
                    A[i,iL] = 1/h2;A[i,iU] = 1/h2; A[i,iR] = 1/h2;
            elif iy==Ny-1:  #TOP
                if ix ==0:  #top left corner
                    A[i,i]= -2/h2;
                    A[i,iR] = 1/h2;A[i,iD] = 1/h2;
                elif ix==Nx-1: #top right corner
                    A[i,i]= -2/h2;
                    A[i,iL] = 1/h2;A[i,iD] = 1/h2;
                else:  #top
                    A[i,i] = -3/h2
                    A[i,iL] = 1/h2;
                    A[i,iD] = 1/h2; 
                    A[i,iR] = 1/h2;
                    
            elif ix ==0:   #left wall
                A[i,i] = -3/h2; A[i,iU] = 1/h2;
                A[i,iD] = 1/h2; 
                A[i,iR] = 1/h2;
            elif ix ==Nx-1:  #right wall
                A[i,i] = -3/h2
                A[i,iD] = 1/h2; A[i,iL] = 1/h2;
                A[i,iU] = 1/h2;
    
            elif ix ==50 and iy ==3:  #pinning random point
                A[i,i] = 1;
              
            else:       #Inner
                A[i,i] = -4/h2;
                A[i,iL] = 1/h2;A[i,iR] = 1/h2;
                A[i,iD] = 1/h2;A[i,iU] = 1/h2;
    
    return A


def pvsolver(Re, Ny, f,beta): #solver that recieve Re, Ny f and beta for converging
    
    #Setting constants and initalize
    Nx = f*Ny;
    Q= 2/3; #flow rate
    nu = 2*Q/Re;
    H=1;
    Ly =2*H;
    Lx = f*Ly;
    
    x = np.linspace(0,Lx,Nx+1);y = np.linspace(0,Ly,Ny+1);
    (Y,X) = np.meshgrid(y,x);
    
    
    U = np.zeros((Nx+3,Ny+2)); V = np.zeros((Nx+2,Ny+3));
    
    n=0;
    #P = np.zeros((Nx,Ny));
    
    
    ## Loop over time step
    
    h= Ly/Ny;
    
    mid= H+Ly/(2*Ny);
    inflow = 0;
    
    # Set up inflow and outflow and  ghost point
    
    inflow
    for i in range(Ny+1):
        if Y[0,i]> 1 and Y[0,i]<= 2:
            U[1,i] = (-4*mid**2) + (12*mid) - 8 #parabola profile
            inflow += U[1,i]*h
            mid+= h
    scale_in = Q/inflow
    U = scale_in*U
    inflow = np.sum(U[1,:]*h)
        
    U[0,int(Ny/2)+1:Ny+1] = U[1,int(Ny/2)+1:Ny+1]
    V[0,int(Ny/2)+1:Ny+2]  = 0;
    
    #Outflow velocity
    mid= Ly/(2*Ny);
    outflow = 0
    for i in range(1,Ny+1):
        U[Nx+1,i] = (-0.5*mid**2) + mid #parabola profile
        outflow += U[Nx+1,i]*h
        mid += h
  
    scale_out = Q/outflow
    U[Nx+1,:] = scale_out* U[Nx+1,:]
    outflow = np.sum(U[Nx+1,:]*h)
    U[Nx+2,1:Ny+1] = U[Nx+1,1:Ny+1]
    
    #Wall condition
    V[Nx+1,1:Ny+2]  = 0    #outflow
    V[1:Nx+1,1]  = 0        #bottom wall
    V[1:Nx+1,Ny+1]  = 0     #TOP
    
    #Cacluate the time step
    
    dt = beta*np.min([ h**2/(4*nu),4*nu/np.max(U**2)])
    
    
    n=1
    #history = np.zeros((1,2));
    
    #Looping to converge steady sate
    while n<200000:
        U,V = ghost(U,V,Nx,Ny);
        F,G,Hx,Hy = Flux(U,V,Nx,Ny,h,nu);
        U,V = Halfmar(U,V,F,G,Hx,Hy,h,Nx,Ny,dt);
        if n ==1:
            print("n=1")
            A = GetA(Nx,Ny,V,U,h,dt)
        P = PPE(Nx,Ny,V,U,h,dt,A)
      
        # Update U and V
        for i in range (2,Nx+1):
              for j in range (1,Ny+1):
                  U[i,j]=U[i,j]-dt*(P[i-1,j-1]-P[i-2,j-1])/h
        for i in range (1,Nx+1):
              for j in range (2,Ny+1):
                  V[i,j] = V[i,j]-dt*(P[i-1,j-1]-P[i-1,j-2])/h    
        print(n)
        
        #Calculate residual
        Rx = 0;Ry = 0;R=0;
        #Calculate L1
        for i in range (Nx-1):
            for j in range (Ny):
                Rx += abs(h*(F[i+1,j]+P[i+1,j]-F[i,j]-P[i,j])+h*(Hx[i+1,j+1]-Hx[i+1,j]))
        for i in range(Nx):
            for j in range(Ny-1):
                Ry += abs(h*(G[i,j+1]+P[i,j+1]-G[i,j]-P[i,j])+h*(Hy[i+1,j+1]-Hy[i,j+1]))
        R +=Rx+Ry
        
        #Save Residual for Re200 only
        #history = np.vstack((history,[n,R]))
        n=n+1
        print(R)
        
        #Break loop if residual lower than 1e-5
        if R < 1E-5:
            break
     
    #save U and V after converging for plotting later
    file = open('U_Re{}_Ny{}'.format(Re,Ny), 'wb'); pickle.dump(U, file); file.close()
    file = open('V_Re{}_Ny{}'.format(Re,Ny), 'wb'); pickle.dump(V, file); file.close()
    
         
    ###Plot residual
    
    # f = plt.figure(figsize =(10,8))
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['savefig.dpi'] = 300 
    # plt.semilogy(history[1:,0], history[1:,1] )
    # plt.xlabel('Number of iterations',fontsize=20)
    # plt.ylabel('$L_1$',fontsize=20)
    # plt.title('$L_1$ vs iterations for $Ny={}$ and $Re={}$'.format(Ny,Re),fontsize=28)
    # plt.savefig('L1_Re{}_Ny_{}.png'.format(Re,Ny)) 
    # #plt.show()

