import numpy as np
import matplotlib.pyplot as plt
import math

Nx,Ny = 0
dx,dy,dA = 0
dx_negsqr = 1/dx/dx, dy_negsqr = 1/dy/dy, one_over_dA = 1/dA


def error(Res,psi,zeta): 
    return np.max(abs(Res[:,:]))/(np.sum(abs(psi[:,:]))*dA+np.max(abs(zeta[:,:])))

def optimalAlpha(): #Relaxation constant
    beta = dx/dy
    sigma = (math.cos(math.pi/Nx)+beta*beta*math.cos(math.pi/Ny))/(1+beta*beta)
    return 2/(1+math.sqrt(1-sigma**2))

def Jacobi(field, init_guess, tol, plot=False):
    count = 1
    Res = np.zeros((Nx+2,Ny+2))
    zeta = np.copy(field)
    psi_now = np.copy(init_guess); psi_next = np.zeros((Nx+2,Ny+2))

    Res[1:Nx-1,1:Ny-1] = dx_negsqr*(psi_now[0:Nx-2,1:Ny-1]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[2:Nx,1:Ny-1])+dy_negsqr*(psi_now[1:Nx-1,0:Ny-2]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[1:Nx-1,2:Ny])-zeta[1:Nx-1,1:Ny-1]
    psi_next[1:Nx-1,1:Ny-1] = psi_now[1:Nx-1,1:Ny-1]+(Res[1:Nx-1,1:Ny-1])*one_over_dA

    psi_now = psi_next

    if(plot):
        fig, ax = plt.subplots()
        CS = ax.contour(psi_now)
        ax.clabel(CS, fontsize=10)
        ax.set_title('Simplest default with labels')
        plt.draw()

    while(error(Res,psi_now,zeta)>tol):
        Res[1:Nx-1,1:Ny-1] = dx_negsqr*(psi_now[0:Nx-2,1:Ny-1]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[2:Nx,1:Ny-1])+dy_negsqr*(psi_now[1:Nx-1,0:Ny-2]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[1:Nx-1,2:Ny])-zeta[1:Nx-1,1:Ny-1]
        psi_next[1:Nx-1,1:Ny-1] = psi_now[1:Nx-1,1:Ny-1]+(Res[1:Nx-1,1:Ny-1])*one_over_dA

        psi_now = psi_next

        print(error(Res,psi_now,zeta))
        count += 1

        if(plot):
            plt.cla()
            CS = ax.contour(psi_now)
            ax.clabel(CS, fontsize=10)
            plt.draw()
            plt.pause(.1)
        
    return psi_now, count

def GaussSeidel(field,initGuess, tol, plot=False):
    count = 0
    zeta = np.copy(field)

    zeta[:,0] = 0
    zeta[:,Ny-1] = 0
    zeta[0,:] = 0
    zeta[Nx-1,:] = 0

    psi_now = np.copy(initGuess)
    psi_now[:,0] = 0
    psi_now[:,Ny-1] = 0
    psi_now[0,:] = 0
    psi_now[Nx-1,:] = 0
    
    Res = np.zeros((Nx,Ny))
    Res[1:Nx-1,1:Ny-1] = dx_negsqr*(psi_now[0:Nx-2,1:Ny-1]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[2:Nx,1:Ny-1])+dy_negsqr*(psi_now[1:Nx-1,0:Ny-2]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[1:Nx-1,2:Ny])-zeta[1:Nx-1,1:Ny-1]

    if(plot):
        fig, ax = plt.subplots()
        CS = ax.contour(psi_now)
        ax.clabel(CS, fontsize=10)
        ax.set_title('Simplest default with labels')
        plt.draw()

    for t in range(1):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Res[i,j] = dx_negsqr*(psi_now[i+1,j]-2*psi_now[i,j]+psi_now[i-1,j])+dy_negsqr*(psi_now[i,j+1]-2*psi_now[i,j]+psi_now[i,j-1])-zeta[i,j]
                psi_now[i,j] = psi_now[i,j]+alpha*one_over_dA*(Res[i,j]) 
        count+=1
    
    while(error(Res,psi_now,zeta)>tol):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Res[i,j] = dx_negsqr*(psi_now[i+1,j]-2*psi_now[i,j]+psi_now[i-1,j])+dy_negsqr*(psi_now[i,j+1]-2*psi_now[i,j]+psi_now[i,j-1])-zeta[i,j]
                psi_now[i,j] = psi_now[i,j]+1*one_over_dA*(Res[i,j]) 

        count += 1
        
        if(plot):
            plt.cla()
            CS = ax.contour(psi_now[1:Nx+1,1:Ny+1])
            ax.clabel(CS, fontsize=10)
            plt.draw()
            plt.pause(.1)

    return psi_now, count

alpha = optimalAlpha()

def SOR(field,initGuess, tol, plot=False):
    count = 0
    zeta = np.copy(field)

    zeta[:,0] = 0
    zeta[:,Ny-1] = 0
    zeta[0,:] = 0
    zeta[Nx-1,:] = 0

    psi_now = np.copy(initGuess)
    psi_now[:,0] = 0
    psi_now[:,Ny-1] = 0
    psi_now[0,:] = 0
    psi_now[Nx-1,:] = 0
    
    Res = np.zeros((Nx,Ny))
    Res[1:Nx-1,1:Ny-1] = dx_negsqr*(psi_now[0:Nx-2,1:Ny-1]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[2:Nx,1:Ny-1])+dy_negsqr*(psi_now[1:Nx-1,0:Ny-2]-2*psi_now[1:Nx-1,1:Ny-1]+psi_now[1:Nx-1,2:Ny])-zeta[1:Nx-1,1:Ny-1]

    if(plot):
        fig, ax = plt.subplots()
        CS = ax.contour(psi_now)
        ax.clabel(CS, fontsize=10)
        ax.set_title('Simplest default with labels')
        plt.draw()

    for t in range(1):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Res[i,j] = dx_negsqr*(psi_now[i+1,j]-2*psi_now[i,j]+psi_now[i-1,j])+dy_negsqr*(psi_now[i,j+1]-2*psi_now[i,j]+psi_now[i,j-1])-zeta[i,j]
                psi_now[i,j] = psi_now[i,j]+alpha*one_over_dA*(Res[i,j]) 
        count+=1
    
    while(error(Res,psi_now,zeta)>tol):
        for i in range(1,Nx-1):
            for j in range(1,Ny-1):
                Res[i,j] = dx_negsqr*(psi_now[i+1,j]-2*psi_now[i,j]+psi_now[i-1,j])+dy_negsqr*(psi_now[i,j+1]-2*psi_now[i,j]+psi_now[i,j-1])-zeta[i,j]
                psi_now[i,j] = psi_now[i,j]+alpha*one_over_dA*(Res[i,j]) 

        count += 1
        
        if(plot):
            plt.cla()
            CS = ax.contour(psi_now[1:Nx+1,1:Ny+1])
            ax.clabel(CS, fontsize=10)
            plt.draw()
            plt.pause(.1)

    return psi_now, count



