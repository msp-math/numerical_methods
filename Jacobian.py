import numpy as np
import matplotlib.pyplot as plt
import math

Nx,Ny = 0
dx,dy = 0
one_dA_fourth = 1/4/dx/dy

def ArakawaJacobian(psi,zeta):
    return (1/3)*(J1(psi,zeta)+J2(psi,zeta)+J3(psi,zeta))

def J1(psi,zeta):
    J = np.zeros((Nx,Ny))
    J[1:Nx-1,1:Ny-1] = (
    (psi[2:Nx,1:Ny-1]-psi[0:Nx-2,1:Ny-1])*(zeta[1:Nx-1,2:Ny]-zeta[1:Nx-1,0:Ny-2])
    -(zeta[2:Nx,1:Ny-1]-zeta[0:Nx-2,1:Ny-1])*(psi[1:Nx-1,2:Ny]-psi[1:Nx-1,0:Ny-2]))*one_dA_fourth
    return J


def J2(psi,zeta):
    J = np.zeros((Nx,Ny))

    J[1:Nx-1,1:Ny-1] = (
    psi[2:Nx,1:Ny-1]*(zeta[2:Nx,2:Ny]-zeta[2:Nx,0:Ny-2])
    -psi[0:Nx-2,1:Ny-1]*(zeta[0:Nx-2,2:Ny]-zeta[0:Nx-2,0:Ny-2])
    -psi[1:Nx-1,2:Ny]*(zeta[2:Nx,2:Ny]-zeta[0:Nx-2,2:Ny])
    +psi[1:Nx-1,0:Ny-2]*(zeta[2:Nx,0:Ny-2]-zeta[0:Nx-2,0:Ny-2]))*one_dA_fourth
    return J

def J3(psi,zeta):
    J = np.zeros((Nx,Ny))

    J[1:Nx-1,1:Ny-1] = (
    -zeta[2:Nx,1:Ny-1]*(psi[2:Nx,2:Ny] - psi[2:Nx,0:Ny-2]) +
    zeta[0:Nx-2,1:Ny-1]*(psi[0:Nx-2,2:Ny] - psi[0:Nx-2,0:Ny-2]) +
    zeta[1:Nx-1,2:Ny]*(psi[2:Nx,2:Ny] - psi[0:Nx-2,2:Ny]) -
    zeta[1:Nx-1,0:Ny-2]*(psi[2:Nx,0:Ny-2] - psi[0:Nx-2,0:Ny-2]))*one_dA_fourth
    return J