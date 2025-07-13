import numpy as np
import matplotlib.pyplot as plt
import math

Nx,Ny = 0
dx,dy = 0

def partial_x(field):
    F = np.zeros((Nx,Ny))
    F[1:Nx-1,1:Ny-1] = (field[2:Nx,1:Ny-1]-field[0:Nx-2,1:Ny-1])*(1/2/dx)
    return F

def partial_y(field):
    F = np.zeros((Nx,Ny))
    F[1:Nx-1,1:Ny-1] = (field[1:Nx-1,2:Ny]-field[1:Nx-1,0:Ny-2])*(1/2/dy)
    return F

def laplacian(field):
    laplace = np.zeros((Nx,Ny))
    laplace[1:Nx-1,1:Ny-1] = (1/dx/dx)*(field[2:Nx,1:Ny-1]-2*field[1:Nx-1,1:Ny-1]+field[0:Nx-2,1:Ny-1])+(1/dy/dy)*(field[1:Nx-1,2:Ny]-2*field[1:Nx-1,1:Ny-1]+field[1:Nx-1,0:Ny-2])
    return laplace