# Yifan Du             dyifan1@jhu.edu
# Tamer A. Zaki        t.zaki@jhu.edu
# Johns Hokpins University
# US patent submitted 03/08/2021

# EDNN solver of heat equation

import numpy as np
import tensorflow as tf
from   tensorflow import keras
import matplotlib.pyplot as plt
import time
import sys
from pdb import set_trace as keyboard
from ednn import EvolutionalDNN
from marching_schemes import * 
from rhs import * 
from tensorflow.keras.optimizers.legacy import Adam, SGD

def u_analytique(x,y):
    return (x**3-x)*(y**3-y)

def HeatData(x,y):
    funValue = 6*x*y*(x**2+ y**2 - 2)
    return funValue

def v_cycle(nx,ny,nb_grille,u,f,level=1):
    if (level==nb_grille):
        u,res = jacobi_relax(level,nx,ny,u,f,iters=50,pre=True)
        return u,res

    # Etape 1 : Relaxation Au=f
    u,res = jacobi_relax(level,nx,ny,u,f,iters=2,pre=True)

    # Etape 2 : Restriction sur grille grossière
    res_coarse = restriction(nx//2,ny//2,res)

    # Etape 3 : Résolution de A e_coarse = res_coarse (récursif)
    e_coarse = np.zeros_like(res_coarse)
    e_coarse,res_coarse = v_cycle(nx//2,ny//2,nb_grille,e_coarse,res_coarse,level+1)

    # Etape 4 : Prolongation (Interpolation) sur la grille fine
    u+= prolongation(nx//2,ny//2,e_coarse)

    # Etape 5 : Relaxation sur la grille
    u,res = jacobi_relax(level,nx,ny,u,f,iters=1,pre=False)

    return u,res

def jacobi_relax(level,nx,ny,u,f,iters=1,pre=False):
    dx=1.0/nx
    dy=1.0/ny
    kx=1/dx**2
    ky=1/dy**2
    kp=1/(2*(kx+ky))

    # Dirichlet
    u[0,:] = -u[1,:]
    u[-1,:] = -u[-2,:]
    u[:, 0] = -u[:, 1]
    u[:,-1] = -u[:,-2]

    for i in range(iters):
        u[1:nx+1,1:ny+1] = kp*(kx*(u[2:nx+2,1:ny+1] + u[0:nx,1:ny+1])
                             + ky*(u[1:nx+1,2:ny+2] + u[1:nx+1,0:ny])
                             - f[1:nx+1,1:ny+1])
        u[0,:] = -u[1,:]
        u[-1,:] = -u[-2,:]
        u[:, 0] = -u[:, 1]
        u[:,-1] = -u[:,-2]

    res=np.zeros([nx+2,ny+2])
    res[1:nx+1,1:ny+1]=f[1:nx+1,1:ny+1]-((kx*(u[2:nx+2,1:ny+1]+u[0:nx,1:ny+1])
                                       + ky*(u[1:nx+1,2:ny+2]+u[1:nx+1,0:ny])
                                       - 2.0*(kx+ky)*u[1:nx+1,1:ny+1]))
    return u,res
def prolongation(nx,ny,v):
    """
    Interpolation de V sur la grille fine
    """
    v_fine = np.zeros([2*nx+2,2*ny+2])

    # Interpolation linéaire
    for i in range(1,nx+1):
        for j in range(1,ny+1):
            v_fine[2*i-1,2*j-1] = v[i,j]
            v_fine[2*i  ,2*j-1] = 0.5*(v[i-1,j-1]+v[i,j-1])
            v_fine[2*i-1,2*j  ] = 0.5*(v[i-1,j-1]+v[i-1,j])
            v_fine[2*i  ,2*j  ] = 0.25*(v[i-1,j-1]+v[i,j-1]+v[i-1,j]+v[i,j])

    # Interpolation 
    # for i in range(1,nx+1):
    #   for j in range(1,ny+1):
    #       v_fine[2*i-1,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j-1])+0.0625*v[i-1,j-1]
    #       v_fine[2*i  ,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j-1])+0.0625*v[i+1,j-1]
    #       v_fine[2*i-1,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j+1])+0.0625*v[i-1,j+1]
    #       v_fine[2*i  ,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j+1])+0.0625*v[i+1,j+1]
    return v_fine
def restriction(nx,ny,v):
    """
    Restriction de V sur la grille grossière
    """
    v_coarse = np.zeros([nx+2,ny+2])

    for i in range(1,nx+1):
        for j in range(1,ny+1):
            v_coarse[i,j]=(1/16)*(v[2*i-1,2*j-1]+v[2*i-1,2*j+1]+v[2*i+1,2*j-1]+v[2*i+1,2*j+1])+(1/8)*(v[2*i,2*j-1]+v[2*i,2*j+1]+v[2*i-1,2*j]+v[2*i+1,2*j])+(1/4)*(v[2*i,2*j])

    # for i in range(1,nx+1):
    #   for j in range(1,ny+1):
    #       v_coarse[i,j]=0.25*(v[2*i-1,2*j-1]+v[2*i,2*j-1]+v[2*i-1,2*j]+v[2*i,2*j])
    return v_coarse

def main():
    # -----------------------------------------------------------------------------
    # Parameters for simulation configuration
    # -----------------------------------------------------------------------------
    # NN and solution directory
    case_name = "HeatNN/"
    # Numer of collocation points
    MAX_CYCLES = 10 # Nombres de V cycles
    NB_LEVELS = 8 # Nombres de grilles
    Nx = 2**(NB_LEVELS-1)
    Ny = 2**(NB_LEVELS-1)
    EPS = 1e-10

    dx = 1/Nx
    dy = 1/Ny

    # if Initial == True, train the neural network for initial condition
    # if Initial == False, march the initial network stored in case_name
    if sys.argv[1] == '0':
        Initial = True
    elif sys.argv[1] == '1':
        Initial = False
    else:
        sys.exit("Wrong flag specified")
    # Physical domain
    x1 = - np.pi
    x2 =   np.pi
    y1 = - np.pi
    y2 =   np.pi
    # Other parameters
    nu = 1
    Nt = 1000
    dt = 1e-3
    tot_eps = 10
     
    # ------------------------------------------------------------------------------
    # Generate the collocation points and initial condition array
    # ------------------------------------------------------------------------------
    x  = np.linspace(x1,x2,num=Nx, dtype=np.float32)
    y  = np.linspace(y1,y2,num=Ny, dtype=np.float32)
    X,Y = np.meshgrid(x,y,indexing = 'ij')
    Xi = X[1:-1,1:-1]
    Yi = Y[1:-1,1:-1]
    #Initial condition
    usq = HeatData(Xi,Yi)
    u = usq.reshape((Nx-2)*(Ny-2),1)
    Input = np.concatenate((X.reshape((Nx)*(Ny),-1),Y.reshape((Nx)*(Ny),-1)),axis = 1)
    InputInterior = np.concatenate((Xi.reshape((Nx-2)*(Ny-2),-1),Yi.reshape((Nx-2)*(Ny-2),-1)),axis = 1)
    InitInterior = u.reshape((Nx-2)*(Ny-2),-1)
    
    Index = np.arange(Nx*Ny).reshape(Nx,Ny)
    
    Index = np.arange(Nx*Ny).reshape(Nx,Ny)
    IE = (0.0 * Index + Index[-1,:].reshape(1,Ny)).astype(int).reshape((Nx)*(Ny),-1)
    IW = (0.0 * Index + Index[0,:].reshape(1,Ny)).astype(int).reshape((Nx)*(Ny),-1)
    IN = (0.0 * Index + Index[:,-1].reshape(Nx,1)).astype(int).reshape((Nx)*(Ny),-1)
    IS = (0.0 * Index + Index[:,0].reshape(Nx,1)).astype(int).reshape((Nx)*(Ny),-1)
    BI = np.concatenate((IE,IW,IN,IS),axis = 1)
    
    #Extract the index of boundary points for the enforcement of B.C. 
    IEInterior = (0.0 * Index[1:-1,1:-1] + Index[-1,1:-1].reshape(1,Ny-2)).astype(int).reshape((Nx-2)*(Ny-2),-1)
    IWInterior = (0.0 * Index[1:-1,1:-1] + Index[0,1:-1].reshape(1,Ny-2)).astype(int).reshape((Nx-2)*(Ny-2),-1)
    INInterior = (0.0 * Index[1:-1,1:-1] + Index[1:-1,-1].reshape(Nx-2,1)).astype(int).reshape((Nx-2)*(Ny-2),-1)
    ISInterior = (0.0 * Index[1:-1,1:-1] + Index[1:-1,0].reshape(Nx-2,1)).astype(int).reshape((Nx-2)*(Ny-2),-1)
    BIInterior = np.concatenate((IEInterior,IWInterior,INInterior,ISInterior),axis = 1)
    
    
    # ------------------------------------------------------------------------------
    # Mutligrid
    # ------------------------------------------------------------------------------
    u = np.zeros([Nx+2,Ny+2])
    f = np.zeros([Nx+2,Ny+2])
    u_real = np.zeros([Nx+2,Ny+2]) 
    
    u_real[1:Nx+1,1:Ny+1] = u_analytique(X,Y)
    f[1:Nx+1,1:Ny+1] = HeatData(X,Y)

    try: 
        nrestart = int(np.genfromtxt(case_name + 'nrestart'))
    except OSError: 
        nrestart = 0
    
    # -----------------------------------------------------------------------------
    # Initialize EDNN
    # -----------------------------------------------------------------------------
    lr = keras.optimizers.schedules.ExponentialDecay(1e-3, 10000000, 0.9)
    layers  = [2] + 4*[20] + [1]
    
    EDNN = EvolutionalDNN(layers,
                             rhs = rhs_2d_heat_eqs, 
                             marching_method = Runge_Kutta,
                             dest=case_name,activation = 'tanh',
                             optimizer=Adam(lr),
                             eq_params=[nu],
                             restore=True)
    print('Learning rate:', EDNN.optimizer._decayed_lr(tf.float32))
    
    
    
    
    if Initial: 
        t0 = time.time()
        # Train the initial condition tot_eps epochs, 
        for i in range(tot_eps):
            InputInteriorBoundary = Input[BIInterior]
            EDNN.train(InputInterior, InputInteriorBoundary, InitInterior, epochs=1,
                   batch_size=100, verbose=False, timer=False)
        # Evaluate and output the initial condition 
        InputBoundary = tf.convert_to_tensor(Input[BI])
        Input = tf.convert_to_tensor(Input)
        [U] = EDNN.output(Input,InputBoundary)
        U = U.numpy().reshape((Nx,Ny))
        X.dump(case_name+'X')
        Y.dump(case_name+'Y')
        U.dump(case_name+'U')
        print(U.shape)
        for i in range(1,MAX_CYCLES+1):
            u,res = v_cycle(Nx,Ny,NB_LEVELS,U,f)
            residual = np.max(np.abs(res))
            if residual<EPS:
                break
            erreur = u_real[1:Nx+1,1:Ny+1]-U[1:Nx+1,1:Ny+1]
            print('Cycle :', i, ", Erreur :", residual)
        print('Temps :', time.time()-t0)
        erreur = u_real[1:Nx+1,1:Ny+1]-U[1:Nx+1,1:Ny+1]
        print('Erreur Finale :', np.max(np.abs(erreur)))
    
    else:
        InputInteriorBoundary = tf.convert_to_tensor(Input[BIInterior])
        InputBoundary = tf.convert_to_tensor(Input[BI])
        InputInterior = tf.convert_to_tensor(InputInterior)
        Input = tf.convert_to_tensor(Input)
    
        nbatch = 1 * 63
        params_marching = [dt,nbatch]
        # March the EDNN class till Nt time steps. 
        for n in range(nrestart,Nt):
            print('time step', n)
            EDNN.Marching(InputInterior,InputInteriorBoundary,params_marching)
            [Uh] = EDNN.output(Input,InputBoundary)
            # The solution field is stored every time step. 
            U = Uh.numpy().reshape((Nx, Ny))
            X.dump(case_name+'X'+str(n))
            Y.dump(case_name+'Y'+str(n))
            U.dump(case_name+'U'+str(n))
            EDNN.save_NN()
            
            np.savetxt(case_name+'nrestart',np.array([n]))
if __name__ == "__main__":
    main()




