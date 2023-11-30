import numpy as np
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

def u_analytique(x,y):
	return (x**3-x)*(y**3-y)

def source(x,y):
	return 6*x*y*(x**2+ y**2 - 2)

def jacobi_relax(level,nx,ny,u,f,iters=1,pre=False):
	dx=1.0/nx
	dy=1.0/ny
	kx=1/dx**2
	ky=1/dy**2
	kp=1/(2*(kx+ky))

	# Dirichlet
	u[0,:] = -u[ 1,:]
	u[-1,:] = -u[-2,:]
	u[:, 0] = -u[:, 1]
	u[:,-1] = -u[:,-2]

	for i in range(iters):
		u[1:nx+1,1:ny+1] = kp*(kx*(u[2:nx+2,1:ny+1] + u[0:nx,1:ny+1])
							 + ky*(u[1:nx+1,2:ny+2] + u[1:nx+1,0:ny])
							 - f[1:nx+1,1:ny+1])
		u[0,:] = 0
		u[-1,:] = 0
		u[:, 0] = 0
		u[:,-1] = 0

	res=np.zeros([nx+2,ny+2])
	res[1:nx+1,1:ny+1]=f[1:nx+1,1:ny+1]-((kx*(u[2:nx+2,1:ny+1]+u[0:nx,1:ny+1])
                                       + ky*(u[1:nx+1,2:ny+2]+u[1:nx+1,0:ny])
                                       - 2.0*(kx+ky)*u[1:nx+1,1:ny+1]))
	return u,res

def restriction(nx,ny,v):
	"""
	Restriction de V sur la grille grossière
	"""
	v_coarse = np.zeros([nx+2,ny+2])

	for i in range(1,nx+1):
		for j in range(1,ny+1):
			v_coarse[i,j]=0.25*(v[2*i-1,2*j-1]+v[2*i,2*j-1]+v[2*i-1,2*j]+v[2*i,2*j])
	return v_coarse

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
	# 	for j in range(1,ny+1):
	# 		v_fine[2*i-1,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j-1])+0.0625*v[i-1,j-1]
	# 		v_fine[2*i  ,2*j-1] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j-1])+0.0625*v[i+1,j-1]
	# 		v_fine[2*i-1,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i-1,j]+v[i,j+1])+0.0625*v[i-1,j+1]
	# 		v_fine[2*i  ,2*j  ] = 0.5625*v[i,j]+0.1875*(v[i+1,j]+v[i,j+1])+0.0625*v[i+1,j+1]
	return v_fine

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


MAX_CYCLES = 50 # Nombres de V cycles
NB_LEVELS = 8 # Nombres de grilles
NX = 2**(NB_LEVELS-1)
NY = 2**(NB_LEVELS-1)
EPS = 1e-10

u = np.zeros([NX+2,NY+2])
f = np.zeros([NX+2,NY+2])
u_real = np.zeros([NX+2,NY+2]) 

dx = 1/NX
dy = 1/NY

x = np.linspace(0.5*dx, 1-0.5*dx,NX)
y = np.linspace(0.5*dy, 1-0.5*dy,NY)

X,Y = np.meshgrid(x,y,indexing='ij')

u_real[1:NX+1,1:NY+1] = u_analytique(X,Y)
f[1:NX+1,1:NY+1] = source(X,Y)

print("Start")
t0 = time.time()

# V-cycle

for i in range(1,MAX_CYCLES+1):
	u,res = v_cycle(NX,NY,NB_LEVELS,u,f)
	residual = np.max(np.abs(res))
	if residual<EPS:
		break
	erreur = u_real[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1]
	print('Cycle :', i, ", Erreur :", residual)

print('Temps :', time.time()-t0)
erreur = u_real[1:NX+1,1:NY+1]-u[1:NX+1,1:NY+1]
print('Erreur Finale :', np.max(np.abs(erreur)))

plt.style.use('dark_background')
plt.figure()
plt.contourf(erreur)
plt.colorbar()
plt.show()

