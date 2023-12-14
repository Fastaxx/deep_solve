import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import timeit
import sys
from tqdm import tqdm

N_POINTS = 41
SIZE = 1.0
DL = SIZE/(N_POINTS-1)
VISCOSITE = 1e-2
TIME_STEP = 0.01
N_ITER = 10000
U_HAUT = -1
DENSITE = 1
PRESSURE_ITER = 10

def main():
	x=np.linspace(0.0,SIZE, N_POINTS)
	y=np.linspace(0.0,SIZE, N_POINTS)
	X,Y = np.meshgrid(x,y)
	u_prev = np.zeros_like(X)
	v_prev = np.zeros_like(X)
	p_prev = np.zeros_like(X)

	def central_difference_x(f):
		diff = np.zeros_like(f)
		diff[1:-1,1:-1] = (f[2:,1:-1] - f[0:-2,1:-1])/(2*DL)
		return diff
	def central_difference_y(f):
		diff = np.zeros_like(f)
		diff[1:-1,1:-1] = (f[1:-1,2:] - f[1:-1,0:-2])/(2*DL)
		return diff
	def laplacian(f):
		diff = np.zeros_like(f)
		diff[1:-1,1:-1]=(f[1:-1,0:-2] + f[0:-2,1:-1] - 4*f[1:-1,1:-1] + f[1:-1,2:] + f[2,1:-1])/(DL**2)
		return diff

	max_time_step = 0.5*DL**2 / VISCOSITE
	print('Max Time Step :', max_time_step)	

	for _ in tqdm(range(N_ITER)):
		du_dx_prev = central_difference_x(u_prev)
		du_dy_prev = central_difference_y(u_prev)
		dv_dx_prev = central_difference_x(v_prev)
		dv_dy_prev = central_difference_y(v_prev)

		laplacien_u_prev = laplacian(u_prev)
		laplacien_v_prev = laplacian(v_prev)

		u_tent = (u_prev + TIME_STEP*(-(u_prev*du_dx_prev+v_prev*du_dy_prev) + VISCOSITE* laplacien_u_prev))
		v_tent = (v_prev + TIME_STEP*(-(u_prev*dv_dx_prev+v_prev*dv_dy_prev) + VISCOSITE* laplacien_v_prev))

		u_tent[0,:] = 0
		u_tent[:,0] = 0
		u_tent[:,-1] = 0
		u_tent[-1,:] = U_HAUT
		v_tent[0,:] = 0
		v_tent[:,0] = 0
		v_tent[:,-1] = 0
		v_tent[-1,:] = 0

		du_dx_tent = central_difference_x(u_tent)
		dv_dy_tent = central_difference_y(v_tent)

		pressure_corr = (DENSITE/TIME_STEP * (du_dx_tent + dv_dy_tent))

		for i in range(PRESSURE_ITER):
			p_next = np.zeros_like(p_prev)
			p_next[1:-1,1:-1] = 1/4*(p_prev[1:-1,0:-2]+p_prev[0:-2,1:-1]+p_prev[1:-1,2:]+p_prev[2:,1:-1]-DL**2 * pressure_corr[1:-1,1:-1])

			p_next[:,-1]= p_next[:,-2]
			p_next[0,:]=p_next[1,:]
			p_next[:,0]=p_next[:,1]
			p_next[-1,:]=0

			p_prev = p_next

		dp_dx_next = central_difference_x(p_next)
		dp_dy_next = central_difference_y(p_next)

		u_next = (u_tent - TIME_STEP/DENSITE * dp_dx_next)
		v_next = (v_tent - TIME_STEP/DENSITE * dp_dy_next)

		u_next[0,:]=0
		u_next[:,0] = 0
		u_next[:,-1] = 0
		u_next[-1,:] = U_HAUT
		v_next[0,:] = 0
		v_next[:,0] = 0
		v_next[:,-1] = 0
		v_next[-1,:] = 0

		u_prev = u_next
		v_prev = v_next
		p_prev = p_next

		plt.contourf(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], cmap="coolwarm")
		plt.colorbar()

		plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
		plt.xlim((0, 1))
		plt.ylim((0, 1))

		plt.draw()
		plt.pause(0.05)
		plt.clf()

	plt.style.use('dark_background')
	plt.figure()
	plt.contourf(X[::2, ::2], Y[::2, ::2], p_next[::2, ::2], cmap="coolwarm")
	plt.colorbar()

	plt.quiver(X[::2, ::2], Y[::2, ::2], u_next[::2, ::2], v_next[::2, ::2], color="black")
	plt.xlim((0, 1))
	plt.ylim((0, 1))
	plt.show()

if __name__ == "__main__":
	main()