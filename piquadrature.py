import numpy as np
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
from scipy.stats import norm

Q = 100
R = 10
sigma = 1/np.sqrt(R)
print('sigma: ', sigma)
dt = 0.005
N = 200
xlim=[-2, 2]
degree=200
xi_scale = 0.12

def q(x):
    return Q*(x-1)**2/2

# dx = (-x^3 + x +u )dt + sigma dw
def b(x):
    return -x**3 + x
def bx(x):
    return -3*x**2 + 1
def intb(x):
    return -1/4*x**4 + 1/2*x**2

def w_cost(x):
    V = q(x) + R / 2 * b(x) ** 2 - (sigma ** 2 * R / 2) * bx(x)
    return np.exp(-V / (sigma ** 2 * R) * dt)
def w_N(x):
    return np.exp(-(q(x)-R*intb(x))/(sigma**2*R))

def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)

# specify weights
[xi, alpha] = hermgauss(degree)
xi = np.array([xi]).T
alpha = np.array([alpha]).T / np.sqrt(np.pi)
xi = xi_scale * np.sqrt(2) * xi
alpha = alpha / norm.pdf(xi,0,xi_scale)
print('qudarature limits: ', np.min(xi),np.max(xi))
Xi = np.array([xi-xi[i] for i in range(degree)])[:,:,0]
gamma = alpha * w_cost(xi)
Gamma = np.diag(gamma[:,0])
# Gamma = np.diag(alpha[:,0])
gamma_N = alpha * w_N(xi)
Phi =norm.pdf(Xi,0,sigma * np.sqrt(dt))
Phi2 = np.matmul(Phi,Gamma)

num_x0=500
num_bins=200
x0_arr = np.linspace(xlim[0],xlim[1],num_x0)
f_arr = np.zeros((num_x0,N-1))
u_arr = np.zeros((num_x0,N-1))
b_arr = np.zeros(num_x0)
x_arr = np.zeros((num_x0,N))
x_bins = np.zeros((num_bins,N-1))
x_arr[:,0]=x0_arr

# Phi3 = np.linalg.matrix_power(Phi2, 1)
# Phi3=np.eye(degree)
# Phi4 = np.matmul(Phi3, gamma_N)
for n in range(N-1):
    print(n)
    x_bins[:,n] = np.histogram(x_arr[:, n],bins=num_bins, range=(xlim[0],xlim[1]))[0]/num_x0
    Phi3 = np.linalg.matrix_power(Phi2, N-1-n)
    Phi4 = np.dot(gamma_N.T,Phi3).T

    for k in range(num_x0):
        x0=x_arr[k,0]
        Phi0 = Gamma @ norm.pdf(xi, x0, sigma * np.sqrt(dt))
        f_arr[k, n] = (Phi4.T @ Phi0)[0, 0]
        u_arr[k, n] = -b(x0) + sigma ** 2 * ( (Phi4.T).dot(Phi0 * (xi - x0) / (sigma ** 2 * dt))[0, 0] ) / f_arr[k, n]

f_arr_particle = np.zeros((num_x0,N-1))
u_arr_particle = np.zeros((num_x0,N-1))
for n in range(N-1):
    print(n)
    x_bins[:,n] = np.histogram(x_arr[:, n],bins=num_bins, range=(xlim[0],xlim[1]))[0]/num_x0
    Phi3 = np.linalg.matrix_power(Phi2, N-1-n)
    Phi4 = np.dot(gamma_N.T,Phi3).T

    for k in range(num_x0):
        x0=x_arr[k,n]
        Phi0 = Gamma @ norm.pdf(xi, x0, sigma * np.sqrt(dt))
        f_arr_particle[k, n] = (Phi4.T @ Phi0)[0, 0]
        u_arr_particle[k, n] = -b(x0) + sigma ** 2 * ( (Phi4.T).dot(Phi0 * (xi - x0) / (sigma ** 2 * dt))[0, 0] ) / f_arr_particle[k, n]
    x_arr[:, n + 1] = np.sort( x_arr[:, n] + (b(x_arr[:, n]) + u_arr_particle[:, n]) * dt + sigma * np.sqrt(dt) * np.random.randn(num_x0))


b_arr[:] = b(x_arr[:,0])

ax=plt.subplot(311)
ax.set_prop_cycle('color', [plt.cm.viridis(j) for j in np.linspace(0, 1, N)])
plt.plot(x_arr[:,0],f_arr)
plt.ylabel('f(t,x)')
plt.xlim(xlim[0],xlim[1])
plt.title('Q=100')

ax=plt.subplot(312)
ax.set_prop_cycle('color', [plt.cm.viridis(j) for j in np.linspace(0, 1, N)])
# plt.plot(x_arr[:,0],b_arr,'r')
plt.plot(x_arr[:,0],u_arr)
plt.xlim(xlim[0],xlim[1])
plt.ylabel('u(t,x)')
ax=plt.subplot(313)
ax.set_prop_cycle('color', [plt.cm.viridis(j) for j in np.linspace(0, 1, N)])
xplot = np.linspace(xlim[0],xlim[1],num_bins)
plt.plot(xplot,x_bins)
p_check = np.exp(-2/sigma**2*(xplot**4/4-xplot**2/2))
p_check = p_check / np.sum(p_check)
# plt.plot(xplot,p_check,'red')
plt.xlim(xlim[0],xlim[1])
plt.ylabel('p(t,x)')
plt.xlabel('x')
plt.show()

