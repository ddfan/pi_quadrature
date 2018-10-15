import numpy as np
from numpy.polynomial.hermite import hermgauss
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import norm
import copy
import timeit

start = timeit.default_timer()

Q = 0.1
R = 1
sigma = .2
print(sigma)
dt = 0.1
N = 42
num_plots=4
num_plot_cols = 2
xlim = np.array([-2, 2])
degree = 20
num_x0 = 40
num_bins = 30
xi_scale = 0.3

def q(x):
    return Q*np.sum(1/2*(x-np.array([-1,-1]))**2,axis=-1)*np.sum(1/2*(x-np.array([1,1]))**2, axis=-1)

# dx = (-x^3 + x +u )dt + sigma dw
def b(x):
    x1=x[..., 0]
    x2=x[..., 1]
    y1 = np.cos(x1 * x2) * np.sin(x1 * x2) * x2 - 1/6*x1**3
    y2 = np.cos(x1 * x2) * np.sin(x1 * x2) * x1 - 1/6*x2**3
    return np.stack([y1,y2],axis=-1)
def bx(x):
    x1 = x[..., 0]
    x2 = x[..., 1]
    y1 = (np.cos(x1 * x2)**2 - np.sin(x1 * x2)**2) * x2**2 - 1 / 2 * x1 ** 2
    y2 = (np.cos(x1 * x2)**2 - np.sin(x1 * x2)**2) * x1**2 - 1 / 2 * x2 ** 2
    return y1+y2
def intb(x):
    x1 = x[..., 0]
    x2 = x[..., 1]
    return -1/2*np.cos(x1*x2)**2 - 1/24*(x1**4+x2**4)
def w_cost(x):
    V = q(x) + R / 2 * np.sum(b(x)**2, axis=-1) - (sigma ** 2 * R / 2) * bx(x)
    return np.exp(-V / (sigma ** 2 * R) * dt)
def w_N(x):
    return np.exp(-(q(x)-R*intb(x))/(sigma**2*R))

# specify weights
[xi_1d, alpha_1d] = hermgauss(degree)
alpha_1d = alpha_1d / np.sqrt(np.pi)
xi_1d = xi_scale * np.sqrt(2) * xi_1d
alpha_1d = alpha_1d / norm.pdf(xi_1d,0,xi_scale)

xi = np.zeros((degree**2, 2))
alpha = np.zeros(degree**2)
for i in range(degree):
    for j in range(degree):
        xi[i*degree+j, :] = np.array([xi_1d[i], xi_1d[j]]).T
        alpha[i*degree+j] = alpha_1d[i]*alpha_1d[j]

print('quadrature limits: ', np.min(xi),np.max(xi))

Xi = np.array([xi-xi[i] for i in range(degree**2)])
gamma = alpha * w_cost(xi)
Gamma = np.diag(gamma)
gamma_N = alpha * w_N(xi)
# p(x_{t+1}|x_t)
Phi = np.prod(norm.pdf(Xi, 0, sigma * np.sqrt(dt)), axis=-1)
Phi2 = np.matmul(Phi, Gamma)
print('largest eigenvalue:', np.amax(np.linalg.eig(Phi2)[0]))
Phi2 = Phi2 / np.amax(np.linalg.eig(Phi2)[0])
# Phi3 = copy.copy(Phi2)

x0_1d = np.linspace(xlim[0], xlim[1], num_bins)


##################################################
#plot nu and uncontrolled particles

# # plotting nu first
# test_arr = np.zeros((num_bins,num_bins))
# for i in range(num_bins):
#     for j in range(num_bins):
#         x0 = np.array([x0_1d[i], x0_1d[j]])
#         test_arr[i,j]=-intb(x0)
# im=plt.imshow(test_arr,cmap=plt.cm.get_cmap('jet'),extent=[xlim[0], xlim[1], xlim[0], xlim[1]],origin='lower')
# im.set_interpolation('bilinear')

# #plot trajectory paths of some particles
# x_arr = np.random.rand(num_x0, 2, N)*(xlim[1]-xlim[0])+xlim[0]

# for n in range(N - 1):
#     x_arr[:,:,n+1] = x_arr[:,:,n] + b(x_arr[:,:,n])*dt + sigma*np.sqrt(dt)*np.random.randn(num_x0,2)

# for j in range(num_x0):
#     plt.plot(x_arr[j,0,:],x_arr[j,1,:],'k',alpha=0.3)
# for j in range(num_x0):
#     # plt.plot(x_arr[j, 0, 0], x_arr[j, 1, 0], 'w.',alpha=0.7)
#     plt.plot(x_arr[j, 0, -1], x_arr[j, 1, -1], 'w.', alpha=0.7)

# divider = make_axes_locatable(plt.gca())
# cax = divider.append_axes("right", size="5%", pad=0.05)
# plt.colorbar(im, cax=cax)


# plt.show()
# quit()


##############################

# # plot f(t,x) on an even grid
f_arr = np.zeros((num_bins, num_bins, N-1))
u_arr = np.zeros((num_bins, num_bins, N-1,2))
for n in range(N-1):
    print(n)
    Phi3 = np.linalg.matrix_power(Phi2, N-1-n)
    Phi4 = np.dot(gamma_N.T,Phi3).T

    for i in range(num_bins):
        for j in range(num_bins):
            x0 = np.array([x0_1d[i], x0_1d[j]])
            Phi0 = Gamma @ np.prod(norm.pdf(xi, x0, sigma * np.sqrt(dt)), axis=-1)
            f_arr[i,j,n] = (Phi4.T @ Phi0)
            # u_arr[i, j, n,:] = -b(x0) + sigma ** 2 * np.sum(np.diag(Phi4 * Phi0) @ (xi - x0), axis=0) / (
            #             sigma ** 2 * dt) / f_arr[i, j, n]

fig, ax = plt.subplots(np.int16(np.ceil(num_plots/num_plot_cols)),num_plot_cols)
# fig = plt.figure()
ax= ax.flatten()
fig.set_size_inches(6, 5)

for i in range(num_plots-1,-1,-1):

    idx = int((i+1)*np.round(N/num_plots))
    # idx=N-2
    fig.sca(ax[i])
    im = plt.imshow(-sigma**2/2*np.log(f_arr[:, :, idx]), cmap=plt.cm.get_cmap('jet'), vmin=(-sigma**2/2*np.log(f_arr)).min(), vmax=(-sigma**2/2*np.log(f_arr)).max()*3/4,
                    extent=[xlim[0], xlim[1], xlim[0], xlim[1]],origin='lower')
    # im = plt.imshow(u_arr[:, :, idx], cmap=plt.cm.get_cmap('jet_r'),
                    # vmin=u_arr.min(), vmax=u_arr.max(), extent=[xlim[0], xlim[1], xlim[0], xlim[1]], origin='lower')
    im.set_interpolation('bilinear')
    # plt.xlabel('t='+ '{0:.2f}'.format(i*dt))
    plt.axis('off')
    plt.title('t=' + '{0:.2f}'.format(idx * dt))

    divider = make_axes_locatable(fig.gca())
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)

#
# ################################
#
# #plot distribution of states p(t,x)
# f_arr = np.zeros((num_x0, num_x0, N-1))
# u_arr = np.zeros((num_x0, num_x0, 2, N-1))
# x_arr = np.zeros((num_x0, num_x0, 2, N))
# for i in range(num_x0):
#     for j in range(num_x0):
#         x_arr[i,j,:,0] = [x0_1d[i],x0_1d[j]]
#
# for n in range(N - 1):
#     print(n)
#     Phi3 = np.linalg.matrix_power(Phi2, N - 1 - n)
#     Phi4 = np.dot(gamma_N.T,Phi3).T
#
#     for i in range(num_x0):
#         for j in range(num_x0):
#             x0 = x_arr[i,j,:,n]
#             Phi0 = Gamma @ np.prod(norm.pdf(xi, x0, sigma * np.sqrt(dt)), axis=-1)
#             f_arr[i, j, n] = (Phi4.T).dot(Phi0)
#             u_arr[i, j, :, n] = -b(x0) + sigma**2 * np.sum(np.diag(Phi4*Phi0) @ (xi-x0),axis=0) / (sigma**2 * dt) / f_arr[i,j, n]
#     x_arr[:,:,:,n+1] = x_arr[:,:,:,n] + (b(x_arr[:,:,:,n]) + u_arr[:,:,:,n])*dt + sigma*np.sqrt(dt)*np.random.randn(num_x0,num_x0,2)
#
# fig=plt.figure()
# fig.set_size_inches(5, 6)
# maxbinval=1
# for i in range(num_plots-1,-1,-1):
#     print(i)
#     ax1 = plt.subplot(np.ceil(num_plots/num_plot_cols),num_plot_cols,i+1)
#     idx = int(i*np.round(N/num_plots))
#     H,xedges,yedges = np.histogram2d(x_arr[:,:,0,idx].flatten(),x_arr[:,:,1,idx].flatten(),
#                                      normed=False, weights = np.ones(num_x0**2)/float(num_x0**2), bins=num_bins,
#                                      range=[[xlim[0], xlim[1]], [xlim[0], xlim[1]]])
#     if i is num_plots-1:
#         maxbinval=np.max(H)
#     im = plt.imshow(H, cmap=plt.cm.RdBu, vmin=0, vmax=maxbinval,
#                     extent=[xlim[0], xlim[1], xlim[1], xlim[0]],origin='lower')
#     plt.axis('off')
#     plt.title('t=' + '{0:.2f}'.format(idx * dt))
#     im.set_interpolation('bilinear')
#     # cb = fig.colorbar(im)
#
# # plt.tight_layout()



################################

#plot trajectory paths of some particles
f_arr = np.zeros((num_x0, N-1))
u_arr = np.zeros((num_x0, 2, N-1))
x_arr = np.random.rand(num_x0, 2, N)*(xlim[1]-xlim[0])+xlim[0]

for n in range(N - 1):
    print(n)
    Phi3 = np.linalg.matrix_power(Phi2, N - 1 - n)
    Phi4 = np.dot(gamma_N.T,Phi3).T

    for i in range(num_x0):
        x0 = x_arr[i,:,n]
        Phi0 = Gamma @ np.prod(norm.pdf(xi, x0, sigma * np.sqrt(dt)), axis=-1)
        f_arr[i, n] = (Phi4.T).dot(Phi0)
        u_arr[i, :, n] = -b(x0) + sigma**2 * np.sum(np.diag(Phi4*Phi0) @ (xi-x0),axis=0) / (sigma**2 * dt) / f_arr[i, n]
    x_arr[:,:,n+1] = x_arr[:,:,n] + (b(x_arr[:,:,n]) + u_arr[:,:,n])*dt + sigma*np.sqrt(dt)*np.random.randn(num_x0,2)

for i in range(num_plots-1,-1,-1):
    fig.sca(ax[i])
    idx = int((i+1)*np.round(N/num_plots))
    # idx=N-2
    for j in range(num_x0):
        plt.plot(x_arr[j,0,:idx],x_arr[j,1,:idx],'k',alpha=0.3)
    for j in range(num_x0):
        # plt.plot(x_arr[j, 0, 0], x_arr[j, 1, 0], 'w.',alpha=0.7)
        plt.plot(x_arr[j, 0, idx-1], x_arr[j, 1, idx-1], 'w.', alpha=0.7)



stop = timeit.default_timer()

print('Time: ', stop - start) 


plt.show()