import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

# this file contains right scaling for FE/FD problem
# if we change only Nx and Ny we converge to the continuous solution with the same regularisation
# the variable of input flux_bc and penalization (heat loss) are chosen to get the same results as Arthur for 4x4 system


## STIFFNESS MATRIX by Finite Differences
def get_K_FD(Nx, Ny, dx, dy, alpha):
    """ Returns stiffness matrix K """
    N = Nx * Ny
    K = sp.sparse.lil_matrix((N, N))
    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j
            if i < Nx-1:
                right = idx + Ny
                K[idx, idx] += alpha / dx**2 # KKK the square is the right scaling
                K[idx, right] -= alpha / dx**2
                K[right, idx] -= alpha / dx**2
                K[right, right] += alpha / dx**2

            if j < Ny-1:
                top = idx + 1
                K[idx, idx] += alpha / dy**2
                K[idx, top] -= alpha / dy**2
                K[top, idx] -= alpha / dy**2
                K[top, top] += alpha / dy**2
    return K

## STIFFNESS MATRIX by Finite Elements
def get_K_FE(Nx, Ny, dx, dy, alpha):
    """ Returns stiffness matrix K """
    N = Nx * Ny
    # elementary matrix of int gradN_i*gradN_j
    K11 = 2*dy/dx + 2*dx/dy
    K12 =   dy/dx - 2*dx/dy
    K13 = - dy/dx -   dx/dy
    K14 =-2*dy/dx +   dx/dy
    M_elem = np.array([ [K11, K12, K13, K14],
                        [K12, K11, K14, K13],
                        [K13, K14, K11, K12],
                        [K14, K13, K12, K11]])
    M_elem *= alpha / (dx*dy)/6 # the same scaling as for finite differencies method
    #M_elem = np.array([[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]])
    # full rigidity matrix
    K = sp.sparse.lil_matrix((N, N))
    # number of elements in x
    size_el_x = Nx-1
    # number of elements in y
    size_el_y = Ny-1
    # number of elements in total
    size_el = size_el_x * size_el_y

    # number of nodes in single element
    size_fem = 4
    common_shift = np.zeros((size_el), dtype=int)
    #common_shift = sp.sparse.lil_matrix((size_el),dtype=int)
    common_shift = np.arange(size_el) + np.kron(np.array(range(size_el_x)),np.zeros(size_el_y)+1)
    for elem in range(size_el):
        lls = common_shift[elem]
        loc_ind = np.array([lls,lls+1, lls+size_el_y+2, lls+size_el_y+1],dtype=int)
        for i in range(size_fem):
            ki = loc_ind[i]
            for j in range(size_fem):
                kj = loc_ind[j]
                if kj<ki : K[ki,kj] += M_elem[i,j]
                elif kj==ki : K[ki,kj] += M_elem[i,j]/2
                else: pass
    # we are using the K-matrix symetry
    K += K.T
    return K



def flux(Nx, Ny, prc, flux_bc, FE = False): 
    """ Returns force vector f """
    #f = sp.sparse.csr_matrix((Nx*Ny,1))  # ddl number
    f = np.zeros(Nx*Ny,dtype=float)  # ddl number
    miny = round((1-prc) * Ny) # pourcentage de la zone a droite où le flux est present
    for i in range(miny, Ny):
        f[(Nx - 1) * Ny + i] = flux_bc
    if FE==True : # correction for flux in FE
        f[-1]= flux_bc/2
        f[(Nx - 1) * Ny + miny -1 ]= flux_bc/2
    return f


## QUANTUM PARAMETERS
nx = 5                            # number of qubits on x axis
ny = 4                            # number of qubits on y axis
nqb = nx + ny                     # number of qubits for circuits
FE = False


print("This is finite elements: ",FE)
## Using finite diff or finite elements
get_K = get_K_FD
if FE==True : get_K = get_K_FE

print(f"Number of Qbits = {nqb}")



## SPACE PARAMETERS
Lx = 1.0  # Length of the square domain
Ly = 1.0  # Length of the square domain
Nx = 2 ** nx  # Number of points in x-direction
Ny = 2 ** ny  # Number of points in y-direction
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
dx = Lx / (Nx-1.)
dy = Ly / (Ny-1.)


## STIFFNESS MATRIX
alpha = 1.0  # Thermal diffusivity
pen = 0.1*9/4 # to get the same result as for 4 qbits simulation before
print("Calculating Rigidity Matrix")
tt0 = time.time()
Ksp = get_K_FD(Nx, Ny, dx, dy, alpha)
KspFE = get_K_FE(Nx, Ny, dx, dy, alpha)
tt1 = time.time()

Ksp += pen  * sp.sparse.eye(Nx*Ny, dtype=np.int8) # regularization
KspFE += pen  * sp.sparse.eye(Nx*Ny, dtype=np.int8) # regularization
#K = Ksp.toarray()




## BOUNDARY CONDITIONS
flux_bc = 3/np.sqrt(2)/4  #KKK flux de chaleur, the value is chosen to get the same resuls for 4 qbits
prc = 0.5

f = flux(Nx, Ny, prc, flux_bc/dx) #KKK flux should be rescaled by the surface it is applied
fFE = flux(Nx, Ny, prc, flux_bc/dx, FE= True) #KKK flux should be rescaled by the surface it is applied



## CALCUL CHAMPS
print("Solving Linear Algebra")

tt2 = time.time()
ucl = sp.sparse.linalg.spsolve(Ksp.tocsr(), f).real
uclFE = sp.sparse.linalg.spsolve(KspFE.tocsr(), fFE).real
tt3 = time.time()
print(f"Rigidity computation time={round(tt1-tt0,3)}")
print(f"Linalg execution time {round(tt3-tt2,3)}")
print(f"Full execution time {round(tt3-tt0,3)}")

# right ordering for imshow plots
ucl = ucl.reshape((Nx, Ny)).T[::-1]
uclFE = uclFE.reshape((Nx, Ny)).T[::-1]

plt.imshow(ucl - uclFE, interpolation="nearest", origin="upper")
plt.colorbar()
plt.show()

ucl_diag = np.fliplr(ucl).diagonal()
ucl_diagFE = np.fliplr(uclFE).diagonal()

coord = np.linspace(0, 1, min(Nx,Ny))
print(ucl.shape)
plt.plot( coord, ucl_diag[::-1],'o-', linewidth=2.0, label='diag_FD')
plt.plot( coord, ucl[0:min(Nx,Ny),Nx-1],'x-', linewidth=2.0, label='vertical_FD')
plt.plot( coord, ucl_diagFE[::-1],'o-', linewidth=2.0, label='diag_FE')
plt.plot( coord, uclFE[0:min(Nx,Ny),Nx-1],'x-', linewidth=2.0, label='vertical_FE')


print("###########FD")
print("min=", ucl.min())
print("max=", ucl.max())

ratio = ucl.max()/ucl.min()	
print("ratio =", ratio)

print("###########FE")
print("min=", uclFE.min())
print("max=", uclFE.max())

ratioFE = uclFE.max()/uclFE.min()	
print("ratio =", ratioFE)

plt.legend(loc='best')
plt.show()