import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

# Finite difference and finit elements discretization of the heat equation
# comparaison of both FE to FD problems
# the variable of input flux_bc and penalization (heat loss) are chosen to get the same results


## STIFFNESS MATRIX by Finite Differences
def get_K_FD(Nx, Ny):
    """ Returns stiffness matrix K """
    N = Nx * Ny
    K = sp.sparse.lil_matrix((N, N))
    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j
            if i < Nx-1:
                right = idx + Ny
                K[idx, idx] += 6 
                K[idx, right] -= 6
                K[right, idx] -= 6
                K[right, right] += 6

            if j < Ny-1:
                top = idx + 1
                K[idx, idx] += 6
                K[idx, top] -= 6
                K[top, idx] -= 6
                K[top, top] += 6
    return K

## STIFFNESS MATRIX by Finite Elements
def get_K_FE(Nx, Ny):
    """ Returns stiffness matrix K """
    N = Nx * Ny
    # elementary matrix of int
    M_elem = np.array([[4,-1,-2,-1],[-1,4,-1,-2],[-2,-1,4,-1],[-1,-2,-1,4]])
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
ny = nx                           # number of qubits on y axis
assert nx==ny                      # this file works only for square systems
nqb = nx + ny                     # number of qubits for circuits

print(f"Number of Qbits = {nqb}")



## SPACE PARAMETERS

Nx = 2 ** nx  # Number of points in x-direction
Ny = 2 ** ny  # Number of points in y-direction
x = np.linspace(0, 1, Nx)
y = np.linspace(0, 1, Ny)



## STIFFNESS MATRIX
pen = 0.01 # heat loss by convection parameter
print("Calculating Rigidity Matrix")

Ksp = get_K_FD(Nx, Ny)
KspFE = get_K_FE(Nx, Ny)

Ksp += pen  * sp.sparse.eye(Nx*Ny, dtype=np.int8) # regularization
KspFE += pen  * sp.sparse.eye(Nx*Ny, dtype=np.int8) # regularization
#K = Ksp.toarray()




## BOUNDARY CONDITIONS
flux_bc = 1  #incoming heat flux
prc = 0.5 # right edge ration to be heated be flux_bc

f = flux(Nx, Ny, prc, flux_bc) 
fFE = flux(Nx, Ny, prc, flux_bc, FE= True) 


## CALCUL CHAMPS
print("Solving Linear Algebra")

ucl = sp.sparse.linalg.spsolve(Ksp.tocsr(), f).real
uclFE = sp.sparse.linalg.spsolve(KspFE.tocsr(), fFE).real

# PLOTTING SOME RELEVANT OBSEVABLES
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