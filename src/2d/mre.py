from fipy import CellVariable, Grid2D, TransientTerm, DiffusionTerm, Viewer, FaceVariable, ImplicitSourceTerm
from fipy.tools import numerix as npx
from fipy.meshes import CylindricalGrid2D, Grid2D

L_total = 1.0
R_total = 0.1

mesh = Grid2D(Lx=L_total, Ly=R_total, dx=0.01, dy=0.01)
x_face, y_face = mesh.faceCenters
x_cell, y_cell = mesh.cellCenters

T = CellVariable(name="Temperature", mesh=mesh, value=290., hasOld=True)
# Define masks
faces_evaporator = mesh.facesLeft
faces_condenser = mesh.facesRight

# constants
rho_0 = 7200.0
# rho_0 = CellVariable(mesh=mesh, value=rho_0) # For spatial dependence, we have to define this explicitly
cp_0 = 440.5
# cp_0 = CellVariable(mesh=mesh, value=cp_0) # For spatial dependence, we have to define this explicitly
k_0 = 55.0
# k_0 = FaceVariable(mesh=mesh, value=k_0) # For spatial dependence, we have to define this explicitly
T_amb = 290. # Used as reference temperature (e.g., 290K)
h = 750.0

# Simple temperature-dependent properties
cp_T = cp_0 * (1 + 0.005 * (T - T_amb)) # cp is defined at cell centers, so we can use T directly

# Constant influx (inside equation)
q = -26.77e3

# Convective boundary condition
Gamma = FaceVariable(mesh=mesh, value=k_0)
Gamma.setValue(0., where=faces_condenser)
MA = npx.MA
tmp = MA.repeat(mesh._faceCenters[..., npx.NewAxis,:], 2, 1)
cellToFaceDistanceVectors = tmp - npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
tmp = npx.take(mesh._cellCenters, mesh.faceCellIDs, axis=1)
tmp = tmp[..., 1,:] - tmp[..., 0,:]
cellDistanceVectors = MA.filled(MA.where(MA.getmaskarray(tmp), cellToFaceDistanceVectors[:, 0], tmp))
dPf = FaceVariable(mesh=mesh,
                   value=mesh._faceToCellDistanceRatio * cellDistanceVectors)
n = mesh.faceNormals
a = FaceVariable(mesh=mesh, value=h*n, rank=1)
b = FaceVariable(mesh=mesh, value=k_0, rank=0)
g = FaceVariable(mesh=mesh, value=h*T_amb, rank=0)
RobinCoeff = faces_condenser * k_0 * n / (dPf.dot(a) + b)
eq = (TransientTerm(coeff=rho_0*cp_T, var=T) == DiffusionTerm(coeff=Gamma, var=T) + (RobinCoeff * g).divergence
       - ImplicitSourceTerm(coeff=(RobinCoeff * a.dot(n)).divergence, var=T)
       # Constant influx
       + (faces_evaporator * (q/k_0)).divergence)

T.setValue(T_amb)  # Set initial temperature

dt = 0.02
t_end = 15.0

viewer = Viewer(vars=T, title="Temperature Distribution")
viewer.plot()

npx.set_printoptions(linewidth=200)

# Solver WITH temperature-dependent properties (DON'T FORGET hasOld=True!)
sweeps = 5
timestep = 0
t = timestep * dt
for t in range(int(t_end/dt)):
    T.updateOld()
    for sweep in range(sweeps):
        res = eq.sweep(var=T, dt=dt)
        print(f"Iteration {t}, Sweep {sweep}, Residual: {res}")
    if __name__ == "__main__":
        viewer.plot()

from fipy import input
input("Press Enter to continue...")