from fipy import *
mesh = Grid1D(nx=100)
phi = CellVariable(name="solution", mesh=mesh, value=0.)
eq = DiffusionTerm()
eq.solve(var=phi)
print("Done!")