from firedrake import *
import numpy as np

try:
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
except:
  warning("Matplotlib not imported")

ufile =      File('/home/wpan1/Data/PythonProjects/ltqg2/ltqg_b.pvd')
qfile =      File('/home/wpan1/Data/PythonProjects/ltqg2/ltqg_q.pvd')
bexactfile = File('/home/wpan1/Data/PythonProjects/ltqg2/ltqg_b_exact.pvd')
qexactfile = File('/home/wpan1/Data/PythonProjects/ltqg2/ltqg_q_exact.pvd')

ndump = 10

T = 20.0
t = 0.0

U = Constant(2)
B = Constant(-2)
beta = Constant(2)
H = Constant(1)

dt = 0.001
Dt = Constant(dt)

n = 64
L = 1
mesh = PeriodicIntervalMesh(n, L)

V = FunctionSpace(mesh, "DG", 1)
W = FunctionSpace(mesh, "CG", 1)

normal = FacetNormal(mesh)

b0 = Function(V, name="b")
q0 = Function(V, name="q")

b_exact = Function(V, name="b exact")
q_exact = Function(V, name="q exact")

b1 = Function(V)
db1 = Function(V)
q1 = Function(V)
dq1 = Function(V)

x, = SpatialCoordinate(mesh)

q0.interpolate(sin(2*pi*x))
b0.interpolate(sin(2*pi*x))

b_exact.assign(b0)
q_exact.assign(q0)

### psi problem ########
m = TestFunction(W)
psi = TrialFunction(W)
psi0 = Function(W)

Lpsi = - q1*m*dx
apsi = ( psi*m + psi.dx(0)*m.dx(0) )*dx

psiprob = LinearVariationalProblem(apsi, Lpsi, psi0, bcs=[])
psi_solver = LinearVariationalSolver(psiprob, solver_parameters={'ksp_type': 'cg'})

### b equation ########
b = TrialFunction(V)
v = TestFunction(V)

Ab_mass = v * b * dx
Ab_int = -U * v.dx(0) * b  * dx  + B * v * psi0.dx(0) * dx
Ab_flux = ( U * ( v('+') * normal('+') + v('-')*normal('-') )[0] * avg(b)  + 0.5 * U * jump(v) * jump(b) ) * dS

arhs_b = Ab_mass - Dt * (Ab_int + Ab_flux)

b_problem = LinearVariationalProblem(Ab_mass, action(arhs_b, b1), db1)
b_solver = LinearVariationalSolver(b_problem)

### q equation #########
q = TrialFunction(V)
w = TestFunction(V)

Aq_mass = w * q * dx
Aq_int = - U * w.dx(0) * q  * dx  - (U-0.5*H)* w * b1.dx(0) * dx  +  (U + B - beta) * w * psi0.dx(0) * dx 
Aq_flux = ( U * ( w('+') * normal('+') + w('-')*normal('-') )[0] * avg(q)  + 0.5 * U * jump(w) * jump(q) ) * dS

arhs_q = Aq_mass - Dt * (Aq_int + Aq_flux) 

q_problem = LinearVariationalProblem(Aq_mass, action(arhs_q, q1), dq1)
q_solver = LinearVariationalSolver(q_problem)


#################################################
b1.assign(b0)
q1.assign(q0)

ufile.write(b0, time=t)
qfile.write(q0, time=t)
bexactfile.write(b_exact, time=t)
qexactfile.write(q_exact, time=t)

dumpn = 0

while (t < T - 0.5*dt):

    b1.assign(b0)
    q1.assign(q0)
    psi_solver.solve()
    b_solver.solve()
    q_solver.solve()

    # # Find intermediate solution q^(1)
    b1.assign(db1)
    q1.assign(dq1)
    psi_solver.solve()
    b_solver.solve()
    q_solver.solve()

    # # Find intermediate solution q^(2)
    b1.assign(0.75 * b0 + 0.25 * db1)
    q1.assign(0.75 * q0 + 0.25 * dq1)
    psi_solver.solve()
    b_solver.solve()
    q_solver.solve()

    # # Find new solution q^(n+1)
    b0.assign(b0 / 3 + 2 * db1 / 3)
    q0.assign(q0 / 3 + 2 * dq1 / 3)

    dumpn += 1
    if dumpn == ndump:
        dumpn -= ndump
        _t = round(t, 3)
        print(_t)

        ufile.write(b0, time=_t)
        qfile.write(q0, time=_t)
        
        b_exact.interpolate(sin(2*pi*(x-U*t)))

        q_exact.interpolate( sin(2*pi*(x-U*t)) + t*2*(U-0.5*H)*pi*cos(2*pi*(x-U*t)) )

        bexactfile.write(b_exact, time=_t)
        qexactfile.write(q_exact, time=_t)
    
    t += dt
