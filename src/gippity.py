import sympy as sp

xi, eta = sp.symbols('xi eta')
N = [0] * 8

N[0] = 0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1)
N[1] = 0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1)
N[2] = 0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1)
N[3] = 0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1)
N[4] = 0.5 * (1 - xi**2) * (1 - eta)
N[5] = 0.5 * (1 + xi) * (1 - eta**2)
N[6] = 0.5 * (1 - xi**2) * (1 + eta)
N[7] = 0.5 * (1 - xi) * (1 - eta**2)
dNdxi = [sp.simplify(sp.diff(Ni, xi)) for Ni in N]
dNdeta = [sp.simplify(sp.diff(Ni, eta)) for Ni in N]

for i in range(8):
    print(f'dNdxi[{i}]  = {dNdxi[i]}')
for i in range(8):
    print(f'dNdeta[{i}] = {dNdeta[i]}')
