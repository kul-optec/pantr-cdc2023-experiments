import casadi as cs

x1 = cs.SX.sym("x1")
x2 = cs.SX.sym("x2")
x3 = cs.SX.sym("x3")
y = cs.sin(x1 * x2) + cs.exp(x1 * x2 * x3)
f = cs.Function("f", [x1, x2, x3], [y])
grad_f = cs.Function("gr_f", [x1, x2, x3], [cs.gradient(y, cs.vertcat(x1, x2, x3))])

print(f(1.1, 1.2, 1.3))
print(grad_f(1.1, 1.2, 1.3))
