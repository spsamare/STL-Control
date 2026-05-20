import pyomo.environ as pyo

# Create a model
model = pyo.ConcreteModel()
# Define variables
model.x = pyo.Var(within=pyo.NonNegativeReals)
model.y = pyo.Var(within=pyo.NonNegativeReals)
# Define objective
model.obj = pyo.Objective(expr=model.x + model.y, sense=pyo.minimize)
# Define constraints
model.con1 = pyo.Constraint(expr=model.x + 2 * model.y >= 4)
model.con2 = pyo.Constraint(expr=model.x - model.y <= 1)
# Select solver
solver = pyo.SolverFactory('glpk')
# Solve the problem
result = solver.solve(model)
# Display results
print('Status:', result.solver.status)
print('Termination Condition:', result.solver.termination_condition)
print('Optimal x:', pyo.value(model.x))
print('Optimal y:', pyo.value(model.y))
print('Optimal Objective:', pyo.value(model.obj))