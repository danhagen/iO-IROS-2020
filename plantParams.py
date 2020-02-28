import numpy as np

# params dictionary

plantParams = {
    "Joint Inertia" : 1.15e-2, # kg⋅m²
    "Joint Damping" : 0.01, # N⋅s⋅m⁻¹
    "Joint Mass" : 0.541, # kg
    "Joint Moment Arm" : 0.05, # m
    "Link Center of Mass" : 0.085, # m
    "Link Length" : 0.3, # m
    "Motor Inertia" : 6.6e-5, # kg⋅m²
    "Motor Damping" : 0.00462, # N⋅s⋅m⁻¹
    "Motor Moment Arm" : 0.02, # m
    "Spring Stiffness Coefficient" : 100, # N
    "Spring Shape Coefficient" : 20, # unit-less
    "Quadratic Stiffness Coefficient 1" : 10000 ,
    "Quadratic Stiffness Coefficient 2": -4962.32258089,
    "Simulation Duration" : 100, # s
    "dt" : 0.001, # s
    "Position Gains" : {
        0 : 3162.3,
        1 : 1101.9,
        2 : 192.0,
        3 : 19.6
    },
    "Stiffness Gains" : {
        0 : 316.2,
        1 : 25.1
    },
    "Joint Angle Bounds" : {
        "LB" : np.pi/2,
        "UB" : 3*np.pi/2
    },
    "Maximum Joint Stiffness" : 100,
    "Boundary Friction Weight" : 3,
    "Boundary Friction Gain" : 1
}

# h is the step used to determine the derivative
h = 0.000001

#gravity
gr = 9.81 # m/s²
# gr = 0
