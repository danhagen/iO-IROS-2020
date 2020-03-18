import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from danpy.sb import dsb
from danpy.useful_functions import save_figures,is_number
from scipy import signal
import numdifftools as nd
import scipy as sp
from plantParams import *
import argparse
import textwrap
from animate import *
import scipy.io as sio

def LP_filt(filter_length, x):
    """
    Finite Impulse Response (FIR) Moving Average (MA) Low-Pass Filter
    """
    b=np.ones(filter_length,)/(filter_length) #Finite Impulse Response (FIR) Moving Average (MA) filter with one second filter length
    a=1
    y = signal.filtfilt(b, a, x)
    return y

class plant_pendulum_1DOF2DOF:
    def __init__(self,plantParams):
        self.params = plantParams

        self.Ij = plantParams.get("Joint Inertia", 1.15e-2) # kg⋅m²
        is_number(self.Ij,"Joint Inertia",default=1.15e-2)

        self.bj = plantParams.get("Joint Damping", 0.001) # N⋅s⋅m⁻¹
        is_number(self.bj,"Joint Damping",default=0.001)

        self.mj = plantParams.get("Joint Mass", 0.541) # kg
        is_number(self.mj,"Joint Mass",default=0.541)

        self.rj = plantParams.get("Joint Moment Arm", 0.05) # m
        is_number(self.rj,"Joint Moment Arm",default=0.05)

        self.Lcm = plantParams.get("Link Center of Mass", 0.085) # m
        is_number(self.Lcm,"Link Center of Mass",default=0.085)

        self.L = plantParams.get("Link Length", 0.3) # m
        is_number(self.L,"Link Length",default=0.3)

        self.Jm = plantParams.get("Motor Inertia", 6.6e-5) # kg⋅m²
        is_number(self.Jm,"Motor Inertia",default=6.6e-5)

        self.bm = plantParams.get("Motor Damping", 0.00462) # N⋅s⋅m⁻¹
        is_number(self.bm,"Motor Damping",default=0.00462)

        self.rm = plantParams.get("Motor Moment Arm", 0.01) # m
        is_number(self.rm,"Motor Moment Arm",default=0.01)

        self.k_spr = plantParams.get("Spring Stiffness Coefficient",1) # N
        is_number(self.k_spr,"",default=1)

        self.b_spr = plantParams.get("Spring Shape Coefficient",100) # unit-less
        is_number(self.b_spr,"",default=1)

        self.simulationDuration = plantParams.get("Simulation Duration", 1000)
        is_number(self.simulationDuration,"Simulation Duration")

        self.dt = plantParams.get("dt", 0.01)
        is_number(self.dt,"dt")

        self.k0 = plantParams.get(
            "Position Gains",
            {
                0 : 3162.3,
                1 : 1101.9,
                2 : 192.0,
                3 : 19.6
            }
        )
        self.ks = plantParams.get(
            "Stiffness Gains",
            {
                0 : 316.2,
                1 : 25.1
            }
        )

        self.jointAngleBounds = plantParams.get(
            "Joint Angle Bounds",
            {
                "LB" : np.pi/2,
                "UB" : 3*np.pi/2
            }
        )
        self.jointAngleRange = (
            self.jointAngleBounds["UB"]
            - self.jointAngleBounds["LB"]
        )
        self.jointAngleMidPoint = (
            self.jointAngleBounds["UB"]
            + self.jointAngleBounds["LB"]
        )/2

        self.jointStiffnessBounds = {
            "LB" : 2*(self.rj**2)*self.k_spr*self.b_spr
            }
        self.jointStiffnessBounds["UB"] = plantParams.get("Maximum Joint Stiffness",100)
        is_number(self.jointStiffnessBounds["UB"],"Maximum Joint Stiffness",default=100)
        self.jointStiffnessBounds["LB"] = 15
        self.jointStiffnessRange = (
            self.jointStiffnessBounds["UB"]
            - self.jointStiffnessBounds["LB"]
        )

        self.jointStiffnessMidPoint = (
            self.jointStiffnessBounds["UB"]
            + self.jointStiffnessBounds["LB"]
        )/2

        self.boundaryFrictionWeight = plantParams.get("Boundary Friction Weight",0.1)
        is_number(self.boundaryFrictionWeight,"Boundary Friction Weight",
            default=0.1,
            notes="It appears that values less than 1 will produce steeper boundaries for the periodic solution, while values greater than 1 will produce sharper bounderies for the quadratic penalty function."
        )
        self.boundaryFrictionGain = plantParams.get("Boundary Friction Gain",1)
        is_number(self.boundaryFrictionGain,"Boundary Friction Gain",default=1)

        self.time = np.arange(
            0,
            self.simulationDuration+self.dt,
            self.dt
        )

    def return_X_o(self,x1o,U_o):
        """
        Returns the initial state X_o where the positions of the motors depends on the initial position of the pendulum and the amount of torque on the motors. This assumes we are at (or near) equilibrium at the start of a simulation to get us close to the actual initial state (and therefore minimize the large transitions at the start).

        U_o should be an array with 2 elements and x1o should be a number given in radians between 0 and 2 pi.
        """
        lTo1 = np.log((U_o[0]/(self.k_spr*self.rm))+1)/self.b_spr
        lTo2 = np.log((U_o[1]/(self.k_spr*self.rm))+1)/self.b_spr
        x35o = (
            (1/(2*self.rm))
            * np.matrix([[1,1],[1,-1]])
            * np.matrix([
                [lTo1+lTo2],
                [2*self.rj*x1o + lTo1 - lTo2]
            ])
        )

        return([x1o,0,x35o[0,0],0,x35o[1,0],0])

    def C(self,X):
        """
        Returns zero until the effects are quantified
        """
        return(
            0
        )
    def dCdx1(self,X):
        return(0)
    def d2Cdx12(self,X):
        return(0)
    def d2Cdx1x2(self,X):
        return(0)
    def dCdx2(self,X):
        return(0)
    def d2Cdx22(self,X):
        return(0)

    def update_state_variables(self,X):

        #>>>> State functions

        self.f1 = self.f1_func(X)
        self.f2 = self.f2_func(X)
        self.f3 = self.f3_func(X)
        self.f4 = self.f4_func(X)
        self.f5 = self.f5_func(X)
        self.f6 = self.f6_func(X)

        #>>>> State functions first gradient

        # self.df1dx1 = 0
        self.df1dx2 = 1
        # self.df1dx3 = 0
        # self.df1dx4 = 0
        # self.df1dx5 = 0
        # self.df1dx6 = 0

        self.df2dx1 = self.df2dx1_func(X)
        self.df2dx2 = self.df2dx2_func(X)
        self.df2dx3 = self.df2dx3_func(X)
        # self.df2dx4 = 0
        self.df2dx5 = self.df2dx5_func(X)
        # self.df2dx6 = 0

        # self.df3dx1 = 0
        # self.df3dx2 = 0
        # self.df3dx3 = 0
        self.df3dx4 = 1
        # self.df3dx5 = 0
        # self.df3dx6 = 0

        # self.df4dx1 = N/A
        # self.df4dx2 = N/A
        # self.df4dx3 = N/A
        # self.df4dx4 = N/A
        # self.df4dx5 = N/A
        # self.df4dx6 = N/A

        # self.df5dx1 = 0
        # self.df5dx2 = 0
        # self.df5dx3 = 0
        # self.df5dx4 = 0
        # self.df5dx5 = 0
        self.df5dx6 = 1

        # self.df6dx1 = N/A
        # self.df6dx2 = N/A
        # self.df6dx3 = N/A
        # self.df6dx4 = N/A
        # self.df6dx5 = N/A
        # self.df6dx6 = N/A

        #>>>> State functions second gradient

        self.d2f2dx12 = self.d2f2dx12_func(X)
        self.d2f2dx1x2 = self.d2f2dx1x2_func(X)
        self.d2f2dx1x3 = self.d2f2dx1x3_func(X)
        self.d2f2dx1x5 = self.d2f2dx1x5_func(X)

        self.d2f2dx22 = self.d2f2dx22_func(X)

        self.d2f2dx32 = self.d2f2dx32_func(X)

        self.d2f2dx52 = self.d2f2dx52_func(X)

    # def motor_coupling_function(self,X,motorNumber):
    #     return(
    #         self.rm*self.k_spr*(
    #             np.exp(
    #                 self.b_spr*(
    #                     self.rm*X[2+2*(motorNumber-1)]
    #                     + ((1.5-motorNumber)/0.5)*self.rj*X[0]
    #                 )
    #             )
    #             -1
    #         )
    #     )
    def tendon_1_FL_func(self,X):
        return(
            self.k_spr*(
                np.exp(self.b_spr*(self.rm*X[2]-self.rj*X[0]))
                - 1
            ) * ((self.rm*X[2]-self.rj*X[0])>=0)
        )
    def tendon_2_FL_func(self,X):
        return(
            self.k_spr*(
                np.exp(self.b_spr*(self.rm*X[4]+self.rj*X[0]))
                - 1
            ) * ((self.rm*X[4]+self.rj*X[0])>=0)
        )

    def f1_func(self,X):
        return(X[1])

    def f2_func(self,X):
        return(
            (
                -self.C(X) # Coriolis and centrifugal torques (zero)
                - self.bj*X[1] # damping torque
                - self.Lcm*self.mj*gr*np.sin(X[0]) # gravitational torque
                + self.rj*self.k_spr * (
                    (np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))-1)
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                    - (np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))-1)
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                ) # total coupling torque between motors and joint
            )/self.Ij
        )
    def df2dx1_func(self,X):
        result = (
            (
                -self.dCdx1(X) # Coriolis and centrifugal torques (zero)
                - self.Lcm*self.mj*gr*np.cos(X[0]) # gravitational torque
                - (self.rj**2)*self.k_spr*self.b_spr * (
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                    + np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                ) # total coupling torque between motors and joint
            )/self.Ij
        )
        return(result)
    def d2f2dx12_func(self,X):
        return(
            (
                -self.d2Cdx12(X) # Coriolis and centrifugal torques (zero)
                + self.Lcm*self.mj*gr*np.sin(X[0]) # gravitational torque
                + (self.rj**3)*self.k_spr*(self.b_spr**2) * (
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                    - np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                ) # total coupling torque between motors and joint
            )/self.Ij
        )
    def d2f2dx1x2_func(self,X):
        return(
            (
                -self.d2Cdx1x2(X) # Coriolis and centrifugal torques (zero)
            )/self.Ij
        )
    def d2f2dx1x3_func(self,X):
        """
        This is equivalently -dSda/Ij
        """
        return(
            -(self.rj**2)*self.rm*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
            ) / self.Ij
        )
    def d2f2dx1x5_func(self,X):
        """
        This is equivalently dSdb/Ij
        """
        return(
            -(self.rj**2)*self.rm*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            ) / self.Ij
        )
    def df2dx2_func(self,X):
        result = (
            (
                -self.dCdx2(X) # Coriolis and centrifugal torques (zero)
                - self.bj # damping torque
            )/self.Ij
        )
        return(result)
    def d2f2dx22_func(self,X):
        return(
            (
                -self.d2Cdx22(X) # Coriolis and centrifugal torques (zero)
            )/self.Ij
        )
    def df2dx3_func(self,X):
        """
        Equivalently, this is the negative value of -Q_{11}/Ij
        """
        result = (
            self.rj*self.rm*self.k_spr*self.b_spr * (
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
            ) / self.Ij
        )
        return(result)
    def d2f2dx32_func(self,X):
        return(
            self.rj*(self.rm**2)*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
            ) / self.Ij
        )
    def df2dx5_func(self,X):
        """
        Equivalently, this is the negative value of -Q_{12}/Ij
        """
        result = (
            -self.rj*self.rm*self.k_spr*self.b_spr * (
                np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            ) / self.Ij
        )
        return(result)
    def d2f2dx52_func(self,X):
        return(
            -self.rj*(self.rm**2)*self.k_spr*(self.b_spr**2) * (
                np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            ) / self.Ij
        )

    def f3_func(self,X):
        return(X[3])

    def f4_func(self,X):
        return(
            (
                -self.bm*X[3]
                - self.rm*self.k_spr*(
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    -1
                ) * ((self.rm*X[2] - self.rj*X[0])>=0)
            )/self.Jm
        )

    def f5_func(self,X):
        return(X[5])

    def f6_func(self,X):
        return(
            (
                -self.bm*X[5]
                - self.rm*self.k_spr*(
                    np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    -1
                ) * ((self.rm*X[4] + self.rj*X[0])>=0)
            )/self.Jm
        )

    def f(self,X):
        result = np.zeros((6,1))
        result[0,0] = self.f1
        result[1,0] = self.f2
        result[2,0] = self.f3
        result[3,0] = self.f4
        result[4,0] = self.f5
        result[5,0] = self.f6
        return(result)
    def g(self,X):
        result = np.matrix(np.zeros((6,2)))
        result[3,0] = 1/self.Jm
        result[5,1] = 1/self.Jm
        return(result)
    def h(self,X):
        result = np.zeros((2,))
        result[0] = X[0]
        result[1] = (self.rj**2)*self.k_spr*self.b_spr*(
            np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
            + np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
        )
        return(result)

    def forward_simulation(self,X_o,U=None,addTitle=None):
        """
        Building our own f_array to reduce the number of calls for f_funcs by making it a static call for each iteration in the FBL instance.
        """
        assert len(X_o)==6, "X_o must have 6 elements, not " + str(len(X_o)) + "."
        if addTitle is None:
            addTitle = "Custom"
        else:
            assert type(addTitle)==str, "addTitle must be a str."

        dt = self.time[1]-self.time[0]
        if U is None:
            U = np.zeros((2,len(self.time)-1))
        else:
            assert np.shape(U)==(2,len(self.time)-1), "U must be either None (default) of have shape (2,len(self.time)-1), not " + str(np.shape(U)) + "."
        X = np.zeros((6,len(self.time)))
        Y = np.zeros((2,len(self.time)))
        X[:,0] = X_o
        Y[:,0] = self.h(X[:,0])
        statusbar=dsb(0,len(self.time)-1,title="Forward Simulation (" + addTitle + ")")
        for i in range(len(self.time)-1):
            X[0,i+1] = X[0,i] + self.dt*self.f1_func(X[:,i])
            X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            if X[0,i+1]<=self.jointAngleBounds["LB"]:
                X[0,i+1] = self.jointAngleBounds["LB"]
                if X[1,i+1]<0: X[1,i+1] = 0
                # X[1,i+1] = X[1,i]
            elif X[0,i+1]>=self.jointAngleBounds["UB"]:
                X[0,i+1] = self.jointAngleBounds["UB"]
                if X[1,i+1]>0: X[1,i+1] = 0
                # X[1,i+1] = X[1,i]
            # X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            X[2,i+1] = X[2,i] + self.dt*self.f3_func(X[:,i])
            X[3,i+1] = X[3,i] + self.dt*(self.f4_func(X[:,i]) + U[0,i]/self.Jm)
            X[4,i+1] = X[4,i] + self.dt*self.f5_func(X[:,i])
            X[5,i+1] = X[5,i] + self.dt*(self.f6_func(X[:,i]) + U[1,i]/self.Jm)

            Y[:,i+1] = self.h(X[:,i+1])
            # self.update_state_variables(X[:,i+1])
            statusbar.update(i)
        return(X,U,Y)

    def h0(self,X):
        return(X[0])
    def Lfh0(self,X):
        return(X[1])
    def Lf2h0(self,X):
        return(self.f2)
    def Lf3h0(self,X):
        result = (
            self.df2dx1*self.f1
            + self.df2dx2*self.f2
            + self.df2dx3*self.f3
            + self.df2dx5*self.f5
        )
        return(result)
    def Lf4h0(self,X):
        return(
            (
                self.d2f2dx12*self.f1
                + self.d2f2dx1x2*self.f2
                + self.df2dx2*self.df2dx1
                + self.d2f2dx1x3*self.f3
                + self.d2f2dx1x5*self.f5
            ) * self.f1
            + (
                self.d2f2dx1x2*self.f1
                + self.df2dx1
                + self.d2f2dx22*self.f2
                + (self.df2dx2**2)
            ) * self.f2
            + (
                self.d2f2dx1x3*self.f1
                + self.df2dx2*self.df2dx3
                + self.d2f2dx32*self.f3
            ) * self.f3
            + (
                self.df2dx3
            ) * self.f4
            + (
                self.d2f2dx1x5*self.f1
                + self.df2dx2*self.df2dx5
                + self.d2f2dx52*self.f5
            ) * self.f5
            + (
                self.df2dx5
            ) * self.f6
        )

    def hs(self,X):
        return(
            (self.rj**2)*self.k_spr*self.b_spr*(
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
                + np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            )
        )
    def Lfhs(self,X):
        return(
            (self.rj**2)*self.k_spr*(self.b_spr**2)*(
                -(self.rj*self.f1 - self.rm*self.f3)*(
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                        * ((self.rm*X[2] - self.rj*X[0])>=0)
                )
                + (self.rj*self.f1 + self.rm*self.f5)*(
                    np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                        * ((self.rm*X[4] + self.rj*X[0])>=0)
                )
            )
        )
    def Lf2hs(self,X):
        return(
            (self.rj**2)*self.k_spr*(self.b_spr**2)*(
                (
                    self.b_spr*(self.rj*self.f1 - self.rm*self.f3)**2
                    - self.rj*self.f2
                    + self.rm*self.f4
                ) * np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))
                    * ((self.rm*X[2] - self.rj*X[0])>=0)
                + (
                    self.b_spr*(self.rj*self.f1 + self.rm*self.f5)**2
                    + self.rj*self.f2
                    + self.rm*self.f6
                ) * np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))
                    * ((self.rm*X[4] + self.rj*X[0])>=0)
            )
        )

    # def Phi(self,X):
    #     return(
    #         np.matrix([[
    #             self.h0(X),
    #             self.Lfh0(X),
    #             self.Lf2h0(X),
    #             self.Lf3h0(X),
    #             self.hs(X),
    #             self.Lfhs(X)
    #         ]]).T
    #     )
    def v0(self,X,x1d):
        result = (
            x1d[4]
            + self.k0[3]*(x1d[3]-self.Lf3h0(X))
            + self.k0[2]*(x1d[2]-self.Lf2h0(X))
            + self.k0[1]*(x1d[1]-self.Lfh0(X))
            + self.k0[0]*(x1d[0]-self.h0(X))
        )
        return(result)
    def vs(self,X,Sd):
        result =(
            Sd[2]
            + self.ks[1]*(Sd[1]-self.Lfhs(X))
            + self.ks[0]*(Sd[0]-self.hs(X))
        )
        return(result)

    def Q(self,X):
        B = np.matrix([
            [1/(self.Jm*self.Ij),0],
            [0,1/self.Jm]
        ])
        W = self.rj*self.rm*self.k_spr*self.b_spr*np.matrix([
            [
                np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))* ((self.rm*X[2] - self.rj*X[0])>=0),
                -np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))* ((self.rm*X[4] + self.rj*X[0])>=0)
            ],
            [
                self.rj*self.b_spr*(
                    np.exp(self.b_spr*(self.rm*X[2] - self.rj*X[0]))* ((self.rm*X[2] - self.rj*X[0])>=0)
                ),
                self.rj*self.b_spr*(
                    np.exp(self.b_spr*(self.rm*X[4] + self.rj*X[0]))* ((self.rm*X[4] + self.rj*X[0])>=0)
                )
            ]
        ])
        return(B*W)
    def return_input(self,X,x1d,Sd):
        Q_inv = self.Q(X)**(-1)
        return(
            Q_inv
            * (
                np.matrix([[-self.Lf4h0(X),-self.Lf2hs(X)]]).T
                + np.matrix([[self.v0(X,x1d),self.vs(X,Sd)]]).T
            )
        )

    def generate_desired_trajectory_STEPS(
            self,
            passProbability,
            type,
            delay=3
        ):

        assert type in ["angle","stiffness","both"], "type must be either 'angle','stiffness' or 'both' (generate both at the same time)."

        if type!='both':
            if type=='angle':
                minValue = self.jointAngleBounds["LB"]
                startValue = self.jointAngleMidPoint
                valueRange = self.jointAngleRange
            elif type=='stiffness':
                minValue = self.jointStiffnessBounds["LB"]
                startValue = self.jointStiffnessBounds["LB"]
                valueRange = self.jointStiffnessRange

            trajectory = startValue*np.ones((1,len(self.time)))
            trajectory[0,int(delay/self.dt)] = (
                valueRange*np.random.uniform(0,1) + minValue
            )
            for i in range(int(delay/self.dt)+1, len(self.time)):
                if np.random.uniform() < passProbability: # change offset
                    trajectory[0,i] = (
                        valueRange*np.random.uniform(0,1) + minValue
                    )
                else: # stay at previous input
                    trajectory[0,i] = trajectory[0,i-1]
        else:
            minValue = [
                self.jointAngleBounds["LB"],
                self.jointStiffnessBounds["LB"]
            ]
            startValue = [
                self.jointAngleMidPoint,
                self.jointStiffnessMidPoint
            ]
            valueRange = [
                self.jointAngleRange,
                self.jointStiffnessRange
            ]

            trajectory = np.ones((2,len(self.time)))
            trajectory[0,:] = startValue[0]*np.ones((1,len(self.time)))
            trajectory[0,int(delay/self.dt)] = (
                valueRange[0]*np.random.uniform(0,1) + minValue[0]
            )

            trajectory[1,:] = startValue[1]*np.ones((1,len(self.time)))
            trajectory[1,int(delay/self.dt)] = (
                valueRange[1]*np.random.uniform(0,1) + minValue[1]
            )


            for i in range(int(delay/self.dt)+1, len(self.time)):
                if np.random.uniform() < passProbability: # change offset
                    trajectory[0,i] = (
                        valueRange[0]*np.random.uniform(0,1) + minValue[0]
                    )
                    trajectory[1,i] = (
                        valueRange[1]*np.random.uniform(0,1) + minValue[1]
                    )
                else: # stay at previous input
                    trajectory[:,i] = trajectory[:,i-1]
        return(trajectory)

    def generate_desired_trajectory_SINUSOIDAL(
            self,
            type,
            delay=3
        ):

        assert type in ["angle","stiffness"], "type must be either 'angle' or 'stiffness'."

        randDelay = np.random.uniform(0.5,1.5)*delay
        randFrequency = np.random.uniform(1/8,1)
        randLength = len(self.time[int(randDelay/self.dt):])

        if type=='angle':
            amplitude = (
                self.jointAngleMidPoint
                - self.jointAngleBounds["LB"]
            )
            offset = self.jointAngleMidPoint
            sinusoidal_func = lambda t:(
                amplitude*np.sin(
                    2*np.pi*randFrequency
                    * (t-randDelay)
                )
                + offset
            )

        elif type=='stiffness':
            amplitude = (
                self.jointStiffnessMidPoint
                - self.jointStiffnessBounds["LB"]
            )
            offset = self.jointStiffnessMidPoint
            sinusoidal_func = lambda t: (
                -amplitude*np.cos(
                    2*np.pi*randFrequency
                    * (t-randDelay)
                )
                + offset
            )

        startValue = sinusoidal_func(randDelay)
        trajectory = startValue*np.ones((1,len(self.time)))
        trajectory[0,int(randDelay/self.dt):] = [
            sinusoidal_func(t) for t in self.time[-randLength:]
        ]
        return(trajectory)

    def forward_simulation_FL(self,X_o,X1d,Sd):
        assert len(X_o)==6, "X_o must have 6 elements, not " + str(len(X_o)) + "."
        dt = self.time[1]-self.time[0]
        U = np.zeros((2,len(self.time)-1),dtype=np.float64)
        X = np.zeros((6,len(self.time)),dtype=np.float64)
        X_measured = np.zeros((6,len(self.time)),dtype=np.float64)
        Y = np.zeros((2,len(self.time)),dtype=np.float64)
        X[:,0] = X_o
        Y[:,0] = self.h(X[:,0])
        self.update_state_variables(X_o)
        statusbar=dsb(0,len(self.time)-1,title="Forward Simulation (FBL)")
        self.desiredOutput = np.array([X1d[0,:],Sd[0,:]])
        for i in range(len(self.time)-1):
            if i>0:
                X_measured[0,i] = X[0,i]
                X_measured[1,i] = (X[0,i]-X[0,i-1])/self.dt
                X_measured[2,i] = X[2,i]
                X_measured[3,i] = (X[2,i]-X[2,i-1])/self.dt
                X_measured[4,i] = X[4,i]
                X_measured[5,i] = (X[4,i]-X[4,i-1])/self.dt
            else:
                X_measured[:,i] = X[:,i]
            U[:,i] = (self.return_input(X[:,i],X1d[:,i],Sd[:,i])).flatten()

            # X[0,i+1] = X[0,i] + self.dt*self.f1_func(X[:,i])
            # X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            # if X[0,i+1]<=self.jointAngleBounds["LB"]:
            #     X[0,i+1] = self.jointAngleBounds["LB"]
            #     if X[1,i+1]<0: X[1,i+1] = 0
            #     # X[1,i+1] = X[1,i]
            # elif X[0,i+1]>=self.jointAngleBounds["UB"]:
            #     X[0,i+1] = self.jointAngleBounds["UB"]
            #     if X[1,i+1]>0: X[1,i+1] = 0
            #     # X[1,i+1] = X[1,i]
            # # X[1,i+1] = X[1,i] + self.dt*self.f2_func(X[:,i])
            # X[2,i+1] = X[2,i] + self.dt*self.f3_func(X[:,i])
            # X[3,i+1] = X[3,i] + self.dt*(self.f4_func(X[:,i]) + U[0,i]/self.Jm)
            # X[4,i+1] = X[4,i] + self.dt*self.f5_func(X[:,i])
            # X[5,i+1] = X[5,i] + self.dt*(self.f6_func(X[:,i]) + U[1,i]/self.Jm)

            X[:,i+1] = (
                X[:,i]
                + self.dt*(
                    self.f(X[:,i])
                    + self.g(X[:,i])@U[:,np.newaxis,i]
                ).T
            )
            Y[:,i+1] = self.h(X[:,i+1])
            self.update_state_variables(X[:,i+1])
            statusbar.update(i)
        return(X,U,Y,X_measured)

    def plot_tendon_tension_deformation_curves(self,X,
            returnValues=False,
            addTitle=None):

        assert type(returnValues)==bool, "returnValues must be either True or False (default)."

        if addTitle is None:
            addTitle = ""
        else:
            assert type(addTitle)==str, "addTitle can either be None (default) or a string."
            addTitle = "\n " + addTitle

        tendonTension1 = np.array(list(map(self.tendon_1_FL_func,X.T)))
        tendonDeformation1 = np.array([-self.rj,0,self.rm,0,0,0])@X
        tendonTension2= np.array(list(map(self.tendon_2_FL_func,X.T)))
        tendonDeformation2 = np.array([self.rj,0,0,0,self.rm,0])@X

        minimumDeformation = min([
            tendonDeformation1.min(),
            tendonDeformation2.min()
        ])
        maximumDeformation = max([
            tendonDeformation1.max(),
            tendonDeformation2.max(),
            0.1
        ])
        deformationRange = maximumDeformation - minimumDeformation
        deformationArray = np.linspace(
            0,
            maximumDeformation + 0.1*deformationRange,
            1001
        )
        actualForceLengthCurve = np.array(list(map(
            lambda x3: self.tendon_1_FL_func([0,0,x3/self.rm,0,0,0]),
            deformationArray
        )))

        fig = plt.figure(figsize=(10,8))
        ax1 = fig.add_subplot(221) # FL 1
        ax2 = fig.add_subplot(223) # self.time v Deformation 1
        ax3 = fig.add_subplot(222) # FL 2
        ax4 = fig.add_subplot(224) # self.time v Deformation 2
        plt.suptitle("Tendon Deformation vs. Tension" + addTitle)
        tAxes = [[ax1,ax2],[ax3,ax4]]
        tendonDeformation = [tendonDeformation1,tendonDeformation2]
        tendonTension = [tendonTension1,tendonTension2]
        colors = ["C0","C0"]
        for i in range(2):
            tAxes[i][0].plot(np.linspace(-1,0,1001),np.zeros((1001,)),'0.70')
            tAxes[i][0].plot(deformationArray,actualForceLengthCurve,'0.70')
            tAxes[i][0].plot(tendonDeformation[i],tendonTension[i],c=colors[i])
            tAxes[i][0].set_xlim([
                minimumDeformation - 0.1*deformationRange,
                maximumDeformation + 0.1*deformationRange
            ])
            tAxes[i][0].set_xlabel("Tendon Deformation (m)")
            tAxes[i][0].set_ylabel("Tendon " + str(i+1) + " Tension (N)")
            tAxes[i][0].spines['right'].set_visible(False)
            tAxes[i][0].spines['top'].set_visible(False)

            tAxes[i][1].plot(tendonDeformation[i],-self.time,c=colors[i])
            tAxes[i][1].set_ylabel("Time (s)")
            tAxes[i][1].set_xlim([
                minimumDeformation - 0.1*deformationRange,
                maximumDeformation + 0.1*deformationRange
            ])
            tAxes[i][0].set_xticklabels([
                "" for tick in tAxes[i][0].get_xticks()
            ])
            tAxes[i][1].set_yticks([-self.time[0],-self.time[-1]])
            tAxes[i][1].set_yticklabels([self.time[0],self.time[-1]])
            tAxes[i][1].xaxis.tick_top()
            tAxes[i][1].spines['right'].set_visible(False)
            tAxes[i][1].spines['bottom'].set_visible(False)

        if returnValues==True:
            return(tendonDeformation,tendonTension)

    def plot_motor_angles(self,X):
        fig = plt.figure()
        ax = plt.gca()
        ax.plot(self.time,X[2,:]*180/np.pi,'C0')
        ax.plot(self.time,X[4,:]*180/np.pi,'C0',ls='--')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Motor Angles (deg)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.legend(["Motor 1","Motor 2"],loc="upper right")

    def plot_states(self,X,**kwargs):
        """
        Take the numpy.ndarray for the state space (X) of shape (M,N), where M is the number of states and N is the same length as time t. Returns a plot of the states.

        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        **kwargs
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        1) Return - must be a bool. Determines if the function returns a function handle. Default is False.

        2) InputString - must be a string. Input to the DescriptiveTitle that can be used to personalize the title. Default is None.

        """
        import numpy as np
        import matplotlib.pyplot as plt

        assert (np.shape(X)[0] in [6,8]) \
                    and (np.shape(X)[1] == len(self.time)) \
                        and (str(type(X)) == "<class 'numpy.ndarray'>"), \
                "X must be a (6,N) or (8,N) numpy.ndarray, where N is the length of t."


        Return = kwargs.get("Return",False)
        assert type(Return)==bool, "Return must be either True or False."

        InputString = kwargs.get("InputString",None)
        assert InputString is None or type(InputString)==str, "InputString must either be None or a str."

        NumStates = np.shape(X)[0]
        X[:6,:] = 180*X[:6,:]/np.pi # converting to deg and deg/s
        X[0,:] -= 180 # centering joint angle at 0 deg.
        if NumStates == 6:
            NumColumns = 2
            NumRows = 3
        else:
            NumColumns = 4
            NumRows = 2

        ColumnNumber = [el%2 for el in np.arange(0,NumStates,1)]
        RowNumber = [int(el/2) for el in np.arange(0,NumStates,1)]
        Units = [
            "(Deg)","(Deg/s)",
            "(Deg)","(Deg/s)",
            "(Deg)","(Deg/s)",
            "(N)","(N)"]
        if InputString is None:
            DescriptiveTitle = "Plotting States vs. Time"
        else:
            assert type(InputString)==str, "InputString must be a string"
            DescriptiveTitle = InputString + " Driven"
        if NumRows == 1:
            FigShape = (NumColumns,)
        else:
            FigShape = (NumRows,NumColumns)
        Figure = kwargs.get("Figure",None)
        assert (Figure is None) or \
                    (    (type(Figure)==tuple) and \
                        (str(type(Figure[0]))=="<class 'matplotlib.figure.Figure'>") and\
                        (np.array([str(type(ax))=="<class 'matplotlib.axes._subplots.AxesSubplot'>" \
                            for ax in Figure[1].flatten()]).all()) and \
                        (Figure[1].shape == FigShape)\
                    ),\
                         ("Figure can either be left blank (None) or it must be constructed from data that has the same shape as X.\ntype(Figure) = " + str(type(Figure)) + "\ntype(Figure[0]) = " + str(type(Figure[0])) + "\nFigure[1].shape = " + str(Figure[1].shape) + " instead of (" + str(NumRows) + "," + str(NumColumns) + ")" + "\ntype(Figure[1].flatten()[0]) = " + str(type(Figure[1].flatten()[0])))
        if Figure is None:
            fig, axes = plt.subplots(NumRows,NumColumns,figsize=(3.5*NumColumns,2*NumRows + 2),sharex=True)
            plt.subplots_adjust(top=0.85,bottom=0.15,left=0.075,right=0.975)
            plt.suptitle(DescriptiveTitle,Fontsize=20,y=0.975)
            for j in range(NumStates):
                axes[RowNumber[j],ColumnNumber[j]].spines['right'].set_visible(False)
                axes[RowNumber[j],ColumnNumber[j]].spines['top'].set_visible(False)
                axes[RowNumber[j],ColumnNumber[j]].plot(self.time,X[j,:])
                if not(RowNumber[j] == RowNumber[-1] and ColumnNumber[j]==0):
                    plt.setp(axes[RowNumber[j],ColumnNumber[j]].get_xticklabels(), visible=False)
                    # axes[RowNumber[j],ColumnNumber[j]].set_xticklabels(\
                    #                     [""]*len(axes[RowNumber[j],ColumnNumber[j]].get_xticks()))
                else:
                    axes[RowNumber[j],ColumnNumber[j]].set_xlabel("Time (s)")
                axes[RowNumber[j],ColumnNumber[j]].set_title(r"$x_{" + str(j+1) + "}$ "+ Units[j])
                # if NumStates%5!=0:
                #     [fig.delaxes(axes[RowNumber[-1],el]) for el in range(ColumnNumber[-1]+1,5)]
        else:
            fig = Figure[0]
            axes = Figure[1]
            for i in range(NumStates):
                if NumRows != 1:
                    axes[RowNumber[i],ColumnNumber[i]].plot(self.time,X[i,:])
                else:
                    axes[ColumnNumber[i]].plot(self.time,X[i,:])
        X[0,:] += 180 # returning to original frame
        X[:6,:] = np.pi*X[:6,:]/180 # returning to radians
        if Return == True:
            return((fig,axes))
        else:
            plt.show()

    def plot_joint_angle_power_spectrum_and_distribution(self,X,**kwargs):
        fig1 = plt.figure(figsize=(7, 5))
        ax1=plt.gca()
        plt.title('PSD: Power Spectral Density')
        plt.xlabel('Frequency')
        plt.ylabel('Power')

        freqs,psd = signal.welch(
            X[0,:],
            1/self.dt
        )
        ax1.semilogx(freqs,psd,c='C0')

        fig2 = plt.figure(figsize=(7,5))
        ax2 = plt.gca()
        ax2.set_ylabel("Percentage",fontsize=14)
        ax2.set_xlabel('Joint Angle (deg.)',fontsize=14)
        ax2.spines["right"].set_visible(False)
        ax2.spines["top"].set_visible(False)

        X[0,:] -= np.pi # shifting vertical position to 0 rad.
        hist,bin_edges=np.histogram(X[0,:],bins=48)
        percentNearBoundaries = (hist[0]+hist[-1])/len(X[0,:])*100 # percent within 180/48 = 3.75 deg of the boundaries.
        _,_,_ = ax2.hist(
            x=X[0,:]*(180/np.pi),
            bins=12,
            color='C0',
            alpha=0.7,
            weights=np.ones(len(X[0,:]*(180/np.pi))) / len(X[0,:]*(180/np.pi))
        )
        _,yMax = ax2.get_ylim()
        ax2.text(
            0,0.9*yMax,
            "{:.2f}".format(percentNearBoundaries) + "%" + " of Data\n" + r"$<3.75^\circ$ from Boundaries",
            fontsize=14,
            wrap=True,
            horizontalalignment='center',
            verticalalignment='center',
            color = "C0",
            bbox=dict(
                boxstyle='round',
                facecolor='C0',
                lw=0,
                alpha=0.2
            )
        )
        # sns.distplot(
        #     X[0,:]*(180/np.pi),
        #     hist=True,
        #     kde=False,
        #     hist_kws = {
        #         'weights':np.ones(len(X[0,:]*(180/np.pi))) / len(X[0,:]*(180/np.pi))
        #     },
        #     color='C0',
        #     ax=ax2
        # )
        ax2.set_xticks([-90,-45,0,45,90])
        ax2.set_xticklabels([str(int(tick))+r"$^\circ$" for tick in ax2.get_xticks()])
        ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
        X[0,:] += np.pi # shifting back

    def save_data(self,X,U,additionalDict=None,path=None):
        fT1 = np.array(list(map(self.tendon_1_FL_func,X.T)))
        fT2 = np.array(list(map(self.tendon_2_FL_func,X.T)))

        outputData = {
            "Time" : self.time,
            "u1" : U[0,:],
            "du1" : np.gradient(U[0,:],self.dt),
            "u2" : U[1,:],
            "du2" : np.gradient(U[1,:],self.dt),
            "x1" : X[0,:],
            "dx1" : X[1,:],
            "d2x1" : np.gradient(X[1,:],self.dt),
            "x3" : X[2,:],
            "dx3" : X[3,:],
            "d2x3" : np.gradient(X[3,:],self.dt),
            "x5" : X[4,:],
            "dx5" : X[5,:],
            "d2x5" : np.gradient(X[5,:],self.dt),
            "fT1" : fT1,
            "dfT1" : np.gradient(fT1,self.dt),
            "d2fT1" : np.gradient(np.gradient(fT1,self.dt),self.dt),
            "fT2" : fT2,
            "dfT2" : np.gradient(fT2,self.dt),
            "d2fT2" : np.gradient(np.gradient(fT2,self.dt),self.dt)
        }

        if additionalDict is not None:
            outputData.update(additionalDict)

        if path is not None:
            assert type(path)==str, "path must be a str."
            sio.savemat(path+"outputData.mat",outputData)
        else:
            sio.savemat("outputData.mat",outputData)

def sweep_plant():
    plantParams["dt"]=0.01
    plantParams["Stage Duration"] = 10
    plantParams["Number of Stiffness Stages"] = 100
    plantParams["Number of Angle Stages"] = 100
    plantParams["Simulation Duration"] = plantParams["Stage Duration"]*(
        plantParams["Number of Stiffness Stages"]
        + plantParams["Number of Angle Stages"]
    )
    plant = plant_pendulum_1DOF2DOF(plantParams)

    stiffnessMinimum = 50
    stiffnessMaximum = 150
    angleMinimum = np.pi/2 + np.pi/9
    angleMaximum = 3*np.pi/2 - np.pi/9
    x1o = np.pi
    X_o = [x1o,0,plant.rj*x1o/plant.rm,0,-plant.rj*x1o/plant.rm,0]
    plantParams["X_o"] = X_o

    # X,U,Y = plant.forward_simulation(X_o)

    desiredAngle = np.zeros((5,len(plant.time)))
    desiredStiffness = np.zeros((3,len(plant.time)))
    timeBreaks = np.array(list(range(
        0,len(plant.time),int(plantParams["Stage Duration"]/plantParams["dt"]
    ))))
    breakDuration = int(plantParams["Stage Duration"]/plantParams["dt"])

    angleSweep = np.concatenate([
        (angleMaximum-np.pi)*np.linspace(0,1,int(breakDuration/2)) + np.pi,
        (angleMinimum-np.pi)*np.linspace(0,1,breakDuration -int(breakDuration/2)) + np.pi
    ])
    constantAngleValues = np.linspace(angleMinimum,angleMaximum,plantParams["Number of Angle Stages"])
    stiffnessSweep = (stiffnessMaximum-stiffnessMinimum)*np.linspace(0,1,breakDuration) + stiffnessMinimum
    constantStiffnessValues = np.linspace(stiffnessMinimum,stiffnessMaximum,plantParams["Number of Stiffness Stages"])
    # Sweep Angles at different stiffness values
    for i in range(plantParams["Number of Stiffness Stages"]):
        desiredAngle[0,timeBreaks[i]:timeBreaks[i+1]] = angleSweep
        desiredStiffness[0,timeBreaks[i]:timeBreaks[i+1]] = constantStiffnessValues[i]*np.ones(breakDuration)
    for i in range(plantParams["Number of Angle Stages"]):
        j = plantParams["Number of Stiffness Stages"] + i
        desiredAngle[0,timeBreaks[j]:timeBreaks[j+1]] = constantAngleValues[i]*np.ones(breakDuration)
        desiredStiffness[0,timeBreaks[j]:timeBreaks[j+1]] = stiffnessSweep
    desiredAngle[0,-1]=desiredAngle[0,-2]
    desiredStiffness[0,-1]=desiredStiffness[0,-2]

    desiredAngle[0,:] = LP_filt(100, desiredAngle[0,:])
    desiredAngle[1,:] = np.gradient(desiredAngle[0,:],plantParams["dt"])
    desiredAngle[2,:] = np.gradient(desiredAngle[1,:],plantParams["dt"])
    desiredAngle[3,:] = np.gradient(desiredAngle[2,:],plantParams["dt"])
    desiredAngle[4,:] = np.gradient(desiredAngle[3,:],plantParams["dt"])

    desiredStiffness[0,:] = LP_filt(100, desiredStiffness[0,:])
    desiredStiffness[1,:] = np.gradient(desiredStiffness[0,:],plantParams["dt"])
    desiredStiffness[2,:] = np.gradient(desiredStiffness[1,:],plantParams["dt"])

    X_FBL,U_FBL,Y_FBL,X_measured = plant.forward_simulation_FL(X_o,desiredAngle,desiredStiffness)
    fig1 = plt.figure(figsize=(10,8))
    ax1=plt.gca()

    ax1.plot(plant.time,(180/np.pi)*Y_FBL[0,:].T,c="C0")
    ax1.plot(plant.time,(180/np.pi)*desiredAngle[0,:],c="C0",linestyle="--")
    ax1.set_title(r"$-$ Actual; --- Desired", fontsize=16)
    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="C0")
    ax1.set_ylabel('Position (deg.)', color="C0")
    # y1_min = np.floor((Y_FBL[0,:].min()*180/np.pi)/22.5)*22.5
    # y1_min = min([y1_min,np.floor((X1d[0,:].min()*180/np.pi)/22.5)*22.5])
    # y1_max = np.ceil((Y_FBL[0,:].max()*180/np.pi)/22.5)*22.5
    # y1_max = max([y1_max,np.ceil((X1d[0,:].max()*180/np.pi)/22.5)*22.5])
    y1_min = 0
    y1_max = 360
    yticks = np.arange(y1_min,y1_max+22.5,22.5)
    yticklabels = []
    for el in yticks:
        if el%45==0:
            yticklabels.append(str(int(el)) + r"$^\circ$")
        else:
            yticklabels.append("")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax2 = ax1.twinx()
    ax2.plot(plant.time,Y_FBL[1,:].T,c="C1")
    ax2.plot(plant.time,desiredStiffness[0,:],c="C1",linestyle="--")
    ax2.tick_params(axis='y', labelcolor="C1")
    ax2.set_ylabel('Stiffness (Nm/rad.)', color="C1")

    fig2 = plt.figure(figsize=(10,8))
    ax3=plt.gca()
    ax3.plot(plant.time,(180/np.pi)*(Y_FBL[0,:]-desiredAngle[0,:]).T,c="C0")
    ax3.set_title("Error", fontsize=16)
    ax3.set_xlabel("Time (s)")
    ax3.tick_params(axis='y', labelcolor="C0")
    ax3.set_ylabel('Positional Error (deg.)', color="C0")
    yticklabels = [str(el)+r"$^\circ$" for el in ax3.get_yticks()]
    ax3.set_yticklabels(yticklabels)
    ax4 = ax3.twinx()
    ax4.plot(plant.time,Y_FBL[1,:] - desiredStiffness[0,:],c="C1")
    ax4.tick_params(axis='y', labelcolor="C1")
    ax4.set_ylabel('Stiffness Error (Nm/rad.)', color="C1")
    ax4.set_ylim([-0.1,0.1])
    ax4.set_yticks([-0.1,-0.05,0,0.05,0.1])

    # plant1.plot_tendon_tension_deformation_curves(X,addTitle="(Unforced)")
    out = plant.plot_tendon_tension_deformation_curves(
        X_FBL,
        returnValues=True,
        addTitle="(Feedback Linearization)"
    )
    tendonDeformation1_FBL,tendonDeformation2_FBL = out[0]
    tendonTension1_FBL,tendonTension2_FBL = out[1]

    plant.plot_boundary_friction_func(X_FBL)

    plant.plot_motor_angles(X_FBL)

    plt.show()
    return(plant.time,X_FBL,U_FBL,Y_FBL,out[1],plant)

def test_plant(plantParams):
    plant1 = plant_pendulum_1DOF2DOF(plantParams)
    plant2 = plant_pendulum_1DOF2DOF(plantParams)

    x1o = np.pi
    X_o = [x1o,0,plant2.rj*x1o/plant2.rm,0,-plant2.rj*x1o/plant2.rm,0]
    plantParams["X_o"] = X_o

    X,U,Y = plant1.forward_simulation(X_o)

    delay = 3
    X1d = np.zeros((5,len(X[0,:])))
    X1d[0,:] = np.pi*np.ones((1,len(X[0,:])))
    ### first transition after delay
    X1d[0,int(delay/plant1.dt)] = (
        plant1.jointAngleRange*np.random.uniform(0,1) # random number inside
        + plant1.jointAngleBounds["LB"]
    )
    Sd = np.zeros((3,len(X[0,:])))
    Sd[0,:] = plant1.jointStiffnessBounds["LB"]*np.ones((1,len(X[0,:])))
    Sd[0,int(delay/plant1.dt)] = (
        plant1.jointStiffnessRange*np.random.uniform(0,1) # random number inside
        + plant1.jointStiffnessBounds["LB"]
    )
    for i in range(int(delay/plant1.dt)+1, len(X[0,:])):
        if np.random.uniform() < 0.00025: # change offset
            X1d[0,i] = (
                plant1.jointAngleRange*np.random.uniform(0,1) # random number inside
                + plant1.jointAngleBounds["LB"]
            )
            Sd[0,i] = (
                plant1.jointStiffnessRange*np.random.uniform(0,1) # random number inside
                + plant1.jointStiffnessBounds["LB"]
            )
        else: # stay at previous input
            X1d[0,i] = X1d[0,i-1]
            Sd[0,i] = Sd[0,i-1]

    X1d[0,:] = LP_filt(100,X1d[0,:])
    X1d[1,:] = np.gradient(X1d[0,:],plant1.dt)
    X1d[2,:] = np.gradient(X1d[1,:],plant1.dt)
    X1d[3,:] = np.gradient(X1d[2,:],plant1.dt)
    X1d[4,:] = np.gradient(X1d[3,:],plant1.dt)

    Sd[0,:] = LP_filt(100,Sd[0,:])
    Sd[1,:] = np.gradient(Sd[0,:],plant1.dt)
    Sd[2,:] = np.gradient(Sd[1,:],plant1.dt)

    # timeBreaks = [
    #     int(el*plantParams["Simulation Duration"]/plantParams["dt"])
    #     for el in [0, 0.13333, 0.21667, 0.41667, .57, .785, 1]
    # ]
    # breakDurations = np.diff(timeBreaks)
    #
    # X1d[0,timeBreaks[0]:timeBreaks[1]] = np.pi*np.ones(breakDurations[0])
    # X1d[0,timeBreaks[1]:timeBreaks[2]] = np.pi*np.ones(breakDurations[1]) - 1
    # X1d[0,timeBreaks[2]:timeBreaks[3]] = (
    #     np.pi
    #     + 0.5*np.sin(
    #         3*np.pi*np.arange(
    #             0,plant1.time[breakDurations[2]],plantParams["dt"]
    #         ) / 5
    #     )
    # )
    # X1d[0,timeBreaks[3]:timeBreaks[4]] = np.pi*np.ones(breakDurations[3]) + 1
    # X1d[0,timeBreaks[4]:timeBreaks[5]] = np.pi*np.ones(breakDurations[4]) + 0.5
    # X1d[0,timeBreaks[5]:timeBreaks[6]] = np.pi*np.ones(breakDurations[5])
    #
    # X1d[0,:] = LP_filt(100, X1d[0,:])
    # X1d[1,:] = np.gradient(X1d[0,:],plant2.dt)
    # X1d[2,:] = np.gradient(X1d[1,:],plant2.dt)
    # X1d[3,:] = np.gradient(X1d[2,:],plant2.dt)
    # X1d[4,:] = np.gradient(X1d[3,:],plant2.dt)
    #
    # Sd = np.zeros((3,len(X[0,:])))
    # Sd[0,:] = 32 - 20*np.cos(16*np.pi*plant1.time/25)
    # Sd[1,:] = np.gradient(Sd[0,:],plant1.dt)
    # Sd[2,:] = np.gradient(Sd[1,:],plant1.dt)


    # Sd[0,:] = 80 - 20*np.cos(16*np.pi*plant1.time/25)
    # Sd[1,:] = 64*np.pi*np.sin(16*np.pi*plant1.time/25)/5
    # Sd[2,:] = (4**5)*(np.pi**2)*np.cos(16*np.pi*plant1.time/25)/(5**3)

    X_FBL,U_FBL,Y_FBL,X_measured = plant2.forward_simulation_FL(X_o,X1d,Sd)
    fig1 = plt.figure(figsize=(10,8))
    ax1=plt.gca()

    ax1.plot(plant1.time,(180/np.pi)*Y_FBL[0,:].T,c="C0")
    ax1.plot(plant1.time,(180/np.pi)*X1d[0,:],c="C0",linestyle="--")
    ax1.set_title(r"$-$ Actual; --- Desired", fontsize=16)
    ax1.set_xlabel("Time (s)")
    ax1.tick_params(axis='y', labelcolor="C0")
    ax1.set_ylabel('Position (deg.)', color="C0")
    # y1_min = np.floor((Y_FBL[0,:].min()*180/np.pi)/22.5)*22.5
    # y1_min = min([y1_min,np.floor((X1d[0,:].min()*180/np.pi)/22.5)*22.5])
    # y1_max = np.ceil((Y_FBL[0,:].max()*180/np.pi)/22.5)*22.5
    # y1_max = max([y1_max,np.ceil((X1d[0,:].max()*180/np.pi)/22.5)*22.5])
    y1_min = 0
    y1_max = 360
    yticks = np.arange(y1_min,y1_max+22.5,22.5)
    yticklabels = []
    for el in yticks:
        if el%45==0:
            yticklabels.append(str(int(el)) + r"$^\circ$")
        else:
            yticklabels.append("")
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(yticklabels)
    ax2 = ax1.twinx()
    ax2.plot(plant1.time,Y_FBL[1,:].T,c="C1")
    ax2.plot(plant1.time,Sd[0,:],c="C1",linestyle="--")
    ax2.tick_params(axis='y', labelcolor="C1")
    ax2.set_ylabel('Stiffness (Nm/rad.)', color="C1")

    fig2 = plt.figure(figsize=(10,8))
    ax3=plt.gca()
    ax3.plot(plant1.time,(180/np.pi)*(Y_FBL[0,:]-X1d[0,:]).T,c="C0")
    ax3.set_title("Error", fontsize=16)
    ax3.set_xlabel("Time (s)")
    ax3.tick_params(axis='y', labelcolor="C0")
    ax3.set_ylabel('Positional Error (deg.)', color="C0")
    yticklabels = [str(el)+r"$^\circ$" for el in ax3.get_yticks()]
    ax3.set_yticklabels(yticklabels)
    ax4 = ax3.twinx()
    ax4.plot(plant1.time,Y_FBL[1,:] - Sd[0,:],c="C1")
    ax4.tick_params(axis='y', labelcolor="C1")
    ax4.set_ylabel('Stiffness Error (Nm/rad.)', color="C1")
    ax4.set_ylim([-0.1,0.1])
    ax4.set_yticks([-0.1,-0.05,0,0.05,0.1])

    # plant1.plot_tendon_tension_deformation_curves(X,addTitle="(Unforced)")
    out = plant2.plot_tendon_tension_deformation_curves(
        X_FBL,
        returnValues=True,
        addTitle="(Feedback Linearization)"
    )
    tendonDeformation1_FBL,tendonDeformation2_FBL = out[0]
    tendonTension1_FBL,tendonTension2_FBL = out[1]

    plant2.plot_motor_angles(X_FBL)

    plt.show()
    return(plant1.time,X1d,Sd,X_FBL,U_FBL,Y_FBL,plant1,plant2)

if __name__ == '__main__':
    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        plant.py

        -----------------------------------------------------------------------------

        A 1 DOF, 2 DOA tendon-driven system with nonlinear tendon
        elasticity in order to predict joint angle from different "sensory"
        states (like tendon tension or motor angle). This system can be controlled via Feedback linearization or by forward integration. The system will also stop when its

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/01/29)

        -----------------------------------------------------------------------------'''
        )
    )
    parser.add_argument(
        '-dt',
        metavar='timestep',
        type=float,
        nargs="?",
        help='Time step for the simulation (float). Default is given by plantParams.py',
        default=plantParams["dt"]
    )
    parser.add_argument(
        '-dur',
        metavar='duration',
        type=float,
        nargs="?",
        help='Duration of the simulation (float). Default is given by plantParams.py',
        default=plantParams["Simulation Duration"]
    )
    parser.add_argument(
        '--savefigs',
        action="store_true",
        help='Option to save figures for babbling trial. Default is false.'
    )
    parser.add_argument(
        '--animate',
        action="store_true",
        help='Option to animate trial. Default is false.'
    )

    args = parser.parse_args()
    plantParams["dt"] = args.dt
    plantParams["Simulation Duration"] = args.dur
    saveFigures = args.savefigs
    animate = args.animate

    time,X1d,Sd,X,U,Y,plant1,plant2 = test_plant()
    if saveFigures==True:
        save_figures(
            "visualizations/",
            "v1",
            plantParams,
            returnPath=False,
            saveAsPDF=True
        )
    if animate==True:
        downsamplingFactor = int(0.3/plant1.dt)
        Yd = np.concatenate([X1d[0,:,np.newaxis],Sd[0,:,np.newaxis]],axis=1).T
        ani = animate_pendulum(
            time[::downsamplingFactor],
            X[:,::downsamplingFactor],
            U[:,::downsamplingFactor],
            Y[:,::downsamplingFactor],
            Yd[:,::downsamplingFactor],
            **plantParams
        )
        ani.start(downsamplingFactor)
