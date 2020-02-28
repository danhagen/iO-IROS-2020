import numpy as np
import matplotlib.pyplot as plt
from math import acos,atan
from danpy.sb import *
from danpy.useful_functions import save_figures,is_number
from matplotlib.patches import Ellipse,Polygon
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from scipy import signal
from plantParams import *
from plant_discrete_contact import *
import sys
import argparse
# plt.rc('text', usetex=True)

class animate_pendulum_ANN:
    def __init__(
            self,
            Time,X,U,
            predictedAngle,
            downsamplingFactor,
            title,
            group,
            errorBounds,
            **plantParams
        ):
        colors = [
            "#2A3179", # all
            "#F4793B", # bio
            "#8DBDE6", # kinapprox
            "#A95AA1" # allmotor
        ]
        if group=='all':
            addTitle = "(All Motor and Tendon Tension States)"
            color = colors[0]
        elif group=='bio':
            addTitle = "(Bio-Inspired Set)"
            color = colors[1]
        elif group=='kinapprox':
            addTitle = "(Motor Position and Velocity Only)"
            color = colors[2]
        elif group=='allmotor':
            addTitle = "(All Motor States)"
            color = colors[3]
        self.L = 25
        self.Lcm = 15.1
        self.rj = 6.95
        self.rm = 4.763
        self.Lsp_Total = 29.033699471476247

        self.X = X[:,::downsamplingFactor]
        self.U = U[:,::downsamplingFactor]
        self.Time = Time - Time[0]
        self.endTime = self.Time[-1]
        self.downsampledTime = self.Time[::downsamplingFactor]
        self.downsamplingFactor = downsamplingFactor
        self.x1Desired = self.X[0,:]
        self.x1Predicted = predictedAngle[::downsamplingFactor]
        self.minimumError = errorBounds[0] # in degrees
        self.maximumError = errorBounds[1] # in degrees

        self.fig = plt.figure(figsize=(12,12*9/16))
        gs = self.fig.add_gridspec(2,3)
        self.ax0 = self.fig.add_axes((-0.01,-0.1,0.6,1.1)) #animation
        self.ax2 = self.fig.add_axes((0.6,0.55,0.35,0.35)) #animation
        self.ax3 = self.fig.add_axes((0.6,0.1,0.35,0.35)) #animation

        # self.ax0 = self.fig.add_subplot(gs[:, :2]) # animation
        # self.ax2 = self.fig.add_subplot(gs[0, 2]) # desired vs. predicted
        # self.ax3 = self.fig.add_subplot(gs[1, 2]) # error

        # self.ax0 = self.fig.add_subplot(221) # animation
        # self.ax1 = self.fig.add_subplot(222) # input (torques)
        # self.ax2 = self.fig.add_subplot(223) # desired vs. predicted pendulum angle
        # self.ax3 = self.fig.add_subplot(224) # prediction error

        self.ax0.set_title(title,fontsize=26,y=1.05)
        self.ax0.text(0,35,addTitle,
            {'color': color, 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )

        groundVertices = np.array([
            [-34.289,-13.144],
            [-34.289,-13.144+3.369],
            [-34.289+23.8995,-13.144+12.006],
            [34.289-23.8995,-13.144+12.006],
            [34.289,-13.144+3.369],
            [34.289,-13.144]
        ])
        self.pendulumGround = Polygon(
            groundVertices,
            facecolor='#888A8C',
            edgecolor='k',
            linewidth=1
        )
        self.ax0.add_patch(self.pendulumGround)
        self.ax0.plot(
            [-34.289-2.474,-34.289],
            [-13.144,-13.144],
            'k',
            linewidth=1
        )
        self.ax0.plot(
            [-34.289-2.474-4.163,-34.289-2.474],
            [-13.144,-13.144],
            'k--',
            linewidth=1
        )
        self.ax0.plot(
            [34.289,34.289+2.474],
            [-13.144,-13.144],
            'k',
            linewidth=1
        )
        self.ax0.plot(
            [34.289+2.474,34.289+2.474+4.163],
            [-13.144,-13.144],
            'k--',
            linewidth=1
        )
        # Tendon
        self.jointWrapAround, = self.ax0.plot(
            self.rj*np.sin(np.linspace(0,2*np.pi,1001)),
            self.rj*np.cos(np.linspace(0,2*np.pi,1001)),
            c='k',
            lw=1,
            zorder=3
        )
        self.motor1Center = [
            -28.13,
            -7.515
        ]
        self.motor1WrapAround, = self.ax0.plot(
            self.motor1Center[0] + self.rm*np.sin(np.linspace(0,2*np.pi,1001)),
            self.motor1Center[1] + self.rm*np.cos(np.linspace(0,2*np.pi,1001)),
            c='k',
            lw=1
        )
        self.motor2Center = [
            28.13,
            -7.515
        ]
        self.motor2WrapAround, = self.ax0.plot(
            self.motor2Center[0] + self.rm*np.sin(np.linspace(0,2*np.pi,1001)),
            self.motor2Center[1] + self.rm*np.cos(np.linspace(0,2*np.pi,1001)),
            c='k',
            lw=1
        )
        self.phi1 = (
            atan(abs(self.motor1Center[1]/self.motor1Center[0]))
            + np.pi
            - atan(self.Lsp_Total/(self.rj-self.rm))
        )
        self.phi2 = (
            atan(self.Lsp_Total/(self.rj-self.rm))
            - atan(abs(self.motor2Center[1]/self.motor2Center[0]))
        )

        springHeight = 1.9567
        self.springArray = (
            springHeight*np.abs(
                signal.sawtooth(5*2*np.pi*np.linspace(0,1,1001)-np.pi/2)
            )
            -0.5*springHeight
        )
        self.spring_y = np.concatenate(
            [
                np.zeros((1001,)),
                self.springArray,
                np.zeros((1001,))
            ]
        )
        self.tendonDeformation1 = self.rm*self.X[2,:]-self.rj*self.X[0,:]
        self.tendonDeformation2 = self.rm*self.X[4,:]+self.rj*self.X[0,:]
        self.maxTendonDeformation = max([
            self.tendonDeformation1.max(),
            self.tendonDeformation2.max()
        ])
        self.spring1_x = np.concatenate(
            [
                np.linspace(
                    0,
                    self.Lsp_Total/6*(2-self.tendonDeformation1/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.Lsp_Total/6*(2-self.tendonDeformation1/self.maxTendonDeformation),
                    self.Lsp_Total/6*(4+self.tendonDeformation1/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.Lsp_Total/6*(4+self.tendonDeformation1/self.maxTendonDeformation),
                    self.Lsp_Total,
                    1001
                )
            ]
        )
        self.spring2_x = np.concatenate(
            [
                np.linspace(
                    0,
                    self.Lsp_Total/6*(2-self.tendonDeformation2/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.Lsp_Total/6*(2-self.tendonDeformation2/self.maxTendonDeformation),
                    self.Lsp_Total/6*(4+self.tendonDeformation2/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.Lsp_Total/6*(4+self.tendonDeformation2/self.maxTendonDeformation),
                    self.Lsp_Total,
                    1001
                )
            ]
        )
        self.spring1Arrays = (
            np.array([
                [self.motor1Center[0] + self.rm*np.cos(self.phi1)],
                [self.motor1Center[1] + self.rm*np.sin(self.phi1)]
            ])
            + np.array([
                [np.cos(self.phi1-90*np.pi/180),-np.sin(self.phi1-90*np.pi/180)],
                [np.sin(self.phi1-90*np.pi/180),np.cos(self.phi1-90*np.pi/180)]
            ])@np.array([self.spring1_x[:,0],self.spring_y])
        )
        self.spring2Arrays = (
            np.array([
                [self.motor2Center[0] + self.rm*np.cos(self.phi2)],
                [self.motor2Center[1] + self.rm*np.sin(self.phi2)]
            ])
            + np.array([
                [np.cos(self.phi2+90*np.pi/180),-np.sin(self.phi2+90*np.pi/180)],
                [np.sin(self.phi2+90*np.pi/180),np.cos(self.phi2+90*np.pi/180)]
            ])@np.array([self.spring2_x[:,0],self.spring_y])
        )
        self.spring1, = self.ax0.plot(
            self.spring1Arrays[0,:],
            self.spring1Arrays[1,:],
            c='k',
            lw=1
        )
        self.spring2, = self.ax0.plot(
            self.spring2Arrays[0,:],
            self.spring2Arrays[1,:],
            c='k',
            lw=1
        )

        # Pendulum

        self.pendulum_Predicted, = self.ax0.plot(
            [0,self.L*np.sin(self.x1Predicted[0])],
            [0,-self.L*np.cos(self.x1Predicted[0])],
            Color=color,
            lw = 20,
            solid_capstyle='round',
            alpha=0.5,
            zorder=1
        )

        self.pendulum_Desired, = self.ax0.plot(
            [0,self.L*np.sin(X[0,0])],
            [0,-self.L*np.cos(X[0,0])],
            Color='#85B8D0',
            lw = 20,
            solid_capstyle='round',
            zorder=1,
            path_effects=[pe.Stroke(linewidth=22, foreground='k'), pe.Normal()]
        )

        # Pendulum Joint

        self.pendulumJoint = plt.Circle((0,0),self.rj,Color='#D1D3D4',zorder=3)
        self.ax0.add_patch(self.pendulumJoint)

        self.pendulumJointRivet, = self.ax0.plot(
            [0],
            [0],
            c='k',
            marker='o',
            lw=2,
            zorder=3
        )

        # Motors
        self.motor1Joint = plt.Circle((self.motor1Center[0],self.motor1Center[1]),self.rm,Color='#D1D3D4')
        self.ax0.add_patch(self.motor1Joint)

        self.motor1Rivet, = self.ax0.plot(
            [self.motor1Center[0]],
            [self.motor1Center[1]],
            c='k',
            marker='.',
            lw=1
        )

        self.motor2Joint = plt.Circle((self.motor2Center[0],self.motor2Center[1]),self.rm,Color='#D1D3D4')
        self.ax0.add_patch(self.motor2Joint)

        self.motor2Rivet, = self.ax0.plot(
            [self.motor2Center[0]],
            [self.motor2Center[1]],
            c='k',
            marker='.',
            lw=1
        )

        self.max_tau = self.U.max()
        if self.max_tau==0: self.max_tau=1

        self.k = 0.075*self.L
        self.input1Label = self.ax0.text(
            self.motor1Center[0] + 2.5*self.rm*np.cos(self.phi1-np.pi/18),
            self.motor1Center[1] + 2.5*self.rm*np.sin(self.phi1-np.pi/18),
            r'$\tau_1$',
            {
                'color': 'red',
                'fontsize': 16,
                'ha': 'center',
                'va': 'center'
            }
        )
        self.inputIndicator1, = self.ax0.plot(
            (
                self.motor1Center[0]
                + 2.5*self.rm*np.cos(
                    np.linspace(
                        self.phi1,
                        self.phi1 + (np.pi-self.phi1)*self.U[0,0]/self.max_tau,
                        20
                    )
                )
            ),
            (
                self.motor1Center[1]
                + 2.5*self.rm*np.sin(
                    np.linspace(
                        self.phi1,
                        self.phi1 + (np.pi-self.phi1)**self.U[0,0]/self.max_tau,
                        20
                    )
                )
            ),
            Color='r',
            lw = 2,
            solid_capstyle = 'round'
            )
        self.inputIndicator1Arrow, = self.ax0.plot(
            (
                self.motor1Center[0]
                + 2.5*self.rm*np.cos(self.phi1 + (np.pi-self.phi1)*self.U[0,0]/self.max_tau)
                + self.k*(self.U[0,0]/self.max_tau)**(1/2)*np.array([
                    np.cos(self.phi1 + (np.pi-self.phi1)*self.U[0,0]/self.max_tau -np.pi/2 - 1.33*30*np.pi/180),
                    0,
                    np.cos(self.phi1 + (np.pi-self.phi1)*self.U[0,0]/self.max_tau -np.pi/2 + 30*np.pi/180)
                ])
            ),
            (
                self.motor1Center[1]
                + 2.5*self.rm*np.sin(self.phi1 + (np.pi-self.phi1)*self.U[0,0]/self.max_tau)
                + self.k*(self.U[0,0]/self.max_tau)**(1/2)*np.array([
                    np.sin(self.phi1 + (np.pi-self.phi1)*self.U[0,0]/self.max_tau -np.pi/2 - 1.33*30*np.pi/180),
                    0,
                    np.sin(self.phi1 + (np.pi-self.phi1)*self.U[0,0]/self.max_tau -np.pi/2 + 30*np.pi/180)
                ])
            ),
            Color='r',
            lw = 2,
            solid_capstyle='round'
        )

        self.input2Label = self.ax0.text(
            self.motor2Center[0] + 2.5*self.rm*np.cos(self.phi2+np.pi/16),
            self.motor2Center[1] + 2.5*self.rm*np.sin(self.phi2+np.pi/16),
            r'$\tau_2$',
            {
                'color': 'red',
                'fontsize': 16,
                'ha': 'center',
                'va': 'center'
            }
        )
        self.inputIndicator2, = self.ax0.plot(
            (
                self.motor2Center[0]
                + 2.5*self.rm*np.cos(
                    np.linspace(
                        self.phi2-self.phi2*self.U[1,0]/self.max_tau,
                        self.phi2,
                        20
                    )
                )
            ),
            (
                self.motor2Center[1]
                + 2.5*self.rm*np.sin(
                    np.linspace(
                        self.phi2-self.phi2*self.U[1,0]/self.max_tau,
                        self.phi2,
                        20
                    )
                )
            ),
            Color='r',
            lw = 2,
            solid_capstyle = 'round'
            )
        self.inputIndicator2Arrow, = self.ax0.plot(
            (
                self.motor2Center[0]
                + 2.5*self.rm*np.cos(self.phi2 - self.phi2*self.U[1,0]/self.max_tau)
                + self.k*(self.U[1,0]/self.max_tau)**(1/2)*np.array([
                    np.cos(self.phi2 - self.phi2*self.U[1,0]/self.max_tau + np.pi/2 - 30*np.pi/180),
                    0,
                    np.cos(self.phi2 - self.phi2*self.U[1,0]/self.max_tau + np.pi/2 + 1.33*30*np.pi/180)
                ])
            ),
            (
                self.motor2Center[1]
                + 2.5*self.rm*np.sin(45*np.pi/180 - self.phi2*self.U[1,0]/self.max_tau)
                + self.k*(self.U[1,0]/self.max_tau)**(1/2)*np.array([
                    np.sin(self.phi2 - self.phi2*self.U[1,0]/self.max_tau + np.pi/2 - 30*np.pi/180),
                    0,
                    np.sin(self.phi2 - self.phi2*self.U[1,0]/self.max_tau + np.pi/2 + 1.33*30*np.pi/180)
                ])
            ),
            Color='r',
            lw = 2,
            solid_capstyle='round'
        )

        self.ax0.get_xaxis().set_ticks([])
        self.ax0.get_yaxis().set_ticks([])
        # self.ax0.set_frame_on(True)
        self.ax0.spines['top'].set_visible(False)
        self.ax0.spines['bottom'].set_visible(False)
        self.ax0.spines['left'].set_visible(False)
        self.ax0.spines['right'].set_visible(False)
        self.ax0.set_xlim([-45,45])
        self.ax0.set_ylim([-17,35])
        self.ax0.set_aspect('equal')

        self.timeStamp = self.ax0.text(
            0,-17,
            "Time: "+str(self.downsampledTime[0])+" s",
            {'color': "0.50", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )

        #pendulum angle (Dont use downsampled data)
        self.x1DesiredInDegrees = X[0,:]*180/np.pi
        self.x1PredictedInDegrees = predictedAngle*180/np.pi
        self.desiredAngle, = self.ax2.plot(self.Time,self.x1DesiredInDegrees,c='0.70',lw=2)
        self.angle, = self.ax2.plot([0],[self.x1PredictedInDegrees[0]],color = color)
        self.ax2.set_xlim(0,self.endTime)
        self.ax2.set_xticks(list(np.linspace(0,self.endTime,5)))
        self.ax2.set_xticklabels([str(0),'','','',str(self.endTime)+"s"])
        self.ax2.set_ylim([
            90-0.1*180,
            270+0.1*180
        ])
        y1_min = 90
        y1_max = 270
        yticks = np.arange(y1_min,y1_max+22.5,22.5)
        yTickLabels = []
        for el in yticks:
        	if el%45==0:
        		yTickLabels.append(str(int(el-180)) + r"$^\circ$")
        	else:
        		yTickLabels.append("")
        self.ax2.set_yticks(yticks)
        self.ax2.set_yticklabels(yTickLabels)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.set_title(
            "Predicted Joint Angle (deg)",
            fontsize=16,
            fontweight=4,
            color=color,
            y=0.95
        )
        self.ax2.text(
            self.endTime/2,
            0.925*self.ax2.get_ylim()[1],
            "(Desired)",
            {'color': "0.70", 'fontsize': 14, 'ha': 'center', 'va': 'center'}
        )

        # Prediction Error
        self.errorArrayInDegrees = self.x1DesiredInDegrees - self.x1PredictedInDegrees
        self.error, = self.ax3.plot([0],[self.errorArrayInDegrees[0]],color=color)
        self.ax3.set_xlim(0,self.endTime)
        self.ax3.set_xticks(list(np.linspace(0,self.endTime,5)))
        self.ax3.set_xticklabels([str(0),'','','',str(self.endTime)+"s"])
        self.rangeError= self.maximumError-self.minimumError
        self.ax3.set_ylim([
            self.minimumError - 0.1*self.rangeError,
            self.maximumError + 0.1*self.rangeError
        ])
        self.ax3.set_yticks(
            np.arange(
                np.ceil((self.minimumError-0.1*self.rangeError)/5)*5,
                np.floor((self.maximumError+0.1*self.rangeError)/5)*5 + 1e-3,
                5
            )
        )
        yTickLabels = []
        for tick in self.ax3.get_yticks():
            if tick%10==0:
                yTickLabels.append(str(int(tick))+ r"$^\circ$")
            else:
                yTickLabels.append("")
        self.ax3.set_yticklabels(yTickLabels)
        self.ax3.spines['right'].set_visible(False)
        self.ax3.spines['top'].set_visible(False)
        self.ax3.set_title(
            "Prediction Error (deg.)",
            fontsize=16,
            fontweight=4,
            color=color,
            y=0.95
        )

    def animate(self,i):
        self.spring1Arrays = (
            np.array([
                [self.motor1Center[0] + self.rm*np.cos(self.phi1)],
                [self.motor1Center[1] + self.rm*np.sin(self.phi1)]
            ])
            + np.array([
                [np.cos(self.phi1-90*np.pi/180),-np.sin(self.phi1-90*np.pi/180)],
                [np.sin(self.phi1-90*np.pi/180),np.cos(self.phi1-90*np.pi/180)]
            ])@np.array([self.spring1_x[:,i],self.spring_y])
        )
        self.spring1.set_xdata(self.spring1Arrays[0,:])
        self.spring1.set_ydata(self.spring1Arrays[1,:])

        self.spring2Arrays = (
            np.array([
                [self.motor2Center[0] + self.rm*np.cos(self.phi2)],
                [self.motor2Center[1] + self.rm*np.sin(self.phi2)]
            ])
            + np.array([
                [np.cos(self.phi2+90*np.pi/180),-np.sin(self.phi2+90*np.pi/180)],
                [np.sin(self.phi2+90*np.pi/180),np.cos(self.phi2+90*np.pi/180)]
            ])@np.array([self.spring2_x[:,i],self.spring_y])
        )
        self.spring2.set_xdata(self.spring2Arrays[0,:])
        self.spring2.set_ydata(self.spring2Arrays[1,:])

        self.pendulum_Desired.set_xdata([0,self.L*np.sin(self.X[0,i])])
        self.pendulum_Desired.set_ydata([0,-self.L*np.cos(self.X[0,i])])

        self.pendulum_Predicted.set_xdata([0,self.L*np.sin(self.x1Predicted[i])])
        self.pendulum_Predicted.set_ydata([0,-self.L*np.cos(self.x1Predicted[i])])

        self.inputIndicator1.set_xdata(
            self.motor1Center[0]
            + 2.5*self.rm*np.cos(
                np.linspace(
                    self.phi1,
                    self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau,
                    20
                )
            )
        )
        self.inputIndicator1.set_ydata(
            self.motor1Center[1]
            + 2.5*self.rm*np.sin(
                np.linspace(
                    self.phi1,
                    self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau,
                    20
                )
            )
        )

        self.inputIndicator1Arrow.set_xdata(
            self.motor1Center[0]
            + 2.5*self.rm*np.cos(self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau)
            + self.k*(self.U[0,i]/self.max_tau)**(1/2)*np.array([
                np.cos(self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau -np.pi/2 - 1.33*30*np.pi/180),
                0,
                np.cos(self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau -np.pi/2 + 30*np.pi/180)
            ])
        )
        self.inputIndicator1Arrow.set_ydata(
            self.motor1Center[1]
            + 2.5*self.rm*np.sin(self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau)
            + self.k*(self.U[0,i]/self.max_tau)**(1/2)*np.array([
                np.sin(self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau -np.pi/2 - 1.33*30*np.pi/180),
                0,
                np.sin(self.phi1 + (np.pi-self.phi1)*self.U[0,i]/self.max_tau -np.pi/2 + 30*np.pi/180)
            ])
        )

        self.inputIndicator2.set_xdata(
            self.motor2Center[0]
            + 2.5*self.rm*np.cos(
                np.linspace(
                    self.phi2-self.phi2*self.U[1,i]/self.max_tau,
                    self.phi2,
                    20
                )
            )
        )
        self.inputIndicator2.set_ydata(
            self.motor2Center[1]
            + 2.5*self.rm*np.sin(
                np.linspace(
                    self.phi2-self.phi2*self.U[1,i]/self.max_tau,
                    self.phi2,
                    20
                )
            )
        )

        self.inputIndicator2Arrow.set_xdata(
            self.motor2Center[0]
            + 2.5*self.rm*np.cos(self.phi2 - self.phi2*self.U[1,i]/self.max_tau)
            + self.k*(self.U[1,i]/self.max_tau)**(1/2)*np.array([
                np.cos(self.phi2 - self.phi2*self.U[1,i]/self.max_tau + np.pi/2 - 30*np.pi/180),
                0,
                np.cos(self.phi2 - self.phi2*self.U[1,i]/self.max_tau + np.pi/2 + 1.33*30*np.pi/180)
            ])
        )
        self.inputIndicator2Arrow.set_ydata(
            self.motor2Center[1]
            + 2.5*self.rm*np.sin(self.phi2 - self.phi2*self.U[1,i]/self.max_tau)
            + self.k*(self.U[1,i]/self.max_tau)**(1/2)*np.array([
                np.sin(self.phi2 - self.phi2*self.U[1,i]/self.max_tau + np.pi/2 - 30*np.pi/180),
                0,
                np.sin(self.phi2 - self.phi2*self.U[1,i]/self.max_tau + np.pi/2 + 1.33*30*np.pi/180)
            ])
        )


        self.timeStamp.set_text(
            "Time: "+"{:.2f}".format(self.downsampledTime[i])+" s",
        )

        self.angle.set_xdata(self.Time[:(i*self.downsamplingFactor)])
        self.angle.set_ydata(
            self.x1PredictedInDegrees[:(i*self.downsamplingFactor)]
        )

        self.error.set_xdata(
            self.Time[:(i*self.downsamplingFactor)]
        )
        self.error.set_ydata(
            self.errorArrayInDegrees[:(i*self.downsamplingFactor)]
        )

        return self.pendulum_Predicted,self.pendulum_Desired,self.spring1,self.spring2,self.inputIndicator1,self.inputIndicator1Arrow,self.inputIndicator2,self.inputIndicator2Arrow,self.angle,self.error,self.timeStamp,

    def start(self,interval):
        self.anim = animation.FuncAnimation(self.fig, self.animate,
            frames=len(self.downsampledTime), interval=interval, blit=False)

if __name__ == '__main__':
    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        animate.py

        -----------------------------------------------------------------------------

        Animation for 1 DOF, 2 DOA tendon-driven system with nonlinear tendon
        elasticity.

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/01/29)

        -----------------------------------------------------------------------------'''
        )
    )
    parser.add_argument(
        '-dt',
        type=float,
        help='Time step for the simulation (float). Default is given by plantParams.py',
        default=plantParams["dt"]
    )
    parser.add_argument(
        '-dur',
        type=float,
        help='Duration of the simulation (float). Default is given by plantParams.py',
        default=plantParams["Simulation Duration"]
    )
    args = parser.parse_args()
    plantParams["dt"] = args.dt
    plantParams["Simulation Duration"] = args.dur

    Time,X1d,Sd,X,U,Y,plant1,plant2 = test_plant(plantParams)
    if len(sys.argv)-1!=0:
        if '--savefigs' in sys.argv:
            save_figures(
                "visualizations/",
                "v0",
                plantParams,
                returnPath=False,
                saveAsPDF=True
            )
        if '--animate' in sys.argv:
            downsamplingFactor = int(0.3/plantParams["dt"])
            ani = animate_pendulum(
                Time[::downsamplingFactor],
                X[:,::downsamplingFactor],
                U[:,::downsamplingFactor],
                Y[:,::downsamplingFactor],
                plant2.desiredOutput[:,::downsamplingFactor],
                **plantParams
            )
            ani.start(downsamplingFactor)
            # ani.anim.save('test.mp4', writer='ffmpeg', fps=1000/downsamplingFactor)
    plt.show()
