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
from plant import *
import sys
import argparse
# plt.rc('text', usetex=True)

class animate_pendulum_babbling:
    def __init__(
            self,
            Time,X,U,
            downsamplingFactor,
            **plantParams
        ):

        self.L = 25
        self.Lcm = 15.1
        self.rj = 6.95
        self.rm = 4.763
        self.Lsp_Total = 29.033699471476247

        self.X = X[:,::downsamplingFactor]
        self.U = U[:,::downsamplingFactor]
        self.Time = Time - Time[0]
        self.endTime = self.Time[-1]
        self.Time = self.Time[::downsamplingFactor]
        inputBounds = plantParams.get("Input Bounds",[0,10])
        inputMaximum = inputBounds[1]
        inputMinimum = inputBounds[0]

        self.fig = plt.figure(figsize=(12,12*9/16))
        gs = self.fig.add_gridspec(2,3)
        self.ax0 = self.fig.add_axes((1.01-0.6,-0.1,0.6,1.1)) #animation
        self.ax1 = self.fig.add_axes((0.05,0.70,0.35,0.25)) # input 1
        self.ax2 = self.fig.add_axes((0.05,0.375,0.35,0.25)) # input 2
        self.ax3 = self.fig.add_axes((0.05,0.05,0.35,0.25)) # joint angle

        self.ax0.set_title("Motor Babbling",fontsize=26,y=1.05)
        totalTime = plantParams.get("Simulation Duration",None)
        if totalTime is not None:
            self.ax0.text(0,35,"("+str(totalTime)+"s Total)",
                {'color': "0.50", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
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

        self.pendulum, = self.ax0.plot(
            [0,self.L*np.sin(self.X[0,0])],
            [0,-self.L*np.cos(self.X[0,0])],
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
            "Time: "+str(self.Time[0])+" s",
            {'color': "r", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )

        # Input 1
        self.ax1.plot([0,self.endTime],[inputMaximum]*2,'0.70',linestyle='--')
        self.ax1.plot([0,self.endTime],[inputMinimum]*2,'0.70',linestyle='--')
        self.ax1.text(
            0.9*self.endTime,
            inputMaximum - 0.1*(inputMaximum-inputMinimum),
            r"$\tau_{\max}$",
            {'color': "0.70", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )
        self.ax1.text(
            0.9*self.endTime,
            inputMinimum + 0.1*(inputMaximum-inputMinimum),
            r"$\tau_{\min}$",
            {'color': "0.70", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )
        self.input1, = self.ax1.plot([0],[self.U[0,0]],'r')
        self.ax1.set_xlim(0,self.Time[-1])
        self.ax1.set_xticks(list(np.linspace(0,self.Time[-1],5)))
        self.ax1.set_xticklabels([str(0),'','','',str(self.Time[-1])+"s"])
        if max(abs(self.U[0,:] - self.U[0,0]))<1e-7 and max(abs(self.U[1,:] - self.U[1,0]))<1e-7:
            self.ax1.set_ylim([min(self.U[:,0]) - 5,max(self.U[:,0]) + 5])
        else:
            self.RangeU = inputMaximum-inputMinimum
            self.ax1.set_ylim([self.U.min()-0.1*self.RangeU,self.U.max()+0.1*self.RangeU])
        #
        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.set_title("Input 1 (Nm)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

        # Input 2
        self.ax2.plot([0,self.endTime],[inputMaximum]*2,'0.70',linestyle='--')
        self.ax2.plot([0,self.endTime],[inputMinimum]*2,'0.70',linestyle='--')
        self.ax2.text(
            0.9*self.endTime,
            inputMaximum - 0.1*(inputMaximum-inputMinimum),
            r"$\tau_{\max}$",
            {'color': "0.70", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )
        self.ax2.text(
            0.9*self.endTime,
            inputMinimum + 0.1*(inputMaximum-inputMinimum),
            r"$\tau_{\min}$",
            {'color': "0.70", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )
        self.input2, = self.ax2.plot([0],[self.U[1,0]],'r')
        self.ax2.set_xlim(0,self.Time[-1])
        self.ax2.set_xticks(list(np.linspace(0,self.Time[-1],5)))
        self.ax2.set_xticklabels([str(0),'','','',str(self.Time[-1])+"s"])
        if max(abs(self.U[0,:] - self.U[0,0]))<1e-7 and max(abs(self.U[1,:] - self.U[1,0]))<1e-7:
            self.ax2.set_ylim([min(self.U[:,0]) - 5,max(self.U[:,0]) + 5])
        else:
            self.RangeU = inputMaximum-inputMinimum
            self.ax2.set_ylim([self.U.min()-0.1*self.RangeU,self.U.max()+0.1*self.RangeU])
        #
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.set_title("Input 2 (Nm)",fontsize=16,fontweight = 4,color = 'r',y = 0.95)

        #pendulum angle
        self.x1d = self.X[0,:]*180/np.pi
        self.ax3.plot([0,self.endTime],[90,90],"0.70",linestyle='--')
        self.ax3.plot([0,self.endTime],[270,270],"0.70",linestyle='--')
        self.ax3.text(
            0.9*self.endTime,
            270 - 0.1*(180),
            r"$\theta_{j,\max}$",
            {'color': "0.70", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )
        self.ax3.text(
            0.9*self.endTime,
            90 + 0.1*(180),
            r"$\theta_{j,\min}$",
            {'color': "0.70", 'fontsize': 16, 'ha': 'center', 'va': 'center'}
        )
        self.angle, = self.ax3.plot([0],[self.x1d[0]],color = 'k')
        self.ax3.set_xlim(0,self.endTime)
        self.ax3.set_xticks(list(np.linspace(0,self.endTime,5)))
        self.ax3.set_xticklabels([str(0),'','','',str(self.endTime)+"s"])
        if max(abs(self.x1d-self.x1d[0]))<1e-7:
            self.ax3.set_ylim([
                self.x1d[0]-2,
                self.x1d[0]+2
            ])
        else:
            self.Rangex1d= (
                max(self.x1d)
                - min(self.x1d)
            )
            self.ax3.set_ylim([
                (
                    min(self.x1d)
                    - 0.1*self.Rangex1d
                ),
                (
                    max(self.x1d)
                    + 0.1*self.Rangex1d
                )
            ])
            y1_min = 90
            y1_max = 270
            yticks = np.arange(y1_min,y1_max+22.5,22.5)
            yticklabels = []
            for el in yticks:
            	if el%45==0:
            		yticklabels.append(str(int(el-180)) + r"$^\circ$")
            	else:
            		yticklabels.append("")
            self.ax3.set_yticks(yticks)
            self.ax3.set_yticklabels(yticklabels)
        self.ax3.spines['right'].set_visible(False)
        self.ax3.spines['top'].set_visible(False)
        self.ax3.set_title("Joint Angle (deg)",fontsize=16,fontweight = 4,color = 'k',y = 0.95)

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

        self.pendulum.set_xdata([0,self.L*np.sin(self.X[0,i])])
        self.pendulum.set_ydata([0,-self.L*np.cos(self.X[0,i])])

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

        self.timeStamp.set_text("Time: "+"{:.2f}".format(self.Time[i])+" s",)

        self.input1.set_xdata(self.Time[:i])
        self.input1.set_ydata(self.U[0,:i])

        self.input2.set_xdata(self.Time[:i])
        self.input2.set_ydata(self.U[1,:i])

        self.angle.set_xdata(self.Time[:i])
        self.angle.set_ydata(self.x1d[:i])

        return self.pendulum,self.spring1,self.spring2,self.inputIndicator1,self.inputIndicator1Arrow,self.inputIndicator2,self.inputIndicator2Arrow,self.input1,self.input2,self.angle,self.timeStamp,

    def start(self,interval):
        self.anim = animation.FuncAnimation(self.fig, self.animate,
            frames=len(self.Time)-1, interval=interval, blit=False)

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
