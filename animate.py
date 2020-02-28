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

class animate_pendulum:
    def __init__(self,Time,X,U,Y,desiredOutput,**plantParams):
        self.L = plantParams.get("Link Length", 0.3)
        is_number(self.L,"Link Length",default=0.3)

        self.Lcm = plantParams.get("Link Center of Mass",0.085)
        is_number(self.Lcm,"Link Center of Mass",default=0.085)

        self.rj = plantParams.get("Joint Moment Arm", 0.05)
        is_number(self.rj,"Joint Moment Arm",default=0.05)

        self.rm = plantParams.get("Motor Moment Arm", 0.02)
        is_number(self.rm,"Motor Moment Arm",default=0.02)

        self.X = X
        self.U = U
        self.Y = Y
        self.Time = Time
        self.x1d = desiredOutput[0,:]
        self.Sd = desiredOutput[1,:]

        self.fig = plt.figure(figsize=(12,10))
        self.ax0 = self.fig.add_subplot(221) # animation
        self.ax1 = self.fig.add_subplot(222) # input (torques)
        self.ax2 = self.fig.add_subplot(223) # pendulum angle
        self.ax4 = self.fig.add_subplot(224) # stiffness

        plt.suptitle("Nonlinear Tendon-Driven Pendulum Example",fontsize=28,y=1.05)

        self.pendulumGround = plt.Rectangle(
            (-0.25*self.L,-self.rj),
            0.5*self.L,
            self.rj,
            Color='0.20'
        )
        self.ax0.add_patch(self.pendulumGround)

        # Tendon
        self.jointWrapAround, = self.ax0.plot(
            self.rj*np.sin(np.linspace(0,2*np.pi,1001)),
            self.rj*np.cos(np.linspace(0,2*np.pi,1001)),
            c='k',
            lw=1
        )
        self.motor1WrapAround, = self.ax0.plot(
            -self.L + self.rm*np.sin(np.linspace(0,2*np.pi,1001)),
            -self.L + self.rm*np.cos(np.linspace(0,2*np.pi,1001)),
            c='k',
            lw=1
        )
        self.motor2WrapAround, = self.ax0.plot(
            self.L + self.rm*np.sin(np.linspace(0,2*np.pi,1001)),
            -self.L + self.rm*np.cos(np.linspace(0,2*np.pi,1001)),
            c='k',
            lw=1
        )
        self.phi1 = 225*np.pi/180 - acos((self.rj-self.rm)/(np.sqrt(2)*self.L))
        self.phi2 = acos((self.rj-self.rm)/(np.sqrt(2)*self.L)) - 45*np.pi/180

        self.springArray = (
            self.rm*np.abs(
                signal.sawtooth(5*2*np.pi*np.linspace(0,1,1001)-np.pi/2)
            )
            -0.5*self.rm
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
                    self.L*np.sqrt(2)/6*(2-self.tendonDeformation1/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.L*np.sqrt(2)/6*(2-self.tendonDeformation1/self.maxTendonDeformation),
                    self.L*np.sqrt(2)/6*(4+self.tendonDeformation1/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.L*np.sqrt(2)/6*(4+self.tendonDeformation1/self.maxTendonDeformation),
                    self.L*np.sqrt(2),
                    1001
                )
            ]
        )
        self.spring2_x = np.concatenate(
            [
                np.linspace(
                    0,
                    self.L*np.sqrt(2)/6*(2-self.tendonDeformation2/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.L*np.sqrt(2)/6*(2-self.tendonDeformation2/self.maxTendonDeformation),
                    self.L*np.sqrt(2)/6*(4+self.tendonDeformation2/self.maxTendonDeformation),
                    1001
                ),
                np.linspace(
                    self.L*np.sqrt(2)/6*(4+self.tendonDeformation2/self.maxTendonDeformation),
                    self.L*np.sqrt(2),
                    1001
                )
            ]
        )
        self.spring1Arrays = (
            np.array([
                [-self.L + self.rm*np.cos(self.phi1)],
                [-self.L + self.rm*np.sin(self.phi1)]
            ])
            + np.array([
                [np.cos(self.phi1-90*np.pi/180),-np.sin(self.phi1-90*np.pi/180)],
                [np.sin(self.phi1-90*np.pi/180),np.cos(self.phi1-90*np.pi/180)]
            ])@np.array([self.spring1_x[:,0],self.spring_y])
        )
        self.spring2Arrays = (
            np.array([
                [self.L + self.rm*np.cos(self.phi2)],
                [-self.L + self.rm*np.sin(self.phi2)]
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
            [0,self.L*np.sin(X[0,0])],
            [0,-self.L*np.cos(X[0,0])],
            Color='0.50',
            lw = 10,
            solid_capstyle='round',
            path_effects=[pe.Stroke(linewidth=12, foreground='k'), pe.Normal()]
        )

        # Pendulum Joint

        self.pendulumJoint = plt.Circle((0,0),self.rj,Color='#4682b4')
        self.ax0.add_patch(self.pendulumJoint)

        self.pendulumJointRivet, = self.ax0.plot(
            [0],
            [0],
            c='k',
            marker='o',
            lw=2
        )

        # Motors
        self.motor1Joint = plt.Circle((-self.L,-self.L),self.rm,Color='#4682b4')
        self.ax0.add_patch(self.motor1Joint)

        self.motor1Rivet, = self.ax0.plot(
            [-self.L],
            [-self.L],
            c='k',
            marker='.',
            lw=1
        )

        self.motor2Joint = plt.Circle((self.L,-self.L),self.rm,Color='#4682b4')
        self.ax0.add_patch(self.motor2Joint)

        self.motor2Rivet, = self.ax0.plot(
            [self.L],
            [-self.L],
            c='k',
            marker='.',
            lw=1
        )

        self.max_tau = self.U.max()
        if self.max_tau==0: self.max_tau=1

        self.k = 0.075*self.L
        self.inputIndicator1, = self.ax0.plot(
            (
                -self.L
                + 2.5*self.rm*np.cos(
                    np.linspace(
                        135*np.pi/180,
                        135*np.pi/180 + np.pi*self.U[0,0]/self.max_tau,
                        20
                    )
                )
            ),
            (
                -self.L
                + 2.5*self.rm*np.sin(
                    np.linspace(
                        135*np.pi/180,
                        135*np.pi/180 + np.pi*self.U[0,0]/self.max_tau,
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
                -self.L
                + 2.5*self.rm*np.cos(135*np.pi/180 + np.pi*self.U[0,0]/self.max_tau)
                + self.k*(self.U[0,0]/self.max_tau)**(1/2)*np.array([
                    np.cos(45*np.pi/180 + np.pi*self.U[0,0]/self.max_tau - 1.33*30*np.pi/180),
                    0,
                    np.cos(45*np.pi/180 + np.pi*self.U[0,0]/self.max_tau + 30*np.pi/180)
                ])
            ),
            (
                -self.L
                + 2.5*self.rm*np.sin(135*np.pi/180 + np.pi*self.U[0,0]/self.max_tau)
                + self.k*(self.U[0,0]/self.max_tau)**(1/2)*np.array([
                    np.sin(45*np.pi/180 + np.pi*self.U[0,0]/self.max_tau - 1.33*30*np.pi/180),
                    0,
                    np.sin(45*np.pi/180 + np.pi*self.U[0,0]/self.max_tau + 30*np.pi/180)
                ])
            ),
            Color='r',
            lw = 2,
            solid_capstyle='round'
        )

        self.inputIndicator2, = self.ax0.plot(
            (
                self.L
                + 2.5*self.rm*np.cos(
                    np.linspace(
                        45*np.pi/180-np.pi*self.U[1,0]/self.max_tau,
                        45*np.pi/180,
                        20
                    )
                )
            ),
            (
                -self.L
                + 2.5*self.rm*np.sin(
                    np.linspace(
                        45*np.pi/180-np.pi*self.U[1,0]/self.max_tau,
                        45*np.pi/180,
                        20
                    )
                )
            ),
            Color='g',
            lw = 2,
            solid_capstyle = 'round'
            )
        self.inputIndicator2Arrow, = self.ax0.plot(
            (
                self.L
                + 2.5*self.rm*np.cos(45*np.pi/180 - np.pi*self.U[1,0]/self.max_tau)
                + self.k*(self.U[1,0]/self.max_tau)**(1/2)*np.array([
                    np.cos(135*np.pi/180 - np.pi*self.U[1,0]/self.max_tau - 30*np.pi/180),
                    0,
                    np.cos(135*np.pi/180 - np.pi*self.U[1,0]/self.max_tau + 1.33*30*np.pi/180)
                ])
            ),
            (
                -self.L
                + 2.5*self.rm*np.sin(45*np.pi/180 - np.pi*self.U[1,0]/self.max_tau)
                + self.k*(self.U[1,0]/self.max_tau)**(1/2)*np.array([
                    np.sin(135*np.pi/180 - np.pi*self.U[1,0]/self.max_tau - 30*np.pi/180),
                    0,
                    np.sin(135*np.pi/180 - np.pi*self.U[1,0]/self.max_tau + 1.33*30*np.pi/180)
                ])
            ),
            Color='g',
            lw = 2,
            solid_capstyle='round'
        )

        self.ax0.get_xaxis().set_ticks([])
        self.ax0.get_yaxis().set_ticks([])
        self.ax0.set_frame_on(True)
        self.ax0.set_xlim([-1.1*self.L-2.5*self.rm,1.1*self.L+2.5*self.rm])
        self.ax0.set_ylim([-1.1*self.L-2.5*self.rm,1.1*self.L])
        self.ax0.set_aspect('equal')

        self.timeStamp = self.ax0.text(
            0,
            -self.L,
            "Time: "+str(self.Time[0])+" s",
            color='0.50',
            fontsize=16,
            horizontalalignment='center'
        )

        #Input

        self.input1, = self.ax1.plot([0],[self.U[0,0]],color = 'r')
        self.input2, = self.ax1.plot([0],[self.U[1,0]],color = 'g')
        self.ax1.set_xlim(0,self.Time[-1])
        self.ax1.set_xticks(list(np.linspace(0,self.Time[-1],5)))
        self.ax1.set_xticklabels([str(0),'','','',str(self.Time[-1])+"s"])
        if max(abs(self.U[0,:] - self.U[0,0]))<1e-7 and max(abs(self.U[1,:] - self.U[1,0]))<1e-7:
            self.ax1.set_ylim([min(self.U[:,0]) - 5,max(self.U[:,0]) + 5])
        else:
            self.RangeU = self.U.max()-self.U.min()
            self.ax1.set_ylim([self.U.min()-0.1*self.RangeU,self.U.max()+0.1*self.RangeU])

        self.ax1.spines['right'].set_visible(False)
        self.ax1.spines['top'].set_visible(False)
        self.ax1.set_title("Motor Torques (Nm)",fontsize=16,fontweight = 4,color = 'k',y = 1.00)

        #pendulum angle
        self.Y1d = self.Y[0,:]*180/np.pi
        self.angle, = self.ax2.plot([0],[self.Y1d[0]],color = 'C0')
        self.desiredAngle, = self.ax2.plot(Time,self.x1d*180/np.pi,c='k',linestyle='--',lw=1)
        self.ax2.set_xlim(0,self.Time[-1])
        self.ax2.set_xticks(list(np.linspace(0,self.Time[-1],5)))
        self.ax2.set_xticklabels([str(0),'','','',str(self.Time[-1])+"s"])
        if max(abs(self.Y1d-self.Y1d[0]))<1e-7:
            self.ax2.set_ylim([self.Y1d[0]-2,self.Y1d[0]+2])
        else:
            self.RangeY1d= max(self.Y1d)-min(self.Y1d)
            self.ax2.set_ylim([min(self.Y1d)-0.1*self.RangeY1d,max(self.Y1d)+0.1*self.RangeY1d])
            # y1_min = np.floor((self.Y[0,:].min()*180/np.pi)/22.5)*22.5
            # y1_min = min([y1_min,np.floor((self.x1d.min()*180/np.pi)/22.5)*22.5])
            # y1_max = np.ceil((self.Y[0,:].max()*180/np.pi)/22.5)*22.5
            # y1_max = max([y1_max,np.ceil((self.x1d.max()*180/np.pi)/22.5)*22.5])
            y1_min = 0
            y1_max = 360
            yticks = np.arange(y1_min,y1_max+22.5,22.5)
            yticklabels = []
            for el in yticks:
            	if el%45==0:
            		yticklabels.append(str(int(el)) + r"$^\circ$")
            	else:
            		yticklabels.append("")
            self.ax2.set_yticks(yticks)
            self.ax2.set_yticklabels(yticklabels)
        self.ax2.spines['right'].set_visible(False)
        self.ax2.spines['top'].set_visible(False)
        self.ax2.set_title("Angle (deg)",fontsize=16,fontweight = 4,color = 'C0',y = 1.00)

        # pendulum stiffness

        self.stiffness, = self.ax4.plot([0],[self.Y[1,0]],color='C1')
        self.desiredStiffness, = self.ax4.plot(Time,self.Sd,c='k',linestyle='--',lw=1)
        self.ax4.set_xlim(0,self.Time[-1])
        self.ax4.set_xticks(list(np.linspace(0,self.Time[-1],5)))
        self.ax4.set_xticklabels([str(0),'','','',str(self.Time[-1])+"s"])
        if max(abs(self.Y[1,:]-self.Y[1,0]))<1e-7:
            self.ax4.set_ylim([self.Y[1,0]-2,self.Y[1,0]+2])
        else:
            self.RangeY2= max(self.Y[1,:])-min(self.Y[1,:])
            self.ax4.set_ylim([min(self.Y[1,:])-0.1*self.RangeY2,max(self.Y[1,:])+0.1*self.RangeY2])
        self.ax4.spines['right'].set_visible(False)
        self.ax4.spines['top'].set_visible(False)
        self.ax4.set_title("Stiffness (Nm/rad)",fontsize=16,fontweight = 4,color = 'C1',y = 1.00)

    def animate(self,i):
        self.spring1Arrays = (
            np.array([
                [-self.L + self.rm*np.cos(self.phi1)],
                [-self.L + self.rm*np.sin(self.phi1)]
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
                [self.L + self.rm*np.cos(self.phi2)],
                [-self.L + self.rm*np.sin(self.phi2)]
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
            -self.L
            + 2.5*self.rm*np.cos(
                np.linspace(
                    135*np.pi/180,
                    135*np.pi/180 + np.pi*self.U[0,i]/self.max_tau,
                    20
                )
            )
        )
        self.inputIndicator1.set_ydata(
            -self.L
            + 2.5*self.rm*np.sin(
                np.linspace(
                    135*np.pi/180,
                    135*np.pi/180 + np.pi*self.U[0,i]/self.max_tau,
                    20
                )
            )
        )

        self.inputIndicator1Arrow.set_xdata(
            -self.L
            + 2.5*self.rm*np.cos(135*np.pi/180 + np.pi*self.U[0,i]/self.max_tau)
            + self.k*(self.U[0,i]/self.max_tau)**(1/2)*np.array([
                np.cos(45*np.pi/180 + np.pi*self.U[0,i]/self.max_tau - 1.33*30*np.pi/180),
                0,
                np.cos(45*np.pi/180 + np.pi*self.U[0,i]/self.max_tau + 30*np.pi/180)
            ])
        )
        self.inputIndicator1Arrow.set_ydata(
            -self.L
            + 2.5*self.rm*np.sin(135*np.pi/180 + np.pi*self.U[0,i]/self.max_tau)
            + self.k*(self.U[0,i]/self.max_tau)**(1/2)*np.array([
                np.sin(45*np.pi/180 + np.pi*self.U[0,i]/self.max_tau - 1.33*30*np.pi/180),
                0,
                np.sin(45*np.pi/180 + np.pi*self.U[0,i]/self.max_tau + 30*np.pi/180)
            ])
        )

        self.inputIndicator2.set_xdata(
            self.L
            + 2.5*self.rm*np.cos(
                np.linspace(
                    45*np.pi/180-np.pi*self.U[1,i]/self.max_tau,
                    45*np.pi/180,
                    20
                )
            )
        )
        self.inputIndicator2.set_ydata(
            -self.L
            + 2.5*self.rm*np.sin(
                np.linspace(
                    45*np.pi/180-np.pi*self.U[1,i]/self.max_tau,
                    45*np.pi/180,
                    20
                )
            )
        )

        self.inputIndicator2Arrow.set_xdata(
            self.L
            + 2.5*self.rm*np.cos(45*np.pi/180 - np.pi*self.U[1,i]/self.max_tau)
            + self.k*(self.U[1,i]/self.max_tau)**(1/2)*np.array([
                np.cos(135*np.pi/180 - np.pi*self.U[1,i]/self.max_tau - 30*np.pi/180),
                0,
                np.cos(135*np.pi/180 - np.pi*self.U[1,i]/self.max_tau + 1.33*30*np.pi/180)
            ])
        )
        self.inputIndicator2Arrow.set_ydata(
            -self.L
            + 2.5*self.rm*np.sin(45*np.pi/180 - np.pi*self.U[1,i]/self.max_tau)
            + self.k*(self.U[1,i]/self.max_tau)**(1/2)*np.array([
                np.sin(135*np.pi/180 - np.pi*self.U[1,i]/self.max_tau - 30*np.pi/180),
                0,
                np.sin(135*np.pi/180 - np.pi*self.U[1,i]/self.max_tau + 1.33*30*np.pi/180)
            ])
        )


        self.timeStamp.set_text("Time: "+"{:.2f}".format(self.Time[i])+" s",)

        self.input1.set_xdata(self.Time[:i])
        self.input1.set_ydata(self.U[0,:i])

        self.input2.set_xdata(self.Time[:i])
        self.input2.set_ydata(self.U[1,:i])

        self.angle.set_xdata(self.Time[:i])
        self.angle.set_ydata(self.Y1d[:i])

        self.stiffness.set_xdata(self.Time[:i])
        self.stiffness.set_ydata(self.Y[1,:i])

        return self.pendulum,self.spring1,self.spring2,self.inputIndicator1,self.inputIndicator1Arrow,self.inputIndicator2,self.inputIndicator2Arrow,self.input1,self.input2,self.angle,self.stiffness,self.timeStamp,

    def start(self,interval):
        self.anim = animation.FuncAnimation(self.fig, self.animate,
            frames=len(self.Time), interval=interval, blit=False)

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
