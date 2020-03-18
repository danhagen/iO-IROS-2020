from plant import plant_pendulum_1DOF2DOF
from save_params import *
from plantParams import *
from animate_babbling import *
from danpy.useful_functions import is_number, save_figures
import scipy.io as sio
import argparse
import textwrap
from scipy import signal
import seaborn as sns

babblingParams = {
    "Seed" : None,
    "Filter Length" : 50,
    "Pass Probability" : plantParams["dt"]/4,
    "Input Bounds" : [0,10],
    "Low Cutoff Frequency" : 1,
    "High Cutoff Frequency" : 10,
    "Buttersworth Filter Order" : 9,
    "Babbling Type" : 'continuous', # ['continuous','step']
    "Force Cocontraction" : True,
    "Cocontraction Standard Deviation" : 2
}

class motor_babbling_1DOF2DOA:
    def __init__(self,plant,babblingParams):
        self.totalParams = plant.params
        self.totalParams.update(babblingParams)
        self.plant = plant

        self.seed = babblingParams.get("Seed",None)
        if self.seed is not None:
            is_number(self.seed,"Seed",default="None")
            np.random.seed(self.seed)

        self.filterLength = babblingParams.get("Filter Length",100)
        is_number(self.filterLength,"Filter Length",default=100)

        self.passProbability = babblingParams.get("Pass Probability",0.001)
        is_number(self.passProbability,"Pass Probability",default=0.001)

        self.inputBounds = babblingParams.get("Input Bounds",[0,100])
        assert (type(self.inputBounds)==list and len(self.inputBounds)==2), \
            "Input Bounds must be a list of length 2. Default is [0,100]."
        is_number(self.inputBounds[0],"Input Minimum")
        is_number(self.inputBounds[1],"Input Maximum")
        assert self.inputBounds[1]>self.inputBounds[0], "Input Bounds must be in ascending order. Default is [0,100]."
        self.inputRange = self.inputBounds[1]-self.inputBounds[0]
        self.inputMinimum = self.inputBounds[0]
        self.inputMaximum = self.inputBounds[1]

        self.lowCutoffFrequency = babblingParams.get("Low Cutoff Frequency",1)
        is_number(self.lowCutoffFrequency,"Low Cutoff Frequency",default=1)

        self.highCutoffFrequency = babblingParams.get("High Cutoff Frequency",10)
        is_number(self.highCutoffFrequency,"High Cutoff Frequency",default=10)
        assert self.lowCutoffFrequency<self.highCutoffFrequency, "The low cutoff frequency for the white noise must be below the high cutoff frequency"

        self.filterOrder = babblingParams.get("Buttersworth Filter Order",5)
        is_number(self.filterOrder,"Buttersworth Filter Order",default=5)

        self.babblingType = babblingParams.get("Babbling Type",'continuous')
        assert self.babblingType in ['continuous','step'], "babblingType must be either 'continuous' (default) or 'step'."
        if self.babblingType == 'continuous':
            '''
            filterLength should be equal to the length of each random number so that the moving average peaks at the random number when exactly in the middle of the phase.

            Each phase should be "one over half of the maximum frequency" seconds long to ensure that the maximum frequency would occur with two phases.
            '''
            self.filterLength = int(1/self.highCutoffFrequency/2/self.plant.dt)
            assert type(babblingParams.get("Force Cocontraction",False))==bool, "'Force Cocontraction' should be either True or False (default)."
            if babblingParams.get("Force Cocontraction",False)==True:
                self.cocontraction = True
                self.cocontractionSTD = babblingParams.get("Cocontraction Standard Deviation",1)
                is_number(self.cocontractionSTD,"Cocontraction Standard Deviation",default=1)
            else:
                self.cocontraction = False

    def band_limited_noise(self):
        numberOfSamples = len(self.plant.time)-1
        samplingFrequency = 1/self.plant.dt
        frequencies = np.abs(np.fft.fftfreq(numberOfSamples, 1/samplingFrequency))
        f = np.zeros(numberOfSamples)
        index = np.where(
            np.logical_and(
                frequencies>=self.lowCutoffFrequency, frequencies<=self.highCutoffFrequency
            )
        )[0]

        f[index] = 1
        f = np.array(f, dtype='complex')

        Np = (len(f) - 1) // 2
        phases = np.random.rand(Np) * 2 * np.pi
        phases = np.cos(phases) + 1j * np.sin(phases)
        f[1:Np+1] *= phases
        f[-1:-1-Np:-1] = np.conj(f[1:Np+1])
        result = np.fft.ifft(f).real
        # result = result/max([result.max(),-result.min()]))
        return(result)
    def plot_signals_power_spectrum_and_amplitude_distribution(self):
        assert hasattr(self,"babblingSignals"), "run generate_step_babbling_input or generate_continuous_babbling_input before plotting the power spectrum."

        fig1, (ax1a,ax1b) = plt.subplots(2,1,sharex=True,figsize=(5,7))
        ax1a.plot([self.plant.time[0],self.plant.time[-1]],[self.inputMaximum]*2,'k--',label='_nolegend_')
        ax1a.plot([self.plant.time[0],self.plant.time[-1]],[self.inputMinimum]*2,'k--',label='_nolegend_')
        ax1a.set_ylabel('Babbling Input 1 (Nm)')
        ax1a.set_ylim([
            self.inputMinimum-0.1*self.inputRange,
            self.inputMaximum+0.1*self.inputRange
        ])
        ax1a.set_xlim([self.plant.time[0],self.plant.time[-1]])
        plt.setp(ax1a.get_xticklabels(), visible=False)
        ax1a.spines["right"].set_visible(False)
        ax1a.spines["top"].set_visible(False)

        ax1b.plot([self.plant.time[0],self.plant.time[-1]],[self.inputMaximum]*2,'k--',label='_nolegend_')
        ax1b.plot([self.plant.time[0],self.plant.time[-1]],[self.inputMinimum]*2,'k--',label='_nolegend_')
        ax1b.set_ylabel('Babbling Input 2 (Nm)')
        ax1b.set_ylim([
            self.inputMinimum-0.1*self.inputRange,
            self.inputMaximum+0.1*self.inputRange
        ])
        ax1b.set_xlim([self.plant.time[0],self.plant.time[-1]])
        ax1b.set_xlabel('Time (s)')
        ax1b.spines["right"].set_visible(False)
        ax1b.spines["top"].set_visible(False)

        fig2 = plt.figure(figsize=(5, 4))
        ax2=plt.gca()
        plt.title('PSD: power spectral density')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'Power ($(Nm)^2/Hz$)')
        plt.tight_layout()

        ax1a.plot(
            self.plant.time[:-1],
            self.babblingSignals[:,0],
            "r"
        )
        ax1b.plot(
            self.plant.time[:-1],
            self.babblingSignals[:,1],
            "r"
        )
        inputLineStyle=[None,'--']
        for i in range(self.babblingSignals.shape[1]):
            freqs,psd = signal.welch(
                self.babblingSignals[:,i],
                1/self.plant.dt
            )
            ax2.semilogx(freqs,psd,c='r',ls=inputLineStyle[i])

        ax2.legend(
            ["Signal " +str(i+1) for i in range(2)],
            loc="upper right"
        )

        fig3, (ax3a,ax3b) = plt.subplots(2,1,figsize=(5,7))
        ax3a.set_ylabel('Percentage')
        ax3a.set_xlabel('Babbling Input 1 (Nm)')
        ax3a.spines["right"].set_visible(False)
        ax3a.spines["top"].set_visible(False)

        ax3b.set_ylabel('Percentage')
        ax3b.set_xlabel('Babbling Input 2 (Nm)')
        ax3b.spines["right"].set_visible(False)
        ax3b.spines["top"].set_visible(False)
        sns.distplot(self.babblingSignals[:,0],hist=True,color='r',ax=ax3a)
        sns.distplot(self.babblingSignals[:,1],hist=True,color='r',ax=ax3b)

    def generate_continuous_babbling_input(self):
        """
        Returns a babbling signal for 2 channels that either steps to some level of torque inputs (inside the bounds) or is zero.
        """
        np.random.seed(self.seed)
        numberOfSamples = len(self.plant.time)-1
        self.babblingSignals = np.zeros((numberOfSamples,2))

        self.stepDuration = 1/self.highCutoffFrequency/2 # 50 ms
        indices = [
            int(el/self.plant.dt) for el in np.arange(0,self.plant.time[-1]+self.stepDuration,self.stepDuration)
        ]
        if self.cocontraction==False:
            for i in range(len(indices)-1):
                stepSize = int(len(self.plant.time[indices[i]:indices[i+1]]))
                self.babblingSignals[i*stepSize:(i+1)*stepSize,0] = [
                    np.random.uniform(self.inputMinimum,self.inputMaximum)
                ]*stepSize
                self.babblingSignals[i*stepSize:(i+1)*stepSize,1] = [
                    np.random.uniform(self.inputMinimum,self.inputMaximum)
                ]*stepSize
        else: #self.cocontraction==True
            for i in range(len(indices)-1):
                stepSize = int(len(self.plant.time[indices[i]:indices[i+1]]))
                u1_rand = np.random.uniform(
                    self.inputMinimum,
                    self.inputMaximum
                )
                u2_rand = np.random.normal(
                    u1_rand,
                    self.cocontractionSTD
                )
                if u2_rand<=self.inputMinimum:
                    u2_rand=self.inputMinimum
                elif u2_rand>=self.inputMaximum:
                    u2_rand=self.inputMaximum

                self.babblingSignals[i*stepSize:(i+1)*stepSize,0] = [
                    u1_rand
                ]*stepSize
                self.babblingSignals[i*stepSize:(i+1)*stepSize,1] = [
                    u2_rand
                ]*stepSize

        ### Filter the offset signals
        # nyq = 0.5 / self.plant.dt
        # low = self.lowCutoffFrequency/nyq
        # high = self.highCutoffFrequency/nyq
        # b, a = signal.butter(self.filterOrder, high, 'low')
        # self.babblingSignals = signal.filtfilt(b, a, self.babblingSignals.T).T
        # # b, a = signal.butter(self.filterOrder, [low, high], btype='band')
        # # self.babblingSignals = signal.filtfilt(
        # #     b,a,
        # #     self.babblingSignals.T
        # # ).T
        b = np.ones(self.filterLength,)/(self.filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with one second filter length
        a=1
        self.babblingSignals = signal.filtfilt(
            b,a,
            self.babblingSignals.T
        ).T

        ### Bound the signals
        for i in range(numberOfSamples):
            if self.babblingSignals[i,0]<=self.inputMinimum:
                self.babblingSignals[i,0]=self.inputMinimum
            elif self.babblingSignals[i,0]>=self.inputMaximum:
                self.babblingSignals[i,0]=self.inputMaximum

            if self.babblingSignals[i,1]<=self.inputMinimum:
                self.babblingSignals[i,1]=self.inputMinimum
            elif self.babblingSignals[i,1]>=self.inputMaximum:
                self.babblingSignals[i,1]=self.inputMaximum

    def generate_step_babbling_input(self):
        """
        Returns a babbling signal for 2 channels that either steps to some level of torque inputs (inside the bounds) or is zero.
        """
        np.random.seed(self.seed)
        numberOfSamples = len(self.plant.time)-1
        self.babblingSignals = np.zeros((numberOfSamples,2))
        self.babblingSignalOffsets = np.zeros((numberOfSamples,2))
        self.babblingNoise = np.concatenate(
            [
                self.band_limited_noise()[:,np.newaxis],
                self.band_limited_noise()[:,np.newaxis]
            ],
            axis=1
        )
        self.babblingNoise = (
            (self.babblingNoise-self.babblingNoise.min())
            / (self.babblingNoise.max() - self.babblingNoise.min())
        )
        delay = 3 # sec

        ### first transition after delay
        self.babblingSignalOffsets[int(delay/self.plant.dt),:] = (
            self.inputRange
            * np.random.uniform(0,1,(2,)) # random number inside
            + self.inputMinimum
        )

        ### Running remainder of simulation
        for i in range(int(delay/self.plant.dt)+1, numberOfSamples):
            if np.random.uniform() < self.passProbability: # change input offset
                self.babblingSignalOffsets[i,:] = (
                    self.inputRange
                    * np.random.uniform(0,1,(2,)) # random number inside
                    + self.inputMinimum
                )
            else: # stay at previous input
                self.babblingSignalOffsets[i,:] = self.babblingSignalOffsets[i-1,:]

        ### Find the DC offsets
        self.noiseAmplitude = np.concatenate(
            [
                abs(
                    self.babblingSignalOffsets.min(axis=1)
                    - self.inputBounds[0]
                )[:,np.newaxis],
                abs(
                    self.inputBounds[1]
                    - self.babblingSignalOffsets.max(axis=1)
                )[:,np.newaxis]
            ],
            axis=1
        ).min(axis=1)

        ### Filter the offset signals
        b = np.ones(self.filterLength,)/(self.filterLength) #Finite Impulse Response (FIR) Moving Average (MA) filter with one second filter length
        a=1
        self.babblingSignalOffsets = signal.filtfilt(
            b,a,
            self.babblingSignalOffsets.T
        ).T

        ### Combine the signals
        self.babblingSignals = (
            self.babblingSignalOffsets
            + (self.noiseAmplitude*(self.babblingNoise.T)).T
        )

        ### Bound the signals
        for i in range(numberOfSamples):
            if self.babblingSignals[i,0]<=self.inputMinimum:
                self.babblingSignals[i,0]=self.inputMinimum
            elif self.babblingSignals[i,0]>=self.inputMaximum:
                self.babblingSignals[i,0]=self.inputMaximum

            if self.babblingSignals[i,1]<=self.inputMinimum:
                self.babblingSignals[i,1]=self.inputMinimum
            elif self.babblingSignals[i,1]>=self.inputMaximum:
                self.babblingSignals[i,1]=self.inputMaximum

    def save_data(self,X,path=None):
        assert hasattr(self,"babblingSignals"), "No babbling signals have been generated, please run self.generate_babbling_signals() before running this function."

        fT1 = np.array(list(map(self.plant.tendon_1_FL_func,X.T)))
        fT2 = np.array(list(map(self.plant.tendon_2_FL_func,X.T)))

        outputData = {
            "Time" : self.plant.time,
            "u1" : self.babblingSignals.T[0,:],
            "du1" : np.gradient(self.babblingSignals.T[0,:],self.plant.dt),
            "u2" : self.babblingSignals.T[1,:],
            "du2" : np.gradient(self.babblingSignals.T[1,:],self.plant.dt),
            "x1" : X[0,:],
            "dx1" : X[1,:],
            "d2x1" : np.gradient(X[1,:],self.plant.dt),
            "x3" : X[2,:],
            "dx3" : X[3,:],
            "d2x3" : np.gradient(X[3,:],self.plant.dt),
            "x5" : X[4,:],
            "dx5" : X[5,:],
            "d2x5" : np.gradient(X[5,:],self.plant.dt),
            "fT1" : fT1,
            "dfT1" : np.gradient(fT1,self.plant.dt),
            "d2fT1" : np.gradient(np.gradient(fT1,self.plant.dt),self.plant.dt),
            "fT2" : fT2,
            "dfT2" : np.gradient(fT2,self.plant.dt),
            "d2fT2" : np.gradient(np.gradient(fT2,self.plant.dt),self.plant.dt)
        }
        if path is not None:
            assert type(path)==str, "path must be a str."
            sio.savemat(path+"outputData.mat",outputData)
        else:
            sio.savemat("outputData.mat",outputData)

    def run_babbling_trial(
            self,
            x1o,
            plot=False,
            saveFigures=False,
            saveAsPDF=False,
            returnData=False,
            saveData=False,
            saveParams=False
        ):

        is_number(
            x1o,"Initial Joint Angle",
            notes="Should be between 0 and 2 pi."
        )

        if saveFigures==True:
            assert plot==True, "No figures will be generated. Please select plot=True."

        ## Generate babbling input
        if self.babblingType=='continuous':
            self.generate_continuous_babbling_input()
        else: #self.babblingType=='step'
            self.generate_step_babbling_input()

        ## running the babbling data through the plant
        X_o = self.plant.return_X_o(x1o,self.babblingSignals[0,:])

        X,U,_ = self.plant.forward_simulation(
            X_o,
            U=self.babblingSignals.T,
            addTitle="Motor Babbling"
        ) # need the transpose for the shapes to align (DAH)

        ## plot (and save) figures
        output = {}
        if plot==True:
            self.plant.plot_states(X)

            self.plant.plot_joint_angle_power_spectrum_and_distribution(X)

            self.plant.plot_tendon_tension_deformation_curves(X)

            self.plot_signals_power_spectrum_and_amplitude_distribution()

            if saveFigures==True:
                trialPath = save_figures(
                    "babbling_trials/",
                    "propAmp",
                    self.totalParams,
                    returnPath=True,
                    saveAsPDF=saveAsPDF
                )

                if saveParams==True:
                    save_params_as_MAT(self.totalParams,path=trialPath)

                if saveData==True:
                    self.save_data(X,path=trialPath)

                if returnData==True:
                    output["time"] = self.plant.time
                    output["X"] = X
                    output["U"] = U
                    output["path"] = trialPath
                    return(output)
                else:
                    output["path"] = trialPath
                    return(output)
            else:
                if saveParams==True:
                    save_params_as_MAT(self.totalParams)

                if saveData==True:
                    self.save_data(X)

                if returnData==True:
                    output["time"] = self.plant.time
                    output["X"] = X
                    output["U"] = U
                    return(output)
        else:
            if saveParams==True:
                save_params_as_MAT(self.totalParams)

            if saveData==True:
                self.save_data(X)

            if returnData==True:
                output["time"] = self.plant.time
                output["X"] = X
                output["U"] = U
                return(output)

if __name__ == '__main__':

    ### Delete the parameters that are not used in Babbling Trials
    if "Stiffness Gains" in plantParams.keys():
        del plantParams["Stiffness Gains"]
    if "Position Gains" in plantParams.keys():
        del plantParams["Position Gains"]
    if "Boundary Friction Weight" in plantParams.keys():
        del plantParams["Boundary Friction Weight"]
    if "Boundary Friction Gain" in plantParams.keys():
        del plantParams["Boundary Friction Gain"]
    if "Quadratic Stiffness Coefficient 1" in plantParams.keys():
        del plantParams["Quadratic Stiffness Coefficient 1"]
    if "Quadratic Stiffness Coefficient 2" in plantParams.keys():
        del plantParams["Quadratic Stiffness Coefficient 2"]

    ### Additional Arguments?
    parser = argparse.ArgumentParser(
        prog = "<filename>",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        motor_babbling_1DOF2DOA.py

        -----------------------------------------------------------------------------

        Motor babbling algorithm for a 1 DOF, 2 DOA tendon-driven system with
        nonlinear tendon elasticity. Low frequency white noise is added to
        random step changes to the input levels (within some bounds).

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
        '--savefigsPDF',
        action="store_true",
        help='Option to save figures for babbling trial as a PDF. Default is false.'
    )
    parser.add_argument(
        '--savedata',
        action="store_true",
        help='Option to save data for babbling trial as a Matlab .MAT file. Default is false.'
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
    saveAsPDF = args.savefigsPDF
    saveParams=saveFigures
    saveData = args.savedata
    animate = args.animate

    ### Define plant and babbling trial
    plant = plant_pendulum_1DOF2DOF(plantParams)
    babblingTrial = motor_babbling_1DOF2DOA(plant,babblingParams)
    output = babblingTrial.run_babbling_trial(
        np.pi,
        plot=True,
        saveAsPDF=saveAsPDF,
        returnData=True,
        saveData=saveData,
        saveParams=saveParams
    )

    if animate==True:
        downsamplingFactor = 10
        time = output["time"]
        X = output["X"]
        U = output["U"]
        # Y = np.array([
        #     X[0,:],
        #     np.array(list(map(lambda X: plant.hs(X),X.T)))
        # ])
        ani = animate_pendulum_babbling(
            time,X,U,
            downsamplingFactor,
            **plantParams
        )
        ani.start(downsamplingFactor)

    plt.show()
