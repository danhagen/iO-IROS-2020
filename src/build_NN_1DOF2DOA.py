from motor_babbling_1DOF2DOA import *
from save_params import *
import os
import matlab.engine
import argparse
import textwrap
from danpy.sb import get_terminal_width
import shutil
import time
import pickle

### ANN parameters
ANNParams = {
    "Number of Nodes" : 15,
    "Number of Epochs" : 50,
    "Number of Trials" : 2,
}
colors = [
    "#2A3179", # all
    "#F4793B", # bio
    "#8DBDE6", # kinapprox
    "#A95AA1" # allmotor
]

def plot_babbling_duration_vs_minimum_performance(directory=None):
    ### get the testing trial directories
    if directory==None:
        directory = "training_trials/"
    else:
        assert os.path.isdir(directory), "Enter a valid directory."
    trialDirectories = [
        name for name in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, name))
        and name[:2]=='Co'
    ]
    babblingDurations = np.array([
        int(name[-4:-1]) for name in trialDirectories
    ])
    totalPerformanceData = {
        "all": [],
        "bio": [],
        "kinapprox": [],
        "allmotor": []
    }
    for n in range(len(trialDirectories)):
        with open(directory+trialDirectories[n]+'/combinedTrainingData.pkl', 'rb') as handle:
            tempOutputData = pickle.load(handle)
        for key in totalPerformanceData:
            totalPerformanceData[key].append(
                180*np.sqrt(tempOutputData[key]["avg_best_perf"])/np.pi
            )

    plt.figure()
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Babbling Duration (sec.)")
    ax.set_xticks(list(babblingDurations))
    ax.set_xticklabels([int(el) for el in babblingDurations])
    ax.set_ylabel("Best Performance (RMSE in deg.)")
    for i in range(len(totalPerformanceData.keys())):
        key = list(totalPerformanceData.keys())[i]
        ax.plot(babblingDurations,totalPerformanceData[key],c=colors[i])
    ax.legend(list(totalPerformanceData.keys()),loc='upper right')

class neural_network:
    def __init__(self,ANNParams,babblingParams,plantParams):
        self.plant = plant_pendulum_1DOF2DOF(plantParams)

        self.totalParams = plantParams
        self.totalParams.update(babblingParams)
        self.totalParams.update(ANNParams)

        self.numberOfTrials = ANNParams.get("Number of Trials",1)
        is_number(self.numberOfTrials,"Number of Trials",
            default=1,note="Must be an int.")

        self.numberOfEpochs = ANNParams.get("Number of Epochs",50)
        is_number(self.numberOfEpochs,"Number of Epochs",
            default=50,note="Must be an int.")

        self.numberOfLayers = ANNParams.get("Number of Nodes",15)
        is_number(self.numberOfLayers,"Number of Nodes",
            default=15,note="Must be an int.")

        self.groups = ["all","bio","allmotor","kinapprox"]

    def plot_performance(self,returnFig=True):
        assert hasattr(self,"ANNOutput"), "neural_network has no attribute 'ANNOutput', please run neural_network.run_trial() before plotting."

        fig = plt.figure(figsize=(8,6))
        plt.yscale("log")
        ax = plt.gca()
        ax.set_xlabel("Epoch #")
        ax.set_ylabel("Log Performance (RMSE in deg.)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks(list(np.linspace(0,self.numberOfEpochs,6)))
        ax.set_xticklabels([int(el) for el in ax.get_xticks()])

        for i in range(len(self.groups)):
            ax.plot(
                np.array(self.ANNOutput[self.groups[i]]["tr"]["epoch"]).T,
                180*np.sqrt(np.array(self.ANNOutput[self.groups[i]]["tr"]["perf"])).T/np.pi,
                c=colors[i]
            )

        ax.legend(self.groups,loc='upper right')

        if returnFig==True:
            return(fig)
        else:
            plt.show()

    def plot_average_performance(self,totalPerformance,returnFig=True):

        epochArray = np.arange(0,self.numberOfEpochs+1,1)

        fig = plt.figure(figsize=(8,6))
        plt.yscale("log")
        plt.title("Average Performance vs. Epoch")
        ax = plt.gca()
        ax.set_xlabel("Epoch #")
        ax.set_ylabel("Performance (RMSE in deg.)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xticks(list(np.linspace(0,self.numberOfEpochs,6)))
        ax.set_xticklabels([int(el) for el in ax.get_xticks()])

        for i in range(len(self.groups)):
            key = self.groups[i]
            ax.plot(
                epochArray,
                180*np.sqrt(totalPerformance[key]["avg_perf"])/np.pi,
                c=colors[i],
                lw=2
            )
        ax.legend(self.groups,loc='upper right')

        if returnFig==True:
            return(fig)
        else:
            plt.show()

    def plot_performance_distributions(self,totalPerformance,returnFig=True):

        fig = plt.figure(figsize=(8,12))
        ax1 = plt.subplot(311)
        ax1.set_xlabel("RMSE (in deg.)")
        ax1.set_ylabel("Percentage of Trials")
        ax2 = plt.subplot(323)
        ax3 = plt.subplot(324)
        ax4 = plt.subplot(325)
        ax5 = plt.subplot(326)
        axs = [ax1,ax2,ax3,ax4,ax5]
        for ax in axs:
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

        plt.suptitle("Best Performance Distributions")

        for i in range(len(self.groups)):
            key = self.groups[i]
            minPerformance = min([
                np.array(totalPerformance[key]["best_perf"]).min()
                for key in totalPerformance
            ])
            maxPerformance = max([
                np.array(totalPerformance[key]["best_perf"]).max()
                for key in totalPerformance
            ])
            bins = np.linspace(minPerformance,maxPerformance,25)
            data = totalPerformance[key]["best_perf"]
            _,_,_ = ax1.hist(
                data,
                bins=bins,
                color=colors[i],
                weights=np.ones(len(data)) / len(data),
                alpha=0.5
            )
            _,_,_ = axs[i+1].hist(
                data,
                bins=bins,
                color=colors[i],
                weights=np.ones(len(data)) / len(data),
                alpha=0.5
            )
            axs[i+1].set_yticklabels(
                ["{:.1f}%".format(100*el) for el in axs[i+1].get_yticks()]
            )
            axs[i+1].set_title(
                self.groups[i],
                fontsize=14,
                color=colors[i],
                y=0.65,
                x=0.65
            )

        ax1.legend(self.groups,loc="upper right")
        ax1.set_yticklabels(
            ["{:.1f}%".format(100*el) for el in ax1.get_yticks()]
        )

        if returnFig==True:
            return(fig)
        else:
            plt.show()

    def run_trial(self):
        ### Generate babbling data
        babblingTrial = motor_babbling_1DOF2DOA(
            self.plant,
            self.totalParams
        )
        babblingOutput = babblingTrial.run_babbling_trial(
            np.pi,
            plot=True,
            saveFigures=False,
            saveAsPDF=False,
            returnData=True,
            saveData=False,
            saveParams=False
        )

        Time = babblingOutput["time"]
        X = babblingOutput["X"]
        U = babblingOutput["U"]

        ### save figures and parameters

        self.trialPath = save_figures(
            "training_trials/",
            "propAmp",
            self.totalParams,
            returnPath=True,
            saveAsPDF=False
        )
        babblingTrial.save_data(X,path=self.trialPath+"babblingTrial_")
        save_params_as_MAT(self.totalParams,path=self.trialPath)

        ### MATLAB ANN func
        eng = matlab.engine.start_matlab()
        self.ANNOutput = eng.ANNmapping_to_python(
            self.trialPath,
            self.numberOfLayers,
            self.numberOfEpochs
        )

        fig1 = self.plot_performance(returnFig=True)

        save_figures(
            self.trialPath[:-18],
            "propAmp",
            self.totalParams,
            figs=[fig1],
            subFolderName=self.trialPath[-18:],
            returnPath=False,
            saveAsPDF=True
        )
        with open(self.trialPath+"trainingData.pkl", 'wb') as handle:
            pickle.dump(self.ANNOutput, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def run_experimental_trial(self,**kwargs):
        returnBabblingData = kwargs.get('returnBabblingData',False)
        assert type(returnBabblingData)==bool, "returnBabblingData must be either True or False (default)."
        fullEpochs = False
        count = 0
        while fullEpochs==False:
            ### Generate babbling data
            babblingTrial = motor_babbling_1DOF2DOA(
                self.plant,
                self.totalParams
            )
            babblingOutput = babblingTrial.run_babbling_trial(
                np.pi,
                plot=True,
                saveFigures=False,
                saveAsPDF=False,
                returnData=True,
                saveData=False,
                saveParams=False
            )

            Time = babblingOutput["time"]
            X = babblingOutput["X"]
            U = babblingOutput["U"]

            ### save figures and parameters

            self.trialPath = save_figures(
                "experimental_trials/",
                "propAmp",
                self.totalParams,
                returnPath=True,
                saveAsPDF=False
            )
            babblingTrial.save_data(X,path=self.trialPath+"babblingTrial_")
            save_params_as_MAT(self.totalParams,path=self.trialPath)

            ### MATLAB ANN func
            eng = matlab.engine.start_matlab()
            ANNOutput = eng.ANNmapping_with_testing_to_python(
                self.trialPath,
                self.numberOfLayers,
                self.numberOfEpochs
            )

            self.ANNOutput,experimentalData = \
                ANNOutput['babbling'],ANNOutput['experiment'] # in rads.

            fullEpochs = np.all(
                [
                    (
                        np.array(
                            self.ANNOutput[key]["tr"]["perf"]
                        ).shape[1]
                        ==
                        (self.numberOfEpochs+1)
                    )
                    for key in self.ANNOutput
                ]
            )
            if fullEpochs==False:
                count +=1
                print("Early Termination, Trying again... " + str(count))
                shutil.rmtree(self.trialPath)
                plt.close('all')

        fig1 = self.plot_performance(returnFig=True)

        save_figures(
            self.trialPath[:-18],
            "training_perf",
            self.totalParams,
            figs=[fig1],
            subFolderName=self.trialPath[-18:],
            returnPath=False,
            saveAsPDF=False
        )
        plt.close('all')

        with open(self.trialPath+"trainingData.pkl", 'wb') as handle:
            pickle.dump(self.ANNOutput, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if returnBabblingData==True:
            return(experimentalData,babblingOutput)
        else:
            return(experimentalData)

    def run_multiple_trials(self):
        assert self.numberOfTrials>1, "Can't run multiple trials when neural_network.numberOfTrials is 1."

        totalPerformance = {}

        for i in range(self.numberOfTrials):
            print("Running Trial " +str(i+1)+"/"+str(self.numberOfTrials)+":")
            fullEpochs = False
            count = 0
            while fullEpochs==False:
                self.run_trial()
                fullEpochs = np.all(
                    [
                        (
                            np.array(
                                self.ANNOutput[key]["tr"]["perf"]
                            ).shape[1]
                            ==
                            (self.numberOfEpochs+1)
                        )
                        for key in self.ANNOutput
                    ]
                )
                if fullEpochs==False:
                    count +=1
                    print("Early Termination, Trying again... " + str(count))

            if i==0:
                for key in self.groups:
                    totalPerformance[key]={}
                    totalPerformance[key]["perf"] = \
                        np.array(self.ANNOutput[key]["tr"]["perf"]) # MSE in rads
                    totalPerformance[key]["best_perf"] = [
                        self.ANNOutput[key]["tr"]["best_perf"]
                    ] # MSE in rads
            else:
                for key in self.groups:
                    totalPerformance[key]["perf"] = np.concatenate(
                        [
                            totalPerformance[key]["perf"],
                            np.array(self.ANNOutput[key]["tr"]["perf"])
                        ],
                        axis=0
                    ) # MSE in rads
                    totalPerformance[key]["best_perf"].append(
                        self.ANNOutput[key]["tr"]["best_perf"]
                    ) # MSE in rads

            plt.close('all')

        for key in self.groups:
            totalPerformance[key]["avg_perf"] = \
                totalPerformance[key]["perf"].mean(axis=0) # MSE in rads
            totalPerformance[key]["std_perf"] = \
                totalPerformance[key]["perf"].std(axis=0) # MSE in rads
            totalPerformance[key]["max_perf"] = \
                totalPerformance[key]["perf"].max(axis=0) # MSE in rads
            totalPerformance[key]["min_perf"] = \
                totalPerformance[key]["perf"].min(axis=0) # MSE in rads
            totalPerformance[key]['avg_best_perf'] = \
                np.average(totalPerformance[key]["best_perf"]) # MSE in rads

        return(totalPerformance)

if __name__=="__main__":

    ### Delete the parameters that are not used in Babbling Trials
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

        build_NN_1DOF2DOA.py

        -----------------------------------------------------------------------------

        Build ANN for 1 DOF, 2 DOA tendon-driven system with nonlinear tendon
        elasticity in order to predict joint angle from different "sensory"
        states (like tendon tension or motor angle).

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
    parser.add_argument(
        '-epochs',
        type=int,
        help='Number of epochs for each network to train. Default is given by ANNParams.',
        default=ANNParams["Number of Epochs"]
    )
    parser.add_argument(
        '-nodes',
        type=int,
        help='Number of Nodes for each network to train (single hidden layer). Default is given by ANNParams.',
        default=ANNParams["Number of Nodes"]
    )
    parser.add_argument(
        '-trials',
        type=int,
        help='Number of trials to babble. Default is given by ANNParams.',
        default=ANNParams["Number of Trials"]
    )
    args = parser.parse_args()
    plantParams["dt"] = args.dt
    plantParams["Simulation Duration"] = args.dur
    ANNParams["Number of Epochs"] = args.epochs
    ANNParams["Number of Nodes"] = args.nodes
    ANNParams["Number of Trials"] = args.trials

    ### Generate Neural Network Class
    ANN = neural_network(ANNParams,babblingParams,plantParams)

    ### Run babbling trial
    if ANN.numberOfTrials==1:
        ANN.run_trial()
    else:
        totalPerformance = ANN.run_multiple_trials()
        fig1 = ANN.plot_performance_distributions(totalPerformance)
        fig2 = ANN.plot_average_performance(totalPerformance)

        folderName = (
            'Consolidated_Trials_'
            + '{:03d}'.format(int(ANN.plant.simulationDuration))
            + 's/'
        )
        path = save_figures(
            ANN.trialPath[:-18],
            "propAmp",
            ANN.totalParams,
            figs = [fig1,fig2],
            subFolderName = folderName,
            saveAsPDF=True,
            returnPath=True
        )
        plt.close('all')

        with open(ANN.trialPath[:-18]+folderName+"combinedTrainingData.pkl", 'wb') as handle:
            pickle.dump(totalPerformance, handle, protocol=pickle.HIGHEST_PROTOCOL)
