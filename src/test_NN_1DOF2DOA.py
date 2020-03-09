from motor_babbling_1DOF2DOA import *
from build_NN_1DOF2DOA import *
from save_params import *
import os
import matlab.engine
import argparse
import textwrap
from danpy.sb import get_terminal_width
from os import path,listdir
from matplotlib.patches import Wedge
import scipy.io as sio
import pickle
from PIL import Image
import time
import shutil

groups = ["all","bio","kinapprox","allmotor"]
colors = [
    "#2A3179", # all
    "#F4793B", # bio
    "#8DBDE6", # kinapprox
    "#A95AA1" # allmotor
]

def plot_babbling_duration_vs_average_performance(metric,directory=None):
    labels = [
        "(Sinusoidal Angle / Sinusoidal Stiffness)",
        "(Step Angle / Sinusoidal Stiffness)",
        "(Sinusoidal Angle / Step Stiffness)",
        "(Step Angle / Step Stiffness)"
    ]

    ### get the testing trial directories
    # assert type(metrics)==list, "metrics must be a list of strings."
    assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'

    if directory==None:
        directory = "experimental_trials/"
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
    totalPerformanceData = {}
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,10))
    axs = [ax1,ax2,ax3,ax4]
    for n in range(len(trialDirectories)):
        with open(directory+trialDirectories[n]+'/consolidatedOutputData.pkl', 'rb') as handle:
            tempOutputData = pickle.load(handle)
        if n==0:
            for key in tempOutputData.keys():
                totalPerformanceData[key] = {}
                for group in groups:
                    totalPerformanceData[key][group] = {}
                    totalPerformanceData[key][group]['values'] = []
                    # totalPerformanceData[key][group]['STDs'] = []
        for key in tempOutputData:
            for group in groups:
                totalPerformanceData[key][group]['values'].append(
                    180*tempOutputData[key][group]["test_"+metric]/np.pi
                )
                # totalPerformanceData[key][group]['values'].append(
                #     180*np.median(tempOutputData[key][group]["test_"+metric+"_list"])/np.pi
                # )
                # totalPerformanceData[key][group]['STDs'].append(
                #     180*np.std(tempOutputData[key][group]["test_"+metric+"_list"])/np.pi
                # )
    for key in totalPerformanceData.keys():
        index = np.where(
            key==np.array(list(totalPerformanceData.keys()))
        )[0][0]
        axs[index].spines["right"].set_visible(False)
        axs[index].spines["top"].set_visible(False)
        axs[index].set_xlabel("Babbling Duration (sec.)")
        axs[index].set_xticks(list(babblingDurations))
        axs[index].set_xticklabels([int(el) for el in babblingDurations])
        axs[index].set_ylabel("Avg. Performance ("+metric+" in deg.)")
        axs[index].set_title(labels[index])
        for i in range(len(groups)):
            axs[index].plot(
                babblingDurations,
                totalPerformanceData[key][groups[i]]['values'],
                c=colors[i]
            )
            # axs[index].fill_between(
            #     babblingDurations,
            #     (
            #         np.array(totalPerformanceData[key][groups[i]]['values'])
            #         + np.array(totalPerformanceData[key][groups[i]]['STDs'])
            #     ),
            #     (
            #         np.array(totalPerformanceData[key][groups[i]]['values'])
            #         - np.array(totalPerformanceData[key][groups[i]]['STDs'])
            #     ),
            #     color=colors[i],
            #     alpha='0.5'
            # )
        if index==1:
            axs[index].legend(groups,loc='upper right')

def generate_and_save_sensory_data(plant,x1d,sd,savePath=None):
    X1d = np.zeros((5,len(plant.time)))
    X1d[0,:] = LP_filt(100,x1d)
    X1d[1,:] = np.gradient(X1d[0,:],plant.dt)
    X1d[2,:] = np.gradient(X1d[1,:],plant.dt)
    X1d[3,:] = np.gradient(X1d[2,:],plant.dt)
    X1d[4,:] = np.gradient(X1d[3,:],plant.dt)

    Sd = np.zeros((3,len(plant.time)))
    Sd[0,:] = LP_filt(100,sd)
    Sd[1,:] = np.gradient(Sd[0,:],plant.dt)
    Sd[2,:] = np.gradient(Sd[1,:],plant.dt)

    X,U,_,_ = plant.forward_simulation_FL(X_o,X1d,Sd)

    additionalDict = {"X1d" : x1d, "Sd" : sd}

    plant.save_data(X,U,additionalDict=additionalDict,path=savePath)

def plot_experimental_data(experimentalData,returnFigs=True):

    # Sin Angle/Sin Stiffness
    fig1, (ax1a,ax1b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Sinusoidal Angle / Sinusoidal Stiffness")

    # Step Angle/Sin Stiffness
    fig2, (ax2a,ax2b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Step Angle / Sinusoidal Stiffness")

    # Sin Angle/Step Stiffness
    fig3, (ax3a,ax3b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Sinusoidal Angle / Step Stiffness")

    # Step Angle/Step Stiffness
    fig4, (ax4a,ax4b) = plt.subplots(2,1,figsize=(8,6),sharex=True)
    plt.suptitle("Step Angle / Step Stiffness")

    figs = [fig1,fig2,fig3,fig4]
    top_axs = [ax1a,ax2a,ax3a,ax4a]
    bot_axs = [ax1b,ax2b,ax3b,ax4b]
    subkeys = list(experimentalData['all'].keys())

    for i in range(4):
        top_axs[i].set_ylabel("Joint Angle (deg.)")
        top_axs[i].spines["right"].set_visible(False)
        top_axs[i].spines["top"].set_visible(False)
        top_axs[i].plot(
            plant.time,
            (
                180/np.pi
                *np.array(
                    experimentalData['all'][subkeys[i]]["expected_out"]
                ).T
            ),
            c='0.70',
            lw=2
        )
        bot_axs[i].set_xlabel("Time (s)")
        bot_axs[i].set_ylabel("Joint Angle Error (deg.)")
        bot_axs[i].spines["right"].set_visible(False)
        bot_axs[i].spines["top"].set_visible(False)

        for key in experimentalData.keys():
            index = np.where(
                key==np.array(list(experimentalData.keys()))
            )[0][0]
            top_axs[i].plot(
                plant.time,
                (
                    180/np.pi
                    * np.array(
                        experimentalData[key][subkeys[i]]["predicted_out"]
                    ).T
                ),
                c=colors[index]
            )
            bot_axs[i].plot(
                plant.time,
                (
                    180/np.pi
                    * np.array(
                        experimentalData[key][subkeys[i]]["test_error"]
                    ).T
                ),
                c=colors[index]
            )

        legendList = list(experimentalData.keys())
        legendList.insert(0,'Desired')
        ax1a.legend(legendList,loc="upper right")
        ax2a.legend(legendList,loc="upper right")
        ax3a.legend(legendList,loc="upper right")
        ax4a.legend(legendList,loc="upper right")

    if returnFigs==True:
        return(figs)
    else:
        plt.show()
#
# def plot_error_signal_power_spectrums(
#         Time,
#         experimentalData,
#         returnFigs=True,
#         addTitle=None,
#         returnFig=False
#     ):
#     baseTitle = "Avg. Error Signal Power Spectrum"
#     xLabel = "Frequency (Hz)"
#
#     if addTitle is None:
#         title = baseTitle
#     else:
#         assert type(addTitle) == str, "title must be a string."
#         title = baseTitle + "\n" + addTitle
#
    groups = ["all","bio","kinapprox","allmotor"]
#     groupNames = [
#         "All\nAvailable\nStates",
#         "The\nBio-Inspired\nSet",
#         "Motor Position\nand\nVelocity Only",
#         "All\nMotor\nStates"
#     ]
#     fig = plt.figure(figsize=(20,12))
#     plt.suptitle(title, fontsize=14)
#     ax1 = plt.subplot(221)
#     ax2 = plt.subplot(222)
#     ax3 = plt.subplot(223)
#     ax3.set_xlabel(xLabel, ha="center")
#     ax3.xaxis.set_label_coords(0.5, -0.1)
#     ax4 = plt.subplot(224)
#     axs = [ax1,ax2,ax3,ax4]
#
#     for i in range(4): # groups
#         axs[i].set_title(groupNames[i],y=0.95,color=colors[i])
#         axs[i].spines["top"].set_visible(False)
#         axs[i].spines["right"].set_visible(False)
#
#         freqs, psd = signal.welch(
#             self.babblingSignals[:, i],
#             1/self.plant.dt
#         )
#     fig2 = plt.figure(figsize=(5, 4))
#     ax2 = plt.gca()
#     plt.title('PSD: power spectral density')
#     plt.xlabel('Frequency')
#     plt.ylabel('Power')
#     plt.tight_layout()
#
#     if self.babblingSignals.shape == (len(Time[:-1]),):
#         numberOfSignals = 1
#     else:
#         numberOfSignals = self.babblingSignals.shape[1]
#     inputLineStyle = [None,"--"]
#     inputLines = []
#     for i in range(self.babblingSignals.shape[1]):
#         freqs, psd = signal.welch(
#             self.babblingSignals[:, i],
#             1/self.plant.dt
#         )
#         inputLine = ax1.plot(Time[:-1], self.babblingSignals[:, i],'r',ls= inputLineStyle[i])
#         inputLines.append(inputLine)
#         ax2.semilogx(freqs, psd, c='r',ls=inputLineStyle[i])
#
#     if numberOfSignals != 1:
#         ax1.legend(
#             inputLines,
#             ["Signal " + str(i+1) for i in range(numberOfSignals)],
#             loc="upper right"
#         )
#         ax2.legend(
#             ["Signal " + str(i+1) for i in range(numberOfSignals)],
#             loc="upper right"
#         )

def plot_training_performance(
        trainingData,
        numberOfEpochs,
        numberOfTrials,
        returnFig=True
    ):
    epochArray = np.arange(0,numberOfEpochs+1,1)

    fig = plt.figure(figsize=(8,6))
    plt.yscale("log")
    plt.title("Average Performance vs. Epoch\n" + "(" + str(numberOfTrials) + " Trials)")
    ax = plt.gca()
    ax.set_xlabel("Epoch #")
    ax.set_ylabel("Performance (RMSE in deg.)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xticks(list(np.linspace(0,numberOfEpochs,6)))
    ax.set_xticklabels([int(el) for el in ax.get_xticks()])

    for i in range(len(trainingData.keys())):
        key = list(trainingData.keys())[i]
        ax.plot(
            epochArray,
            180*np.sqrt(trainingData[key]["perf"])/np.pi,
            c=colors[i],
            lw=2
        )
    ax.legend(list(trainingData.keys()),loc='upper right')

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def plot_bar_plots(outputData,metric="MAE",returnFig=True):
    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."
    if metric == "MAE":
        baseTitle = "Bar Plots of MAE by Movement Type"
        ylabel = "Mean Absolute Error (deg.)"
        valueKey = "test_MAE"
    elif metric == 'STD':
        baseTitle = "Bar Plots of Error Std Dev by Movement Type"
        ylabel = "Error Standard Deviation (deg.)"
        valueKey = "test_STD"
    elif metric == 'RMSE':
        baseTitle = "Bar Plots of RMSE by Movement Type"
        ylabel = "Root Mean Squared Error (deg.)"
        valueKey = "test_RMSE"

    labels = [
        "Sinusoidal Angle \n Sinusoidal Stiffness",
        "Step Angle \n Sinusoidal Stiffness",
        "Sinusoidal Angle \n Step Stiffness",
        "Step Angle \n Step Stiffness"
    ]
    allValue = [
        (180/np.pi)*outputData[key]["all"][valueKey]
        for key in outputData.keys()
    ]
    bioValue = [
        (180/np.pi)*outputData[key]["bio"][valueKey]
        for key in outputData.keys()
    ]
    kinapproxValue = [
        (180/np.pi)*outputData[key]["kinapprox"][valueKey]
        for key in outputData.keys()
    ]
    allmotorValue = [
        (180/np.pi)*outputData[key]["allmotor"][valueKey]
        for key in outputData.keys()
    ]

    xticks = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots(figsize=(12,5))
    rects1 = ax.bar(
        xticks - 3*width/2, allValue, width,
        label="all", color=colors[0]
    )
    rects2 = ax.bar(
        xticks - width/2, bioValue, width,
        label="bio", color=colors[1]
    )
    rects3 = ax.bar(
        xticks + width/2, kinapproxValue, width,
        label="kinapprox", color=colors[2]
    )
    rects4 = ax.bar(
        xticks + 3*width/2, allmotorValue, width,
        label="allmotor", color=colors[3]
    )

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(ylabel)
    ax.set_title(baseTitle)
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if returnFig==True:
        return(fig)
    else:
        plt.show()

def return_radial_bins(errorArrays,jointAngleArrays,bins=12):
    theta_rays = np.arange(0,np.pi+1e-3,np.pi/bins)
    radial_bins={
        "bins" : bins,
        "maxMAE" : 0,
        "maxRMSE" : 0,
        "maxSTD" : 0,
        "all" : {},
        "bio" : {},
        "kinapprox" : {},
        "allmotor" : {}
    }
    for i in range(len(groups)):
        tempJointAngle = (jointAngleArrays[i,:]-np.pi/2).flatten()
        for j in range(len(theta_rays)-1):
            bin_name = (
                '{:0.1f}'.format(180*theta_rays[j]/np.pi)
                + " to "
                + '{:0.1f}'.format(180*theta_rays[j+1]/np.pi)
            )
            radial_bins[groups[i]][bin_name] = {}
            indices = np.array(
                np.where(
                    np.logical_and(
                        tempJointAngle<theta_rays[j+1],
                        tempJointAngle>=theta_rays[j]
                    )
                )
            )
            radial_bins[groups[i]][bin_name]["abs errors"] = np.array([
                (180/np.pi)*abs(errorArrays[i,k])
                for k in indices
            ]) # in degrees
            radial_bins[groups[i]][bin_name]["errors"] = np.array([
                (180/np.pi)*errorArrays[i,k]
                for k in indices
            ]) # in degrees

            ### Mean absolute error
            radial_bins[groups[i]][bin_name]["MAE"] = \
                radial_bins[groups[i]][bin_name]["abs errors"].mean() # in degrees
            radial_bins["maxMAE"] = max([
                radial_bins["maxMAE"],
                radial_bins[groups[i]][bin_name]["MAE"]
            ]) # in degrees

            ### Root mean squared error
            radial_bins[groups[i]][bin_name]["RMSE"] = np.sqrt(
                (radial_bins[groups[i]][bin_name]["errors"]**2).mean()
            ) # in degrees
            radial_bins["maxRMSE"] = max([
                radial_bins["maxRMSE"],
                radial_bins[groups[i]][bin_name]["RMSE"]
            ]) # in degrees
            radial_bins[groups[i]][bin_name]["abs error std"] = \
                radial_bins[groups[i]][bin_name]["abs errors"].std() # in degrees
            radial_bins[groups[i]][bin_name]["min abs error"] = \
                radial_bins[groups[i]][bin_name]["errors"].min() # in degrees
            radial_bins[groups[i]][bin_name]["max abs error"] = \
                radial_bins[groups[i]][bin_name]["errors"].max() # in degrees

            radial_bins[groups[i]][bin_name]["avg error"] = \
                radial_bins[groups[i]][bin_name]["errors"].mean() # in degrees
            radial_bins[groups[i]][bin_name]["STD"] = \
                radial_bins[groups[i]][bin_name]["errors"].std() # in degrees
            radial_bins["maxSTD"] = max([
                radial_bins["maxSTD"],
                radial_bins[groups[i]][bin_name]["STD"]
            ]) # in degrees
    return(radial_bins)

def plot_polar_bar_plots(
        radial_bins,
        metric="MAE",
        addTitle=None,
        returnFig=False
    ):

    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default) or 'STD'."
    if metric == "MAE":
        baseTitle = "Polar Bar Plots of MAE vs. Joint Angle"
        xLabel = "Log MAE (in deg.)"
        maxValue = np.log10(radial_bins["maxMAE"])+2
        offset = 2
    elif metric == 'STD':
        baseTitle = "Polar Bar Plots of Error Std Dev vs. Joint Angle"
        xLabel = "Log Error Std Dev (in deg.)"
        maxValue = np.log10(radial_bins["maxSTD"])+2
        offset = 2
    elif metric == 'RMSE':
        baseTitle = "Polar Bar Plots of RMSE vs. Joint Angle"
        xLabel = "Log RMSE (in deg.)"
        maxValue = np.log10(radial_bins["maxRMSE"])+2
        offset = 2

    # assert maxValue<3.3, "Bounds not configured for values this large. Please check values again and determine if bounds need to be changed."

    if addTitle is None:
        title = baseTitle
    else:
        assert type(addTitle)==str, "title must be a string."
        title = baseTitle + "\n" + addTitle

    subTitles = [
        "All\nAvailable\nStates",
        "The\nBio-Inspired\nSet",
        "Motor Position\nand\nVelocity Only",
        "All\nMotor\nStates"
    ]
    fig = plt.figure(figsize=(20,12))
    plt.suptitle(title,fontsize=14)
    ax1=plt.subplot(221)
    ax2=plt.subplot(222)
    ax3=plt.subplot(223)
    ax3.set_xlabel(xLabel,ha="center")
    ax3.xaxis.set_label_coords(0.8, -0.1)
    ax4=plt.subplot(224)
    axs=[ax1,ax2,ax3,ax4]

    slices = radial_bins['bins']
    theta_rays = np.arange(0,np.pi+1e-3,np.pi/slices)
    for i in range(4):
        for j in range(len(theta_rays)-1):
            bin_name = (
                '{:0.1f}'.format(180*theta_rays[j]/np.pi)
                + " to "
                + '{:0.1f}'.format(180*theta_rays[j+1]/np.pi)
            )
            if j%2==0:
                axs[i].add_patch(
                    Wedge(
                        (0,0), 3.3,
                        (180/np.pi)*theta_rays[j],
                        (180/np.pi)*theta_rays[j+1],
                        color = "0.85"
                    )
                )
            axs[i].add_patch(
                Wedge(
                    (0,0),
                    np.log10(radial_bins[groups[i]][bin_name][metric])+offset,
                    (180/np.pi)*theta_rays[j],
                    (180/np.pi)*theta_rays[j+1],
                    color = colors[i],
                    alpha=0.65
                )
            )
        axs[i].set_aspect('equal')
        axs[i].set_ylim([0,3.3])
        axs[i].set_xlim([-3.3,3.3])
        xticks = np.arange(0,3+1e-3,1)
        xticks = np.concatenate([
            -np.array(list(reversed(xticks[1:]))),
            xticks[1:]
        ])
        axs[i].set_xticks(xticks)
        axs[i].set_xticklabels([
            r"$10^{1}$",r"$10^{0}$",r"$10^{-1}$",
            r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$"
        ])
        axs[i].add_patch(Wedge((0,0),1,0,360,color ='w'))
        xticksMinor = np.concatenate(
            [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
        )
        xticksMinor = np.concatenate(
            [-np.array(list(reversed(xticksMinor))),xticksMinor]
        )
        xticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in xticksMinor]
        axs[i].set_xticks(xticksMinor,minor=True)

        yticks = list(np.arange(0,3+1e-3,1))
        axs[i].set_yticks(yticks[1:])
        axs[i].set_yticklabels(["" for tick in axs[i].get_yticks()])
        yticksMinor = np.concatenate(
            [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
        )
        yticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in yticksMinor]
        axs[i].set_yticks(yticksMinor,minor=True)

        radii = list(axs[i].get_yticks())
        theta = np.linspace(0,np.pi,201)
        for radius in radii:
            axs[i].plot(
                [radius*np.cos(el) for el in theta],
                [radius*np.sin(el) for el in theta],
                "k",
                lw=0.5
            )
        axs[i].plot([1,3.3],[0,0],'k',linewidth=1.5)# double lw because of ylim
        axs[i].plot([-3.3,-1],[0,0],'k',linewidth=1.5)# double lw because of ylim
        axs[i].plot([0,0],[1,3.3],'k',linewidth=0.5)

        axs[i].text(
            0,0.25,
            subTitles[i],
            color=colors[i],
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=16
        )
        axs[i].spines['bottom'].set_position('zero')
        axs[i].spines['left'].set_position('zero')
        axs[i].spines['bottom'].set_visible(False)
        axs[i].spines['left'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].spines['right'].set_visible(False)
    if returnFig==True:
        return(fig)

def plot_polar_bar_plots_together(
        radial_bins,
        metric="MAE",
        addTitle=None,
        returnFig=False
    ):

    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."
    if metric == "MAE":
        title = "MAE\n vs.\n Joint Angle"
        xLabel = "MAE (in deg.)"
        maxValue = np.log10(radial_bins["maxMAE"])+2
    elif metric == 'STD':
        title = "Error Std Dev\n vs.\n Joint Angle"
        xLabel = "Error Std Dev (in deg.)"
        maxValue = np.log10(radial_bins["maxSTD"])+2
    elif metric == 'RMSE':
        title = "RMSE\n vs.\n Joint Angle"
        xLabel = "Log RMSE (in deg.)"
        maxValue = np.log10(radial_bins["maxRMSE"])+2
        offset = 2

    # assert maxValue<3.3, "Bounds not configured for values this large. Please check values again and determine if bounds need to be changed."

    basePath = path.dirname(__file__)
    filePath = path.abspath(path.join(basePath, "..", "SupplementaryFigures", "Schematic_1DOF2DOA_system.png"))
    im = Image.open(filePath)
    height = im.size[1]
    width = im.size[0]
    aspectRatio = width/height

    fig = plt.figure(figsize=(10,8))
    if addTitle is not None:
        assert type(addTitle)==str, "title must be a string."
        plt.title(addTitle,fontsize=16,y=-0.35)


    newHeight = int(np.ceil(0.15*fig.bbox.ymax)+10)
    size = int(newHeight*aspectRatio),newHeight
    im.thumbnail(size, Image.ANTIALIAS)
    fig.figimage(im, fig.bbox.xmax/2 - im.size[0]/2.2, 0.95*im.size[1],zorder=10)
    ax = plt.gca()

    slices = radial_bins['bins']
    theta_rays = np.arange(0,np.pi+1e-3,np.pi/slices)
    sectorWidth = np.pi/slices/5
    theta_rays_times_4 = []
    for j in range(len(theta_rays)-1):
        midAngle = (theta_rays[j+1]+theta_rays[j])/2
        theta_rays_times_4.append(
            [(midAngle + i*sectorWidth) for i in [-2,-1,0,1]]
        )
    theta_rays_times_4 = np.concatenate(theta_rays_times_4)

    for j in range(len(theta_rays)-1):
        bin_name = (
            '{:0.1f}'.format(180*theta_rays[j]/np.pi)
            + " to "
            + '{:0.1f}'.format(180*theta_rays[j+1]/np.pi)
        )
        if j%2==0:
            ax.add_patch(
                Wedge(
                    (0,0), 3.3,
                    (180/np.pi)*theta_rays[j],
                    (180/np.pi)*theta_rays[j+1],
                    color = "0.85"
                )
            )
        for i in range(len(groups)):
            ax.add_patch(
                Wedge(
                    (0,0),
                    np.log10(radial_bins[groups[i]][bin_name][metric])+2,
                    (180/np.pi)*theta_rays_times_4[4*j+i],
                    (180/np.pi)*(theta_rays_times_4[4*j+i]+sectorWidth),
                    color = colors[i],
                    alpha=0.65
                )
            )

    ax.set_aspect('equal')
    ax.set_ylim([0,3.3])
    ax.set_xlim([-3.3,3.3])
    xticks = np.arange(0,3+1e-3,1)
    xticks = np.concatenate([-np.array(list(reversed(xticks[1:]))),xticks[1:]])
    ax.set_xticks(xticks)
    ax.set_xticklabels([
        r"$10^{1}$",r"$10^{0}$",r"$10^{-1}$",
        r"$10^{-1}$",r"$10^{0}$",r"$10^{1}$"
    ])
    ax.add_patch(Wedge((0,0),1,0,360,color ='w'))
    xticksMinor = np.concatenate(
        [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
    )
    xticksMinor = np.concatenate(
        [-np.array(list(reversed(xticksMinor))),xticksMinor]
    )
    xticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in xticksMinor]
    ax.set_xticks(xticksMinor,minor=True)

    yticks = list(np.arange(0,3+1e-3,1))
    ax.set_yticks(yticks[1:])
    ax.set_yticklabels(["" for tick in ax.get_yticks()])
    yticksMinor = np.concatenate(
        [np.linspace(10**(i),10**(i+1),10)[1:-1] for i in range(-1,1)]
    )
    yticksMinor = [np.sign(el)*(np.log10(abs(el))+2) for el in yticksMinor]
    ax.set_yticks(yticksMinor,minor=True)

    radii = list(ax.get_yticks())
    theta = np.linspace(0,np.pi,201)
    for radius in radii:
        ax.plot(
            [radius*np.cos(el) for el in theta],
            [radius*np.sin(el) for el in theta],
            "k",
            lw=0.5
        )
    ax.plot([1,3.3],[0,0],'k',linewidth=1.5)# double lw because of ylim
    ax.plot([-3.3,-1],[0,0],'k',linewidth=1.5)# double lw because of ylim
    ax.plot([0,0],[1,3.3],'k',linewidth=0.5)

    props = dict(
        boxstyle='round',
        facecolor='w',
        edgecolor='0.70'
    )
    ax.text(
        -0.9*3,3,
        title,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=16,
        bbox=props
    )
    ax.text(
        2,-0.35,
        xLabel,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12
    )
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if returnFig==True:
        return(fig)

def plot_all_polar_bar_plots(outputData,metric,returnFigs=True):
    assert type(metric)==str and metric in ["MAE","STD","RMSE"], "metric must be either 'MAE' (default), 'RMSE' or 'STD'."

    ### radial average error versus positions
    labels = [
        "(Sinusoidal Angle / Sinusoidal Stiffness)",
        "(Step Angle / Sinusoidal Stiffness)",
        "(Sinusoidal Angle / Step Stiffness)",
        "(Step Angle / Step Stiffness)"
    ]

    figs = []
    for key in outputData.keys():
        index = np.where(
            key==np.array(list(outputData.keys()))
        )[0][0]
        jointAngleArrays = np.concatenate(
            [
                outputData[key][subkey]['expected_out'].flatten()[np.newaxis,:]
                for subkey in groups
            ],
            axis=0
        ) # in radians
        errorArrays = np.concatenate(
            [
                outputData[key][subkey]['test_error'].flatten()[np.newaxis,:]
                for subkey in groups
            ],
            axis=0
        ) # in radians

        radial_bins = return_radial_bins(errorArrays,jointAngleArrays,bins=15)
        tempFig = plot_polar_bar_plots(
            radial_bins,
            metric=metric,
            addTitle=labels[index],
            returnFig=True
        )
        figs.append(tempFig)

        tempFig = plot_polar_bar_plots_together(
            radial_bins,
            metric=metric,
            addTitle=labels[index],
            returnFig=True
        )
        figs.append(tempFig)

    if returnFigs==True:
        return(figs)
    else:
        plt.show()

def plot_all_error_distributions(outputData,returnFigs=True):
    labels = [
        "(Sinusoidal Angle / Sinusoidal Stiffness)",
        "(Step Angle / Sinusoidal Stiffness)",
        "(Sinusoidal Angle / Step Stiffness)",
        "(Step Angle / Step Stiffness)"
    ]

    figs = []
    for key in outputData.keys():
        index = np.where(
            key==np.array(list(outputData.keys()))
        )[0][0]
        tempFig, axs = plt.subplots(2,2,figsize=(10,10))
        plt.suptitle(labels[index],fontsize=16)
        for i in range(len(groups)):
            data = 180*outputData[key][groups[i]]['test_error'].flatten()/np.pi
            axs[int(i/2)][i%2].hist(
                data,
                weights=np.ones(len(data)) / len(data),
                bins=60,
                color=colors[i]
            )
            axs[int(i/2)][i%2].set_yticklabels(["{:.1f}%".format(100*el) for el in axs[int(i/2)][i%2].get_yticks()])
            axs[int(i/2)][i%2].set_title(
                groups[i],
                fontsize=14,
                color=colors[i]
            )
            axs[int(i/2)][i%2].spines['top'].set_visible(False)
            axs[int(i/2)][i%2].spines['right'].set_visible(False)
        figs.append(tempFig)

    if returnFigs==True:
        return(figs)

def plot_consolidated_data(babblingDuration,directory=None,metrics=None):
    if metrics is None:
        metrics = ["MAE"]
        metricKeys = ["test_MAE"]
    else:
        assert type(metrics)==list, "metrics must be a list of strings."
        for metric in metrics:
            assert metric in ["RMSE","STD","MAE"], 'Invalid metric! metrics must include "RMSE","STD", or "MAE".'
        metricKeys = ["test_"+metric for metric in metrics]

    ### get the testing trial directories
    if directory==None:
        directory = "experimental_trials/"
    else:
        assert path.isdir(directory), "Enter a valid directory."

    folderName = (
        'Consolidated_Trials_'
        + '{:03d}'.format(int(babblingDuration))
        + 's/'
    )

    trialDirectories = [
        name for name in listdir(directory)
        if path.isdir(path.join(directory, name))
        and name[:2]=='20'
    ]
    numberOfTrials = len(trialDirectories)

    # Training Data

    totalTrainingData = {
        "all" : {"perf":{},"avg_best_perf":{}},
        "bio" : {"perf":{},"avg_best_perf":{}},
        "kinapprox" : {"perf":{},"avg_best_perf":{}},
        "allmotor" : {"perf":{},"avg_best_perf":{}}
    }

    for n in range(numberOfTrials):
        with open(directory+trialDirectories[n]+'/trainingData.pkl', 'rb') as handle:
            tempTrainingData = pickle.load(handle)
        if n == 0:
            for key in tempTrainingData:
                totalTrainingData[key]["perf"] = np.array(
                    tempTrainingData[key]["tr"]["perf"]._data
                ) / numberOfTrials
                totalTrainingData[key]["avg_best_perf"] = (
                    tempTrainingData[key]["tr"]["best_perf"]
                    / numberOfTrials
                )
        else:
            for key in tempTrainingData:
                totalTrainingData[key]["perf"] += np.array(
                    tempTrainingData[key]["tr"]["perf"]._data
                ) / numberOfTrials
                totalTrainingData[key]["avg_best_perf"] += (
                    tempTrainingData[key]["tr"]["best_perf"]
                    / numberOfTrials
                )
    numberOfEpochs = len(totalTrainingData['all']["perf"])-1

    plot_training_performance(totalTrainingData,numberOfEpochs,numberOfTrials)

    saveParams = {
        "Babbling Duration" : babblingDuration,
        "Number of Trials" : numberOfTrials,
        "Number of Epochs" : numberOfEpochs
    }
    save_figures(
        directory,
        "perf_v_epoch",
        saveParams,
        subFolderName=folderName,
        saveAsPDF=True
    )
    plt.close('all')

    # Experimental Data

    totalOutputData = {}

    subsubkey_list = [
        "test_error",
        "expected_out"
    ]
    [subsubkey_list.append(metricKey) for metricKey in metricKeys]

    for n in range(numberOfTrials):
        with open(directory+trialDirectories[n]+'/experimentalData.pkl', 'rb') as handle:
            tempOutputData = pickle.load(handle)
        if n == 0:
            totalOutputData = tempOutputData
            for key in tempOutputData:
                for subkey in tempOutputData[key]:
                    for subsubkey in metricKeys:
                        totalOutputData[key][subkey][subsubkey+"_list"] = [
                            totalOutputData[key][subkey][subsubkey]
                        ]
                        totalOutputData[key][subkey][subsubkey] = (
                            totalOutputData[key][subkey][subsubkey]
                            / numberOfTrials
                        )
        else:
            for key in tempOutputData:
                for subkey in tempOutputData[key]:
                    for subsubkey in metricKeys:
                        totalOutputData[key][subkey][subsubkey+"_list"].append(
                            tempOutputData[key][subkey][subsubkey]
                        )
                        totalOutputData[key][subkey][subsubkey] += (
                            tempOutputData[key][subkey][subsubkey]
                            / numberOfTrials
                        )
        # else:
        #     for key in tempOutputData:
        #         for subkey in tempOutputData[key]:
        #             for subsubkey in subsubkey_list:
        #                 if subsubkey not in metricKeys:
        #                     totalOutputData[key][subkey][subsubkey] = \
        #                         np.concatenate([
        #                             totalOutputData[key][subkey][subsubkey],
        #                             tempOutputData[key][subkey][subsubkey]
        #                         ],
        #                         axis=0)
        #                 else:
        #                     totalOutputData[key][subkey][subsubkey+"_list"].append(
        #                         tempOutputData[key][subkey][subsubkey]
        #                     )
        #                     totalOutputData[key][subkey][subsubkey] += (
        #                         tempOutputData[key][subkey][subsubkey]
        #                         / numberOfTrials
        #                     )
        # delete trial directory
        shutil.rmtree(directory+trialDirectories[n])

    # plot_all_error_distributions(totalOutputData,returnFigs=True)
    #
    # save_figures(
    #     directory+folderName,
    #     "err_dist",
    #     {},
    #     subFolderName="Error_Distributions/"
    # )
    # plt.close('all')

    for metric in metrics:
        fig = plot_bar_plots(
            totalOutputData,
            metric=metric,
            returnFig=True
        )
        # figs = plot_all_polar_bar_plots(
        #     totalOutputData,
        #     metric=metric,
        #     returnFigs=True
        # )
        # figs.insert(0,fig)
        figs = [fig]

        save_figures(
            directory+folderName,
            metric,
            {},
            figs=figs,
            subFolderName=metric+"/",
            saveAsPDF=True
        )
        plt.close('all')
    """
    Consolidated to include:
        - AVERAGE metrics (this is not the same as the metric average for all data points, but instead the average metric across trials)
        - metric lists
    """
    consolidatedOutputData = {}
    for key in totalOutputData.keys():
        consolidatedOutputData[key]={}
        for subkey in totalOutputData[key].keys():
            consolidatedOutputData[key][subkey]={}
            for metric in metricKeys:
                consolidatedOutputData[key][subkey][metric] = \
                    totalOutputData[key][subkey][metric]
                consolidatedOutputData[key][subkey][metric+"_list"] = \
                    totalOutputData[key][subkey][metric+"_list"]
    fileName = (
        "consolidatedOutputData.pkl"
    )
    with open(directory+folderName+fileName, 'wb') as handle:
        pickle.dump(
            consolidatedOutputData,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )

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

    ### ANN parameters
    ANNParams = {
        "Number of Layers" : 15,
        "Number of Epochs" : 50,
        "Number of Trials" : 50,
    }

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
        '-layers',
        type=int,
        help='Number of layers for each network to train (single hidden layer). Default is given by ANNParams.',
        default=ANNParams["Number of Layers"]
    )
    parser.add_argument(
        '-trials',
        type=int,
        default=1,
        help='Number of trials to run. Default is 1.'
    )
    parser.add_argument(
        '-metrics',
        type=str,
        nargs="+",
        default='MAE',
        help="Metrics to be compared. Should be either MAE, RMSE, or STD. Default is MAE."
    )
    parser.add_argument(
        '--consol',
        action="store_true",
        help='Consolidate all trials and generate comparison plots. Default is false.'
    )
    parser.add_argument(
        '--consolALL',
        action="store_true",
        help='Consolidate all trials and generate comparison plots. Default is false.'
    )
    args = parser.parse_args()
    if type(args.metrics)==str:
        metrics = [args.metrics]
    else:
        metrics = args.metrics
    for metric in metrics:
        assert metric in ["RMSE","MAE","STD"], "Invalid metric! Must be either 'RMSE', 'MAE', or 'STD'"

    if args.consol==True:
        plot_consolidated_data(args.dur,metrics=metrics)
    else:
        if args.consolALL==True:
            pathName = (
                'experimental_trials/'
            )
            folderName = (
                'All_Consolidated_Trials_'
                + '{:03d}'.format(int(args.trials))
                + '/'
            )
            plot_babbling_duration_vs_average_performance('RMSE')
            save_figures(
                pathName,
                "propAmp",
                {"Number of Trials":args.trials},
                subFolderName=folderName,
                saveAsPDF=True
            )
            plt.close('all')
        else:
            startTime = time.time()
            trialStartTime = startTime
            for i in range(args.trials):
                print("Running Trial " + str(i+1) + "/" + str(args.trials))

                plantParams["dt"] = args.dt
                ANNParams["Number of Epochs"] = args.epochs
                ANNParams["Number of Layers"] = args.layers

                ### Generate plant
                tempSimulationDuration = 30
                plantParams["Simulation Duration"] = tempSimulationDuration
                plant = plant_pendulum_1DOF2DOF(plantParams)
                X_o = plant.return_X_o(np.pi,[0,0])
                passProbability = 0.0005

                allDone = False
                count = 0
                while allDone==False:
                    ### Generate Testing DATA (Angle Step, Stiffness Step)
                    basePath = "experimental_trials/"

                    print("Angle Step / Stiffness Step")
                    filePath = (basePath + "angleStep_stiffStep_")
                    if path.exists(filePath+"outputData.mat"):
                        print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                    else:
                        [x1d,sd] = plant.generate_desired_trajectory_STEPS(
                            passProbability,'both'
                        )
                        try:
                            generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
                        except:
                            pass

                    ### Generate Testing DATA (Angle Step, Stiffness Sinusoid)

                    print("Angle Step / Stiffness Sinusoid")
                    filePath = (basePath + "angleStep_stiffSin_")
                    if path.exists(filePath+"outputData.mat"):
                        print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                    else:
                        x1d = plant.generate_desired_trajectory_STEPS(passProbability,'angle')
                        sd = plant.generate_desired_trajectory_SINUSOIDAL('stiffness')
                        try:
                            generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
                        except:
                            pass

                    ### Generate Testing DATA (Angle Sinusoid, Stiffness Step)

                    print("Angle Sinusoid / Stiffness Step")
                    filePath = (basePath + "angleSin_stiffStep_")
                    if path.exists(filePath+"outputData.mat"):
                        print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                    else:
                        x1d = plant.generate_desired_trajectory_SINUSOIDAL('angle')
                        sd = plant.generate_desired_trajectory_STEPS(
                            passProbability,'stiffness'
                        )
                        try:
                            generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
                        except:
                            pass

                    ### Generate Testing DATA (Angle Sinusoid, Stiffness Sinusoid)

                    print("Angle Step / Stiffness Step")
                    filePath = (basePath + "angleSin_stiffSin_")
                    if path.exists(filePath+"outputData.mat"):
                        print("ALREADY COMPLETED!!! (DELETE TO RUN AGAIN...)")
                    else:
                        x1d = plant.generate_desired_trajectory_SINUSOIDAL('angle')
                        sd = plant.generate_desired_trajectory_SINUSOIDAL('stiffness')
                        try:
                            generate_and_save_sensory_data(plant,x1d,sd,savePath=filePath)
                        except:
                            pass

                    if np.all([
                        path.exists(basePath+el+"outputData.mat")
                        for el in [
                            "angleStep_stiffStep_",
                            "angleStep_stiffSin_",
                            "angleSin_stiffStep_",
                            "angleSin_stiffSin_"
                        ]
                    ]):
                        allDone = True
                    else:
                        count+=1
                        assert count<10, "Too many unsuccessful trials, please check code and run again."

                ### Generate babbling data and SAVE ALL FIGURES AND DATA IN SPECIFIC FOLDER
                plantParams["Simulation Duration"] = args.dur # returned to original value.

                ANN = neural_network(ANNParams,babblingParams,plantParams)
                experimentalData = ANN.run_experimental_trial()

                # ### Plot experimental data
                # figs = plot_experimental_data(experimentalData,returnFigs=True)

                # SAVE EXPERIMENTAL DATA TO TRIAL FOLDER
                formattedData = {
                    "all" : {},
                    "bio" : {},
                    "kinapprox" : {},
                    "allmotor" : {}
                }
                formattedData = {}
                for key in experimentalData["all"]:
                    formattedData[key] = {}
                    for subkey in experimentalData:
                        formattedData[key][subkey] = {}
                        for subsubkey in experimentalData[subkey][key]:
                            if type(experimentalData[subkey][key][subsubkey])==float:
                                formattedData[key][subkey][subsubkey] = \
                                    experimentalData[subkey][key][subsubkey]
                            else:
                                formattedData[key][subkey][subsubkey] = np.array(
                                    experimentalData[subkey][key][subsubkey]._data
                                )
                experimentalData = formattedData
                with open(ANN.trialPath + '\experimentalData.pkl', 'wb') as handle:
                    pickle.dump(experimentalData, handle, protocol=pickle.HIGHEST_PROTOCOL)

                for metric in metrics:
                    ### bar plots
                    fig = plot_bar_plots(experimentalData,metric=metric)

                    ### radial average error versus positions
                    figs = [fig]
                    newFigs=plot_all_polar_bar_plots(experimentalData,metric)
                    [figs.append(fig) for fig in newFigs]

                    ### save figs
                    save_figures(
                        ANN.trialPath,
                        "propAmp",
                        {},
                        figs=figs,
                        subFolderName=metric+"/"
                    )
                    plt.close('all')
                print('\a')
                runTime = time.time()-startTime
                seconds = runTime % (24 * 3600)
                hour = seconds // 3600
                seconds %= 3600
                minutes = seconds // 60
                seconds %= 60
                runTime = "%d:%02d:%02d" % (hour, minutes, seconds)
                trialRunTime = time.time()-trialStartTime
                seconds = trialRunTime % (24 * 3600)
                hour = seconds // 3600
                seconds %= 3600
                minutes = seconds // 60
                seconds %= 60
                trialRunTime = "(+%d:%02d:%02d)" % (hour, minutes, seconds)
                trialStartTime = time.time()
                print('Run Time: ' + runTime + " " + trialRunTime + "\n")
