from test_NN_1DOF2DOA_ncTDS_ObservabilityExperiment import *
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

basePath = "white_noise_babbling_trials/Experimental_Trials_001/"
for fileNameBase in [
        "angleSin_stiffSin_",
        "angleStep_stiffSin_",
        "angleSin_stiffStep_",
        "angleStep_stiffStep_"
    ]:
    assert path.exists(basePath+fileNameBase+"outputData.mat"), "No experimental data to test. Please run FBL to generate data."

if __name__=='__main__':
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

        run_multiple_trials_with_different_babbling_durations.py

        -----------------------------------------------------------------------------

        Runs multiple ANNs for different babbling durations to find the average best performance across 4 different movement tasks.

        -----------------------------------------------------------------------------'''),
        epilog=textwrap.dedent('''\
        -----------------------------------------------------------------------------

        Written by Daniel A. Hagen (2020/01/29)

        -----------------------------------------------------------------------------'''
        )
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
        default=['RMSE'],
        help="Metrics to be compared. Should be either MAE, RMSE, or STD. Default is MAE."
    )
    args = parser.parse_args()
    if type(args.metrics)==str:
        metrics = [args.metrics]
    else:
        metrics = args.metrics
    for metric in metrics:
        assert metric in ["RMSE","MAE","STD"], "Invalid metric! Must be either 'RMSE', 'MAE', or 'STD'"

    babblingDurations = list(np.arange(30,360+1,15))
    numberOfTrials = args.trials

    for dur in babblingDurations:

        startTime = time.time()
        trialStartTime = startTime
        print("Babbling Duration: " + str(dur) + "s")
        for i in range(numberOfTrials):
            print("Running Trial " + str(i+1) + "/" + str(numberOfTrials))

            plantParams["Simulation Duration"] = int(dur) # returned to original value.

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
                # newFigs=plot_all_polar_bar_plots(experimentalData,metric)
                # [figs.append(fig) for fig in newFigs]

                ### save figs
                save_figures(
                    ANN.trialPath,
                    "perf_vs_bab_dur",
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

        print("Consolidating Data from " + str(dur) + "s Babbling Trials...")
        plot_consolidated_data(dur,metrics=args.metrics)

    pathName = (
        'white_noise_babbling_trials/Experimental_Trials_001/'
    )
    folderName = (
        'All_Consolidated_Trials_'
        + '{:03d}'.format(int(args.trials))
        + '/'
    )
    print("Plotting all data!")
    for metric in metrics:
        plot_babbling_duration_vs_average_performance(metric)
        save_figures(
            pathName,
            "perf_v_bab_dur_"+metric,
            {"Number of Trials":args.trials},
            subFolderName=folderName,
            saveAsPDF=True
        )
        plt.close('all')
