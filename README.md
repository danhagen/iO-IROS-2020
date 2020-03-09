# iO-IROS-2020
### ***(insideOut)* Submission for IEEE IROS 2020:**
## Bio-inspired Foundation for Joint Angle Estimation from Non-Collocated Sensors in Tendon-Driven Robotics
### Daniel A. Hagen, Ali Marjaninejad, & Francisco J. Valero-Cuevas

Estimates of limb posture are critical for the control of robotic systems. This is generally accomplished by utilizing on-location joint angle encoders which may complicate the design, increase limb inertia, and add noise to the system. Conversely, some innovative or smaller robotic morphologies can benefit from non-collocated sensors when encoder size becomes prohibitively larger or the joints are less accessible or subject to damage (e.g., distal joints of a robotic hand or foot sensors subject to repeated impact). These concerns are especially important for tendon-driven systems where motors (and their sensors) are not placed at the joints. This repository represents a framework for joint angle estimation by which artificial neural networks (ANNs) use limited-experience from motor babbling to predict joint angles. We draw our inspiration for this project from Nature where (i) muscles and tendons have mechanoreceptors, (ii) there are *no dedicated joint-angle sensors*, and (iii) dedicated neural networks perform *sensory fusion*. To demonstrate this algorithm, we simulated an inverted pendulum driven by an agonist-antagonist pair of motors that pull on tendons with nonlinear elasticity.

## Installation from GitHub
```bash
git clone https://github.com/danhagen/iO-IROS-2020.git && cd iO-IROS-2020/src
pip install -r requirements.txt
pip install .
```

Please note that you can find help for many of the python functions in this repository by using the command `run <func_name> -h`.

## The Plant 

<p align="center">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/Schematic_1DOF2DOA_system.png?raw=true">
</p>

Here we used a physical inverted pendulum that was controlled by two simulated brushed DC motors (i.e., backdriveable) that pulled on tendons with nonlinear (exponential) stiffness. This plant can either be given feedfoward inputs or controlled via a *feedback linearization controller* that takes advantage of the fact that joint stiffness and joint angle can be controlled independently. Simply prescribe trajectories for both output measures and the controller will track it.

The default `run plant.py` command will test the feedback linearization algorithm. Choosing the options `--saveFigures` will save the figures and `--animate` will animate the output.


## Generating Babbling Data
In order to generate motor babbling data, we use the class `motor_babbling_1DOF2DOA`. The default `run motor_babbling_1DOF2DOA.py` will produce plots of random motor babbling and the resulting states of the plant. Figures will be saved in a time-stamped folder You also have the option to animate the babbling data (`--animate`). 

<p align="center">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/babblingInputs.png?raw=true">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/Plant_States_vs_Time_from_Babbling.png?raw=true">
</p>

## Train Articifical Neural Networks
To build, train, and test these ANNs, use `build_NN_1DOF2DOA` and `test_NN_1DOF2DOA`.

## Run Multiple Trials and Plot All Data
To run the entire experiment, run `run run_multiple_trials_with_different_babbling_durations.py`. This will sweep across babbling durations (30,45,...,360) seconds, train multiple ANNs (*default*:50 trials), and plot average performances. You can choose to plot metrics such as mean absolute error (MAE), root mean squared error (RMSE), or standard deviation of the error (STD) by adding the additional arguments `-metrics [METRICS ...]`. 

## Animate a Single Trial (All 4 ANNs Over 4 Different Movements)
To visualize the performance of ANNs and their ability to generalize to other movement tasks, use the function `animate_sample_trials.py`. This will create an animation of how well each ANN did at predicting joint angle and will sweep across 4 different movements (joint angle and stiffness are either sinusoidal or point-to-point). 


  
 <a href="https://youtu.be/w0AV4tzIW98"><img src="https://user-images.githubusercontent.com/16945786/75698768-24aca280-5c64-11ea-8b74-4999e1bd62b5.gif" alt="Youtube Video Link"></a>
</p>

