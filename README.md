<h1 align="center">iO-IROS-2020</br><em>(insideOut)</em></h1>
<h2 align="center">Bio-inspired Foundation for Joint Angle Estimation</br>from Non-Collocated Sensors in Tendon-Driven Robotics</h2>
<h3 align="center">Daniel A. Hagen, Ali Marjaninejad, & Francisco J. Valero-Cuevas</h3>

<h3 align="center"> 
  Accepted Abstract/Oral Presentation for</br>
  IEEE International Conference on Intelligent Robots and Systems (IROS) 2020</br> 
  (Preprint Available <a href="https://valerolab.org/Papers/IROS_2020_Accepted_Article_Preprint.pdf">Here</a>.)
</h3>

Estimates of limb posture are critical for the control of robotic systems. This is generally accomplished by utilizing on-location joint angle encoders which may complicate the design, increase limb inertia, and add noise to the system. Conversely, some innovative or smaller robotic morphologies can benefit from non-collocated sensors when encoder size becomes prohibitively larger or the joints are less accessible or subject to damage (e.g., distal joints of a robotic hand or foot sensors subject to repeated impact). These concerns are especially important for tendon-driven systems where motors (and their sensors) are not placed at the joints. This repository represents a framework for joint angle estimation by which artificial neural networks (ANNs) use limited-experience from motor babbling to predict joint angles. We draw our inspiration for this project from Nature where (i) muscles and tendons have mechanoreceptors, (ii) there are *no dedicated joint-angle sensors*, and (iii) dedicated neural networks perform *sensory fusion*. To demonstrate this algorithm, we simulated an inverted pendulum driven by an agonist-antagonist pair of motors that pull on tendons with nonlinear elasticity.

<h2 align="center">Installation from GitHub</h2>

Please follow the instructions <a href='https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html'>here</a> in order to install MATLAB engine API for python. Once that is done, you can clone into this repository and install the remaining required packages by copy and pasting the following code into the terminal.

```bash
git clone https://github.com/danhagen/iO-IROS-2020.git && cd iO-IROS-2020/src
pip install -r requirements.txt
pip install .
```

Please note that you can find help for many of the python functions in this repository by using the command `run <func_name> -h`.

<h2 align="center">The Plant</h2> 

<p align="center">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/Schematic_1DOF2DOA_system.png?raw=true"></br>
  <small>Fig. 1: Simple 1 degree of freedom revolute joint actuated by two motors that pull on tendons with nonlinear elasticity.</small>
</p>

Here we used a physical inverted pendulum that was controlled by two simulated brushed DC motors (i.e., backdriveable) that pulled on tendons with nonlinear (exponential) stiffness. This plant can either be given feedfoward inputs or controlled via a *feedback linearization controller* that takes advantage of the fact that joint stiffness and joint angle can be controlled independently. Simply prescribe trajectories for both output measures and the controller will track it.

The default `run plant.py` command will test the feedback linearization algorithm. Choosing the options `--saveFigures` will save the figures and `--animate` will animate the output.


<h2 align="center">Generating Babbling Data</h2>

In order to generate motor babbling data, we use the class `motor_babbling_1DOF2DOA`. The default `run motor_babbling_1DOF2DOA.py` will produce plots of random motor babbling and the resulting states of the plant. Figures will be saved in a time-stamped folder You also have the option to animate the babbling data (`--animate`). 

<p align="center">
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/babblingInputs.png?raw=true"></br>
  <small>Fig. 2: Sample plots of random motor babbling inputs.</small></br>
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/SupplementaryFigures/Plant_States_vs_Time_from_Babbling.png?raw=true"></br>
  <small>Fig. 3: Sample plots of plant states that are the result of random motor babbling inputs.</small>
</p>

<h2 align="center">Train Articifical Neural Networks</h2>

To build, train, and test these ANNs, use `build_NN_1DOF2DOA` and `test_NN_1DOF2DOA`.

<h2 align="center">Run Multiple Trials and Plot All Data</h2>

To run the entire experiment, run `run run_multiple_trials_with_different_babbling_durations.py`. This will sweep across babbling durations (30,45,...,360) seconds, train multiple ANNs (*default*:50 trials), and plot average performances. You can choose to plot metrics such as mean absolute error (MAE), root mean squared error (RMSE), or standard deviation of the error (STD) by adding the additional arguments `-metrics [METRICS ...]`. 

<h2 align="center">Animate a Single Trial (All 4 ANNs Over 4 Different Movements)</h2>

To visualize the performance of ANNs and their ability to generalize to other movement tasks, use the function `animate_sample_trials.py`. This will create an animation of how well each ANN did at predicting joint angle and will sweep across 4 different movements (joint angle and stiffness are either sinusoidal or point-to-point). **Click on the animation below to see more!**


  
 <a href="https://youtu.be/w0AV4tzIW98"><img src="https://user-images.githubusercontent.com/16945786/75698768-24aca280-5c64-11ea-8b74-4999e1bd62b5.gif" alt="Youtube Video Link"></a>
</p>

