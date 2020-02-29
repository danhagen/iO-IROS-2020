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
  <img width="500" src="https://github.com/danhagen/iO-IROS-2020/blob/master/img/Schematic_1DOF2DOA_system.png?raw=true">
</p>

Here we used a physical inverted pendulum that was controlled by two simulated brushed DC motors (i.e., backdriveable) that pulled on tendons with nonlinear (exponential) stiffness. This plant can either be given feedfoward inputs or controlled via a *feedback linearization controller* that takes advantage of the fact that joint stiffness and joint angle can be controlled independently. Simply prescribe trajectories for both output measures and the controller will track it.

The default `run plant.py` command will test the feedback linearization algorithm. Choosing the options `--saveFigures` will save the figures and `--animate` will animate the output.


## Generating Babbling Data
In order to generate motor babbling data, we use the class `motor_babbling_1DOF2DOA`. Running the function from the command line, you can alter the following optional arguments. 

```
usage: <filename> [-h] [-dt [timestep]] [-dur [duration]] [--savefigs]
                  [--savefigsPDF] [--savedata] [--animate]

-----------------------------------------------------------------------------

motor_babbling_1DOF2DOA.py

-----------------------------------------------------------------------------

Motor babbling algorithm for a 1 DOF, 2 DOA tendon-driven system with
nonlinear tendon elasticity. Low frequency white noise is added to
random step changes to the input levels (within some bounds).

-----------------------------------------------------------------------------

optional arguments:
  -h, --help       show this help message and exit
  -dt [timestep]   Time step for the simulation (float). Default is given by
                   plantParams.py
  -dur [duration]  Duration of the simulation (float). Default is given by
                   plantParams.py
  --savefigs       Option to save figures for babbling trial. Default is
                   false.
  --savefigsPDF    Option to save figures for babbling trial as a PDF. Default
                   is false.
  --savedata       Option to save data for babbling trial as a Matlab .MAT
                   file. Default is false.
  --animate        Option to animate trial. Default is false.

-----------------------------------------------------------------------------
```
You have the option to animate the babbling data or just simply save the output figures.



