# RL-TensorLab 2021

Modern, real-life reinforcement learning environments tend to be noise and uncertainty prone. However, current state-of-the-art MPC planning schemes fail to account for such uncertainty while planning. In this project, we set out to explore whether incorporating system stochasticity while planning trajectories could improve the overall performance of an algorithm. Secondly, in case improvements were observed, we wanted to determine if noise could be modeled in addition to the raw environment and used to guide planning. 

**General Framework**
<ol>
  <li> Collect 2000 samples through a random agent running from either a determinstic/stochastic environment & store these in a replay buffer
  <li> Train our dynamics model on these samples & use it for computing an ideal trajectory. We plan using an MPC and use CEM for creating and evaluating action sequences. If planning stochastically sample noise from a distribution and add it. 
  <li> Execute the first step from this sequence on a stochastic environment + record returns
    
</ol>

**Algorithmic Sub Branches**
We model 4 independent systems relying on varying optimization schemes. More loosely the four schemes can be captured as follows:
<ol> 
  <li> <b> Determinsitic Model + Determinstic Planner: </b> Dynamics model learned from deterministic samples & deterministic planner
  <li> <b> Deterministic Model + Stochastic Planner: </b> Dynamics model learned from deterministic samples & stochastic planner
  <li> <b> Stochastic Model + Determinstic Planner: </b> Dynamics model learned from stochastic samples & deterministic planner
  <li> <b> Stochastic Ensemble + Deterministic Planner: </b> Dynamics ensemble (combination of n Multi Layer Perceptrons) learned from deterministic samples & deterministic planner
</ol>

![Algorithmic Overview](https://user-images.githubusercontent.com/41270824/131066039-f1fb345b-c112-471b-855b-13710ad380f5.png)

# Results
In general stochastic planning with a deterministic MLP seems to perform really well, attaining significantly higher performance than the doubly deterministic scheme (scenario 1) and scenario 3, which uses a single MLP and then plans deterministically. 

<ol>
  <li>
    We do note, however, that for low noise levels the Stochastic Ensemble outperforms the proposed scheme, attaining rewards roughly 12, 10, and 13 points higher for the 0.01, 0.015, and 0.020 noise levels, respectively. 
  </li>
  <li>
    We also notice that using a stochastic model with a single MLP is ineffective and in several cases hurts performance. Particularly, for the three lowest noise levels, 0.010, 0.015, and 0.020, it performs worse than even the doubly deterministic mechanism in scenario 1. This makes sense as with a single model representing noise the dynamic model repeatedly biases results in the same direction.
  </li>
</ol>
  


![Results 1](https://user-images.githubusercontent.com/41270824/131067007-cbb8ee57-e904-4a6f-a076-223425727945.png)



We can also examine the percentage-wise improvements from the proposed scheme. Each cell represents the percentage by which the proposed method improved performance. The 3 bolded negative numbers in the rightmost column correspond to cases where the proposed algorithm lowered performance. Looking at the table we note:
<ol>
  <li> 
    The best improvements from the proposed plan were observed at a noise level of 0.030, where scenario 2 provides 121.78%, 102.96%, and 54.601% increases over scenarios 1, 3, and 4 respectively.
  </li>
  <li>
    On the other hand, for the worst-performing case at a noise level of 0.020, our proposed method has 69.79% and 83.29% improvements over scenarios 1 and 3, while a minimal loss of 7% compared to the stochastic ensembling method (scenario 4).
  </li>
  <li>
    In general, the proposed scheme seems to be the most stable and consistent in boosting performance, but if environmental noise is known to be low, the ensembling approach may work better.
  </li>
</ol>


![Results 2](https://user-images.githubusercontent.com/41270824/131067013-de18e408-160e-46b0-b6d6-343a41230197.png)

These trends are perhaps easiest to note in these graphs where in cases where the ensemble outperforms the stochastic planning, the results are neck and neck. However, when stochastic planning performs better, it does so by a far larger margin.

![Results 3](https://user-images.githubusercontent.com/41270824/131067030-1d62e5a4-8c8c-4f2b-9db0-79c32b14c93c.PNG)


# Script Summary
I modified the traditional Cartpole, Pendulum, and Mountain Car environments present in the mbrl and gym libraries. Some of the major changes include custom-engineered reward functions, addition of stochasticity within otherwise deterministic environments, use of random agents for sample collection, and choices for dynamics models. Here is a summary of the additional scripts contained contained within this folder:
<ol>
  <li>
    <b> Utils_noise.py: </b> Contains functions for the different algorithmic choices (1-4) we consider.
  </li>
  <li>
    <b> Cartpole - Clean.py: </b> General script for launching functions defined in Utils_noise.py
  </li>
  <li>
    <b> cartpole_continuous.py: </b> A continuous analog to the gym cartpole environment with options to toggle noise on/off
  </li>
  <li>
    <b> cartpole_continuous_noiseless.py: </b> Conntious version of gym cartpole environment without any noise.
  </li>
  <li>
    <b> Mountain.py: </b> Script for launching analysis on Mountain-Car environment
  </li>
  <li>
    <b> Pendulum.py: </b> Script for launching analysis on the Pendulum environment
  </li>
  
</ol>
