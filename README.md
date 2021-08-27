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

# Script Summary
Summary of scripts contained contained within this folder:
<ol>
  
</ol>
