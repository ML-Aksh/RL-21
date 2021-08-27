# RL-TensorLab 2021

Modern, real-life reinforcement learning environments tend to be noise and uncertainty prone. However, current state-of-the-art MPC planning schemes fail to account for such uncertainty while planning. In this project, we set out to explore whether incorporating system stochasticity while planning trajectories could improve the overall performance of an algorithm. Secondly, in case improvements were observed, we wanted to determine if noise could be modeled in addition to the raw environment and used to guide planning. 

We model 4 independent systems relying on varying optimization schemes. More loosely the four schemes can be captured as follows:
<ol> 
  <li> **Determinsitic Model + Determinstic Planner: ** Dynamics model learned from deterministic samples & deterministic planner
  <li> **Deterministic Model + Stochastic Planner: ** Dynamics model learned from deterministic samples & stochastic planner
  <li> **Stochastic Model + Determinstic Planner: ** Dynamics model learned from stochastic samples & deterministic planner
  <li> **Stochastic Ensemble + Deterministic Planner: ** Dynamics ensemble (combination of n Multi Layer Perceptrons) learned from deterministic samples & deterministic planner
</ol>

# Script Summary
Summary of scripts contained contained within this folder:
<ol>
  
</ol>
