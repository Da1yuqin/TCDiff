# Harmonious Group Choreography with Trajectory-Controllable Diffusion
This is the official code for our paper: "Harmonious Group Choreography with Trajectory-Controllable Diffusion". It is a novel approach that harnesses non-overlapping trajectories to facilitate coherent dance movements. 

The code is being released soon!

[<a href="https://wanluzhu.github.io/TCDiffusion/"><strong>Project Page</strong></a>]

![model](Fig/Pipline.jpg)
Our framework consists of two main components: the Dance-Beat Navigator (DBN) and Trajectory-Controllable Diffusion (TCDiff). 
To address dancer ambiguity, initially, we employ DBN to model dancer positions, as dancers' coordinates exhibit distinct differences and are less prone to confusion.
Subsequently, TCDiff utilizes this result for conditional diffusion to generate corresponding dance movements. During this process, a fusion projection enhances group information before inputting it into the multi-dance transformer, while a footwork adaptor adjusts the final footwork.



# Citation
```
Dai, Y., Zhu, W., Li, R., Ren, Z., Zhou, X., Li, X., ... & Yang, J. (2024). Harmonious Group Choreography with Trajectory-Controllable Diffusion. arXiv preprint arXiv:2403.06189.
```
