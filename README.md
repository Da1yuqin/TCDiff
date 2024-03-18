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
@article{dai2024harmonious,
  title={Harmonious Group Choreography with Trajectory-Controllable Diffusion},
  author={Dai, Yuqin and Zhu, Wanlu and Li, Ronghui and Ren, Zeping and Zhou, Xiangzheng and Li, Xiu and Li, Jun and Yang, Jian},
  journal={arXiv preprint arXiv:2403.06189},
  year={2024}
}
```
