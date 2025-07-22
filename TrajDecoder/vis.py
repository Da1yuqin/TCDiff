import os
from pathlib import Path
from tempfile import TemporaryDirectory

import librosa as lr
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from matplotlib import cm
from matplotlib.colors import ListedColormap
from pytorch3d.transforms import (axis_angle_to_quaternion, quaternion_apply,
                                  quaternion_multiply)
from tqdm import tqdm
from p_tqdm import p_map
import pickle

smpl_joints = [
    "root",  # 0
    "lhip",  # 1
    "rhip",  # 2
    "belly", # 3
    "lknee", # 4
    "rknee", # 5
    "spine", # 6
    "lankle",# 7
    "rankle",# 8
    "chest", # 9
    "ltoes", # 10
    "rtoes", # 11
    "neck",  # 12
    "linshoulder", # 13
    "rinshoulder", # 14
    "head", # 15
    "lshoulder", # 16
    "rshoulder",  # 17
    "lelbow", # 18
    "relbow",  # 19
    "lwrist", # 20
    "rwrist", # 21
    "lhand", # 22
    "rhand", # 23
]

smpl_parents = [
    -1, # root
    0, # connect with root
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    21,
]

smpl_offsets = [
    [0.0, 0.0, 0.0],
    [0.05858135, -0.08228004, -0.01766408],
    [-0.06030973, -0.09051332, -0.01354254],
    [0.00443945, 0.12440352, -0.03838522],
    [0.04345142, -0.38646945, 0.008037],
    [-0.04325663, -0.38368791, -0.00484304],
    [0.00448844, 0.1379564, 0.02682033],
    [-0.01479032, -0.42687458, -0.037428],
    [0.01905555, -0.4200455, -0.03456167],
    [-0.00226458, 0.05603239, 0.00285505],
    [0.04105436, -0.06028581, 0.12204243],
    [-0.03483987, -0.06210566, 0.13032329],
    [-0.0133902, 0.21163553, -0.03346758],
    [0.07170245, 0.11399969, -0.01889817],
    [-0.08295366, 0.11247234, -0.02370739],
    [0.01011321, 0.08893734, 0.05040987],
    [0.12292141, 0.04520509, -0.019046],
    [-0.11322832, 0.04685326, -0.00847207],
    [0.2553319, -0.01564902, -0.02294649],
    [-0.26012748, -0.01436928, -0.03126873],
    [0.26570925, 0.01269811, -0.00737473],
    [-0.26910836, 0.00679372, -0.00602676],
    [0.08669055, -0.01063603, -0.01559429],
    [-0.0887537, -0.00865157, -0.01010708],
]


def set_line_data_3d(line, x): 
    if len(x.shape) == 2: 
        line.set_data(x[:, :2].T)
        line.set_3d_properties(x[:, 2])
    else: # multi dancer
        for i in range(x.shape[0]):
            line[i].set_data(x[i,:,:2].T) # (2,2)
            line[i].set_3d_properties(x[i,:,2])



def set_scatter_data_3d(scat, x, c): 
    if len(x.shape) == 3: # multi dancer: (dancer_num,1,3),scat [dancer_num], color [dancer_num]
        dancer_num = x.shape[0]
        for i in range(dancer_num):
            scat[i].set_offsets(x[i, :, :2]) # singel person (1,3) 
            scat[i].set_3d_properties(x[i, :, 2], "z")
            scat[i].set_facecolors([c[i]])
    else:
        scat.set_offsets(x[:, :2]) # singel person (1,3) 
        scat.set_3d_properties(x[:, 2], "z")
        scat.set_facecolors([c])
    


def get_axrange(poses):
    pose = poses[0]
    x_min = pose[:, 0].min()
    x_max = pose[:, 0].max()

    y_min = pose[:, 1].min()
    y_max = pose[:, 1].max()

    z_min = pose[:, 2].min()
    z_max = pose[:, 2].max()

    xdiff = x_max - x_min
    ydiff = y_max - y_min
    zdiff = z_max - z_min

    biggestdiff = max([xdiff, ydiff, zdiff])
    return biggestdiff

def plot_multi_pose(num, poses, lines, ax, axrange, scat): 
    pose = poses[:,num] # Shape: (n, J, 3), where n = number of dancers
    dancer_num = poses.shape[0]  # Total number of dancers

    for i, p in enumerate(smpl_parents): # plot lines
        # Skip plotting the root joint
        if i == 0: 
            if num > 1 : 
                # Use this empty line to draw dancer trajectories;
                # requires at least 2 frames of data
                for line_id in range(dancer_num):
                    lines[line_id * 24].set_data(poses[line_id,:num, 0, :2].T) # (2,2) # Plot trajectory in XY plane
                    lines[line_id * 24].set_3d_properties(0) # Set Z value for trajectory line.
                    # Must provide a Z value; otherwise, the line won't appear.
                    # Note: Z = -1.5 overlaps the character; Z = 2.5 places the line above the character.
            continue
        break

    if num == 0:        # Set the axis limits for zooming the plot
        if isinstance(axrange, int):
            axrange = (axrange, axrange, axrange)
        # Center the axes
        # AIOZ-Dataset setting: characters mainly face the negative Y axis,
        # so Y center is shifted to 2.5
        xcenter, ycenter, zcenter = 0, 2.5, 2.5 
        stepx, stepy, stepz = axrange[0] / 2, axrange[1] / 2, axrange[2] / 2

        x_min, x_max = xcenter - stepx, xcenter + stepx
        y_min, y_max = ycenter - stepy, ycenter + stepy
        z_min, z_max = zcenter - stepz, zcenter + stepz

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

    return lines, scat



def skeleton_render(
    poses,
    epoch=0,
    out="renders",
    name="",
    sound=True,
    stitch=False,
    sound_folder="ood_sliced",
    contact=None,
    render=True,
):
    dancer_num = 1
    multi_dancer = False

    if len(poses.shape) == 4: # multi_dancer
        multi_dancer = True
        dancer_num = poses.shape[0]


    if render:
        # generate the pose with FK
        Path(out).mkdir(parents=True, exist_ok=True)
        num_steps = poses.shape[1] if multi_dancer else poses.shape[0] # 150
        
        fig = plt.figure(figsize=(8, 8))  
        ax = fig.add_subplot(projection="3d")
        
        point = np.array([0, 0, 1])
        normal = np.array([0, 0, 1])
        d = -point.dot(normal)
        xx, yy = np.meshgrid(np.linspace(-4, 4, 2), np.linspace(-4, 4, 2)) 
        z = (-normal[0] * xx - normal[1] * yy - d) * 1.0 / normal[2]
 

        # Camera view
        # ax.view_init(elev=90, azim=-90) # Front view (AIOZ-dataset) - rotate 90° to match AIOZ-GDance dataset
        # ax.view_init(elev=50, azim=-90) # Top-side view (AIOZ-dataset) - shows both stance and motion
        # ax.view_init(elev=40, azim=90)  # Test view - characters face left
        # ax.view_init(elev=40, azim=-90)  # Set camera angle
        ax.view_init(elev=90, azim=-90) # Top-down view 


        # # Create lines initially without data
        lines = [
            ax.plot([], [], [], zorder=10, linewidth=3.5)[0]
            for _ in smpl_parents*dancer_num 
        ] 
        scat = [
            ax.scatter([], [], [], zorder=10, s=0, cmap=ListedColormap(["r", "g", "b"])) 
            for _ in range(4*dancer_num) # dots per frame
        ] 

        axrange = 4 # Adjust the plotting range; the larger the range, the smaller the figure of the person appears

        # Adjust the figure layout to remove extra whitespace
        fig.tight_layout()

        # Select plotting function based on number of dancers
        if multi_dancer:
            plot_pose = plot_multi_pose
        else:
            plot_pose = plot_single_pose # Not implemented yet

        # Create the animation object to render poses frame by frame
        anim = animation.FuncAnimation( 
            fig,
            func = plot_pose,       # The update function for each frame
            frames = num_steps,     # Total number of frames
            fargs=(poses, lines, ax, axrange, scat),  # Pose data and plotting objects
            interval=1000 // 30,    # Frame interval for ~30 FPS
        )

        # Adjust layout again to ensure tight fit
        fig.tight_layout()

    if sound:
        # Create a temporary directory to save intermediate GIF files
        if render:
            temp_dir = TemporaryDirectory()
            gifname = os.path.join(temp_dir.name, f"{epoch}.gif")
            anim.save(gifname) # Temporarily save the animation as a video or gif

        # Stitch multiple audio segments into one file
        # Using a sliding window where each segment overlaps by half its length
        if stitch:
            assert type(name) == list  # must be a list of names to do stitching
            name_ = [os.path.splitext(x)[0] + ".wav" for x in name]
            audio, sr = lr.load(name_[0], sr=None)
            ll, half = len(audio), len(audio) // 2
            total_wav = np.zeros(ll + half * (len(name_) - 1))
            total_wav[:ll] = audio
            idx = ll
            for n_ in name_[1:]:
                audio, sr = lr.load(n_, sr=None)
                total_wav[idx : idx + half] = audio[half:]
                idx += half
            # Save a dummy spliced audio file
            audioname = f"{temp_dir.name}/tempsound.wav" if render else os.path.join(out, f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.wav')
            sf.write(audioname, total_wav, sr)
            outname = os.path.join(
                out,
                f'{epoch}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.mp4', # 把sliced 的信息省略了
            )
        else:
            assert type(name) == str
            assert name != "", "Must provide an audio filename"
            audioname = name
            outname = os.path.join(
                out, f"{epoch}_{os.path.splitext(os.path.basename(name))[0]}.mp4"
            )
        if render:
            out = os.system(
                f"/usr/bin/ffmpeg -loglevel error -stream_loop 0 -y -i {gifname} -i {audioname} -shortest -c:v libx264 -c:a libmp3lame -q:a 4 {outname}" # libx264, MP3 decoder used for audio
            )
    else:
        if render:
            # actually save the gif
            path = os.path.normpath(name)
            pathparts = path.split(os.sep)
            gifname = os.path.join(out, f"{pathparts[-1][:-4]}.gif")
            anim.save(gifname, savefig_kwargs={"transparent": True, "facecolor": "none"},)
    plt.close()

def render_sample( 
    samples, 
    epoch,
    render_out,
    fk_out=None,
    name=None,
    sound=True,
    mode="normal",
    noise=None,
    constraint=None,
    sound_folder="ood_sliced",
    start_point=None,
    render=True,
    required_dancer_num = 3,
    x_0 = None,
):
    
    b, s, ds, c = samples.shape
    pos = samples  

    b = pos.shape[0] 
    poses = pos.reshape(b,-1,required_dancer_num,3).detach().cpu().numpy()
    poses = np.transpose(poses[:,:,:,None,:],(0,2,1,3,4))

    def inner(xx):
        num, pose = xx
        filename = name[num] if name is not None else None
        skeleton_render(
            pose,
            epoch=f"e{epoch}_b{num}",
            out=render_out,
            name=filename,
            sound=sound,
        )

    p_map(inner, enumerate(poses)) 

    if fk_out is not None and mode != "long": 
        Path(fk_out).mkdir(parents=True, exist_ok=True)
        for num, (pos_, filename) in enumerate(zip(pos, name)):
            path = os.path.normpath(filename)
            pathparts = path.split(os.sep)
            pathparts[-1] = pathparts[-1].replace("npy", "wav")
            # path is like "data/train/features/name"
            pathparts[2] = "wav_sliced"
            audioname = os.path.join(*pathparts)
            outname = f"{epoch}_{num}_{pathparts[-1][:-4]}.pkl"
            pickle.dump(
                {
                    "smpl_trans": pos_.cpu().numpy(),
                },
                open(f"{fk_out}/{outname}", "wb"),
            )


class SMPLSkeleton:
    def __init__(
        self, device=None,
    ):
        offsets = smpl_offsets
        parents = smpl_parents
        assert len(offsets) == len(parents)

        self._offsets = torch.Tensor(offsets).to(device)
        self._parents = np.array(parents)
        self._compute_metadata()

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)

    def forward(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 3) tensor of axis-angle rotations describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert len(root_positions.shape) == 3
        # transform from axis angle to quaternion
        rotations = axis_angle_to_quaternion(rotations)

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(
            rotations.shape[0],
            rotations.shape[1],
            self._offsets.shape[0],
            self._offsets.shape[1],
        )

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(
                    quaternion_apply(
                        rotations_world[self._parents[i]], expanded_offsets[:, :, i]
                    )
                    + positions_world[self._parents[i]]
                )
                if self._has_children[i]:
                    rotations_world.append(
                        quaternion_multiply(
                            rotations_world[self._parents[i]], rotations[:, :, i]
                        )
                    )
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)
