import numpy as np
import matplotlib.pyplot as plt

plt.set_loglevel("warning")

CUBE_COLORS = ["red", "blue", "green", "orange"]
CUBE_CMAPS = ["Reds", "Blues", "Greens", "Oranges"]


def make_trajectory_images(env_id, trajectory, batch_size, start, goal, plot_end_points=True):
    """3D plots of cube trajectories (t, b, 3 * n), one image per batch item; start (P) and goal (X) per cube."""
    images = []
    num_cubes = trajectory.shape[-1] // 3
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.set_facecolor("gray")
        ax.view_init(elev=20, azim=30)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        for i in range(num_cubes):
            ax.scatter(trajectory[:, batch_idx, i*3], trajectory[:, batch_idx, i*3+1], trajectory[:, batch_idx, i*3+2],
                       c=np.arange(len(trajectory)), cmap=CUBE_CMAPS[i])
        if plot_end_points:
            for points, marker in ((start, "P"), (goal, "X")):
                points = np.array(points[batch_idx])
                for i in range(num_cubes):
                    ax.scatter(points[i*3], points[i*3+1], points[i*3+2], c=CUBE_COLORS[i], s=100, marker=marker)
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        images.append(np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy().reshape(img_shape))
        plt.close()
    return images
