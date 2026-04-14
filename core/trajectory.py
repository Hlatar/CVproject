import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Trajectory3D:
    def __init__(self):
        self.points = []

    def clear(self):
        self.points.clear()

    def add(self, point):
        self.points.append(np.array(point, dtype=np.float32))

    def __len__(self):
        return len(self.points)

    def save_plot(self, filename="trajectory_3d.png"):
        if len(self.points) < 2:
            print("Недостаточно точек для построения 3D-графика.")
            return

        traj = np.array(self.points, dtype=np.float32)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, label="Trajectory")
        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=70, marker="o", label="Start")
        ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], s=90, marker="x", label="End")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("3D trajectory of tracked object")
        ax.legend()

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)

        print(f"3D график сохранён в {filename}")