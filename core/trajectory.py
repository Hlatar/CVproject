import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class Trajectory3D:
    def __init__(self, smoothing_window=5,  max_jump=2.0):
        self.points = []
        self.smoothing_window = smoothing_window  # окно для медианной фильтрации
        self.buffer = []  # буфер точек
        self.max_jump = max_jump

    def clear(self):
        self.points.clear()
        self.buffer.clear()

    def add(self, point):
        """
        Добавляет точку с защитой от выбросов и медианной фильтрацией.
        """

        point = np.array(point, dtype=np.float32)

        if not np.all(np.isfinite(point)):
            return False

        # Отсекаем резкие скачки
        if len(self.points) > 0:
            last = self.points[-1]
            dist = np.linalg.norm(point - last)

            if dist > self.max_jump:
                print(f"Пропущен выброс траектории: jump={dist:.3f} m")
                return False

        self.buffer.append(point)

        if len(self.buffer) >= self.smoothing_window:
            stacked = np.stack(self.buffer)
            filtered = np.median(stacked, axis=0)
            self.points.append(filtered)
            self.buffer.pop(0)
        else:
            self.points.append(point)

        return True

    def __len__(self):
        return len(self.points)

    def get_points(self):
        return np.array(self.points, dtype=np.float32)

    def save_plot(
        self,
        filename="trajectory_3d.png",
        fixed_limits=None,  # например: [(-1, 1), (-1, 1), (0, 2)] для X,Y,Z
        equal_aspect=False,
    ):
        if len(self.points) < 2:
            print("Недостаточно точек для построения 3D-графика.")
            return

        traj = np.array(self.points, dtype=np.float32)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=2, label="Trajectory")
        ax.scatter(
            traj[0, 0], traj[0, 1], traj[0, 2], s=70, marker="o", label="Start"
        )
        ax.scatter(
            traj[-1, 0], traj[-1, 1], traj[-1, 2], s=90, marker="x", label="End"
        )

        # ============ ФИКСИРОВАННЫЙ МАСШТАБ ОСЕЙ ============
        if fixed_limits is not None:
            # Пользовательские пределы
            ax.set_xlim(fixed_limits[0])
            ax.set_ylim(fixed_limits[1])
            ax.set_zlim(fixed_limits[2])
        else:
            # Автоматические, но с запасом 10% от размаха данных
            x_range = traj[:, 0].max() - traj[:, 0].min()
            y_range = traj[:, 1].max() - traj[:, 1].min()
            z_range = traj[:, 2].max() - traj[:, 2].min()

            max_range = max(x_range, y_range, z_range) * 0.5

            mid_x = (traj[:, 0].max() + traj[:, 0].min()) * 0.5
            mid_y = (traj[:, 1].max() + traj[:, 1].min()) * 0.5
            mid_z = (traj[:, 2].max() + traj[:, 2].min()) * 0.5

            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)

        # ============ ОДИНАКОВЫЙ МАСШТАБ ОСЕЙ ============
        if equal_aspect:
            # Делаем кубическое отображение
            ax.set_box_aspect([1, 1, 1])

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("3D trajectory of tracked object")
        ax.legend()

        # Добавляем сетку для лучшего восприятия масштаба
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filename, dpi=200)
        plt.close(fig)

        print(f"3D график сохранён в {filename}")