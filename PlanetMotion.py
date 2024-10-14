import numpy as np
import matplotlib.pyplot as plt

plt.style.use("ggplot")

class PlanetMotion(object):
    def __init__(self):
        self.small_m = 1.0
        self.large_m = 1.0
        self.g_const = 1.0
        self.x = 1.0
        self.y = 0.0
        self.vx = 0.0
        self.vy = 1.0
        self.x_list = []
        self.y_list = []

    def eval_force(self, x, y):
        distance = (x ** 2 + y ** 2) ** 0.5
        fx = - (self.large_m * self.g_const / distance ** 3) * x
        fy = - (self.large_m * self.g_const / distance ** 3) * y
        return fx, fy

    def euler_dynamics(self, dt):
        fx, fy = self.eval_force(self.x, self.y)
        self.x_list.append(self.x)
        self.y_list.append(self.y)
        # update velocity
        self.vx += fx / self.small_m * dt
        self.vy += fy / self.small_m * dt
        # update position
        self.x += self.vx * dt
        self.y += self.vy * dt

    def leap_frog_dynamics(self, dt):
        fx, fy = self.eval_force(self.x, self.y)
        self.x_list.append(self.x)
        self.y_list.append(self.y)
        # update velocity
        self.vx += fx / self.small_m * 0.5 * dt
        self.vy += fy / self.small_m * 0.5 * dt
        # update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        fx, fy = self.eval_force(self.x, self.y)
        # update velocity
        self.vx += fx * 0.5 * dt
        self.vy += fy * 0.5 * dt

    def runge_kutta_dynamics(self, dt):
        self.x_list.append(self.x)
        self.y_list.append(self.y)
        k1vx, k1vy = self.eval_force(self.x, self.y)
        k1vx = k1vx / self.small_m * dt
        k1vy = k1vy / self.small_m * dt
        k1x = self.vx * dt
        k1y = self.vy * dt

        k2vx, k2vy = self.eval_force(self.x + 0.5 * k1x, self.y + 0.5 * k1y)
        k2vx = k2vx / self.small_m * dt
        k2vy = k2vy / self.small_m * dt
        k2x = (self.vx + 0.5 * k1vx) * dt
        k2y = (self.vy + 0.5 * k1vy) * dt

        k3vx, k3vy = self.eval_force(self.x + 0.5 * k2x, self.y + 0.5 * k2y)
        k3vx = k3vx / self.small_m * dt
        k3vy = k3vy / self.small_m * dt
        k3x = (self.vx + 0.5 * k2vx) * dt
        k3y = (self.vy + 0.5 * k2vy) * dt

        k4vx, k4vy = self.eval_force(self.x + k3x, self.y + k3y)
        k4vx = k4vx / self.small_m * dt
        k4vy = k4vy / self.small_m * dt
        k4x = (self.vx + k3vx) * dt
        k4y = (self.vy + k3vy) * dt

        self.vx += (k1vx + 2 * k2vx + 2 * k3vx + k4vx) / 6.0
        self.vy += (k1vy + 2 * k2vy + 2 * k3vy + k4vy) / 6.0
        self.x += (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0
        self.y += (k1y + 2 * k2y + 2 * k3y + k4y) / 6.0

    def plot_curvature(self):
        x_arr = np.array(self.x_list)
        y_arr = np.array(self.y_list)
        time_arr = np.arange(len(self.x_list))
        plt.scatter(x_arr, y_arr, c=time_arr, cmap="autumn")
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    system = PlanetMotion()
    num_time_steps = 1000
    dt = 0.05
    for i in range(num_time_steps):
        system.euler_dynamics(dt)
        system.leap_frog_dynamics(dt)
        system.runge_kutta_dynamics(dt)
    system.plot_curvature()
