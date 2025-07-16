# data_gen.py
import numpy as np
import pandas as pd


class DataGen:
    def __init__(self, hs=2.13, l=4.115, hKorb=3.048, Rr=0.2286, Rb=0.1219, g=9.807):
        self.params = {
            "l": l,
            "hKorb": hKorb,
            "Rr": Rr,
            "Rb": Rb,
            "g": g,
        }
        self.hs = hs
        self.wurf_hoehe = 1.25 * hs

    def wurfgeschwindigkeit(self, theta):
        l = self.params["l"]
        h = self.params["hKorb"] - self.wurf_hoehe
        g = self.params["g"]
        return (l / np.cos(theta)) * np.sqrt(g / (2 * (l * np.tan(theta) - h)))

    def wurftrajektorien(self, theta, v0, t):
        g = self.params["g"]
        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        x = vx * t
        y = vy * t - 0.5 * g * t**2 + self.wurf_hoehe
        return x, y

    def generate_dataset(self, base_theta_deg=48.43, n=100):
        base_theta = np.deg2rad(base_theta_deg)
        base_v0 = self.wurfgeschwindigkeit(base_theta)

        categories = {
            0: {"theta": (-0.15, -0.05), "v0": (-0.5, -0.2)},
            1: {"theta": (-0.03, 0.03), "v0": (-0.1, 0.1)},
            2: {"theta": (0.06, 0.15), "v0": (0.3, 0.5)},
        }

        all_data = []
        for label, variation in categories.items():
            all_data.extend(
                self._generate_wurf(base_theta, base_v0, variation, label, n)
            )

        df = pd.DataFrame(all_data)
        return df

    def _generate_wurf(self, theta_base, v0_base, variation_range, label, n):
        data = []
        g = self.params["g"]
        for _ in range(n):
            theta = theta_base + np.random.uniform(*variation_range["theta"])
            v0 = v0_base + np.random.uniform(*variation_range["v0"])
            T = 2 * v0 * np.sin(theta) / g
            t_vals = np.linspace(0, T, 50)
            x, y = self.wurftrajektorien(theta, v0, t_vals)
            for xi, yi in zip(x, y):
                data.append(
                    {"x": xi, "y": yi, "theta": theta, "v0": v0, "label": label}
                )
        return data
