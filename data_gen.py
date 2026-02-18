# data_gen.py
import numpy as np
import pandas as pd
import uuid


class DataGen:
    """
    Synthetic Free-Throw Generator (ballistic, no drag).
    """

    def __init__(self, hs=2.13, l=4.115, hKorb=3.048, Rr=0.2286, Rb=0.1219, g=9.807):
        self.params = {
            "l": float(l),
            "hKorb": float(hKorb),
            "Rr": float(Rr),
            "Rb": float(Rb),
            "g": float(g),
        }
        self.hs = float(hs)
        self.wurf_hoehe = 1.25 * self.hs  # release height

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
        return x, y, vx, vy

    def _t_at_height_desc(self, theta, v0, y_target):
        g = self.params["g"]
        y0 = self.wurf_hoehe

        vx = v0 * np.cos(theta)
        vy = v0 * np.sin(theta)
        if vx <= 0:
            return None

        a = 0.5 * g
        b = -vy
        c = y_target - y0

        disc = b * b - 4 * a * c
        if disc <= 0:
            return None

        sqrt_disc = np.sqrt(disc)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        t_desc = max(t1, t2)
        if t_desc <= 0:
            return None
        return float(t_desc)

    def label_from_trajectory(self, theta, v0, margin=0.0):
        l = self.params["l"]
        h = self.params["hKorb"]
        Rr = self.params["Rr"]
        Rb = self.params["Rb"]

        vx = v0 * np.cos(theta)
        r_clear = max(0.0, (Rr - Rb) + margin)

        t_desc = self._t_at_height_desc(theta, v0, h)
        if t_desc is None or vx <= 0:
            return 0

        x_at_hoop_height = vx * t_desc

        if abs(x_at_hoop_height - l) <= r_clear:
            return 1
        elif x_at_hoop_height < l - r_clear:
            return 0
        else:
            return 2

    def generate_dataset_balanced(
        self,
        base_theta_deg=48.43,
        n_per_class=50,
        n_points=50,
        with_noise=False,
        noise_std=0.01,
        dtheta_range=(-0.20, 0.20),
        dv0_range=(-0.8, 0.8),
        margin=0.0,
        max_resample=5000,
        force_exact_endpoint=True,
    ):
        base_theta = np.deg2rad(base_theta_deg)
        base_v0 = self.wurfgeschwindigkeit(base_theta)

        h = self.params["hKorb"]

        target_counts = {0: n_per_class, 1: n_per_class, 2: n_per_class}
        counts = {0: 0, 1: 0, 2: 0}

        rows = []
        wid = 0

        # generate in rounds: 0,1,2,0,1,2,... until all filled
        desired_labels = [0, 1, 2]

        while any(counts[k] < target_counts[k] for k in counts):
            for desired in desired_labels:
                if counts[desired] >= target_counts[desired]:
                    continue

                theta = v0 = T = None

                for _ in range(max_resample):
                    dth = np.random.uniform(*dtheta_range)
                    dv0 = np.random.uniform(*dv0_range)
                    theta_try = base_theta + dth
                    v0_try = base_v0 + dv0

                    t_desc = self._t_at_height_desc(theta_try, v0_try, h)
                    if t_desc is None:
                        continue

                    label_try = self.label_from_trajectory(
                        theta_try, v0_try, margin=margin
                    )
                    if label_try == desired:
                        theta, v0, T = theta_try, v0_try, t_desc
                        label = label_try
                        break

                if T is None:
                    raise RuntimeError(
                        f"Could not sample enough examples for class {desired} "
                        f"within max_resample={max_resample}. "
                        f"Try widening dtheta_range/dv0_range or increasing max_resample."
                    )

                t_vals = np.linspace(0.0, T, n_points, dtype=np.float32)
                x, y, vx, vy = self.wurftrajektorien(theta, v0, t_vals)

                x_clean = x.copy()
                y_clean = y.copy()

                if force_exact_endpoint:
                    vx0 = v0 * np.cos(theta)
                    x_clean[-1] = float(vx0 * T)
                    y_clean[-1] = float(h)

                if with_noise:
                    x_noisy = x_clean + np.random.normal(
                        0.0, noise_std, size=x_clean.shape
                    )
                    y_noisy = y_clean + np.random.normal(
                        0.0, noise_std, size=y_clean.shape
                    )
                    if force_exact_endpoint:
                        x_noisy[-1] = x_clean[-1]
                        y_noisy[-1] = y_clean[-1]
                else:
                    x_noisy = x_clean.copy()
                    y_noisy = y_clean.copy()

                x_sel = x_noisy if with_noise else x_clean
                y_sel = y_noisy if with_noise else y_clean

                for ti, xc, yc, xn, yn, xi, yi in zip(
                    t_vals, x_clean, y_clean, x_noisy, y_noisy, x_sel, y_sel
                ):
                    rows.append(
                        {
                            "wurf_id": wid,
                            "t": float(ti),
                            "x_clean": float(xc),
                            "y_clean": float(yc),
                            "x_noisy": float(xn),
                            "y_noisy": float(yn),
                            "x": float(xi),
                            "y": float(yi),
                            "T": float(T),
                            "label": int(label),
                            "theta": float(theta),
                            "v0": float(v0),
                        }
                    )

                counts[desired] += 1
                wid += 1

        return pd.DataFrame(rows)
