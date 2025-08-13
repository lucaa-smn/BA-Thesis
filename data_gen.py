# data_gen.py
import numpy as np
import pandas as pd
import uuid


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

    def generate_dataset(self, base_theta_deg=48.43, n=100, with_noise=False):
        """
        Generiert einen Datensatz aus drei Klassen von Würfen.
        Optional kann Rauschen hinzugefügt werden.
        """
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

        if with_noise:
            df = self.full_noise_pipeline(df)

        return df

    def _generate_wurf(self, theta_base, v0_base, variation_range, label, n):
        """
        Erzeugt n Würfe mit zufälligen Variationen um theta_base und v0_base.
        """
        data = []
        g = self.params["g"]
        for _ in range(n):
            # Variiere die Startparameter leicht um die Basiseinstellungen
            theta = theta_base + np.random.uniform(*variation_range["theta"])
            v0 = v0_base + np.random.uniform(*variation_range["v0"])

            # Flugzeit und Zeitstempel berechnen
            T = 2 * v0 * np.sin(theta) / g
            t_vals = np.linspace(0, T, 50)

            # Wurftrajektorie berechnen
            x, y = self.wurftrajektorien(theta, v0, t_vals)

            # Eine eindeutige ID für diesen Wurf erzeugen
            wurf_id = str(uuid.uuid4())

            # Alle 50 Zeitpunkte als Datenpunkte abspeichern
            for xi, yi in zip(x, y):
                data.append(
                    {
                        "x": xi,
                        "y": yi,
                        "theta": theta,
                        "v0": v0,
                        "label": label,
                        "wurf_id": wurf_id,
                    }
                )

        return data

    def add_position_noise(self, df, std_x=0.01, std_y=0.01):
        """Fügt Rauschen zu den x- und y-Koordinaten hinzu."""
        df_noisy = df.copy()
        df_noisy["x"] += np.random.normal(0, std_x, size=len(df))
        df_noisy["y"] += np.random.normal(0, std_y, size=len(df))
        return df_noisy

    def add_initial_param_noise(self, df, std_theta=0.005, std_v0=0.05):
        """Fügt Rauschen zu theta und v0 hinzu."""
        df_noisy = df.copy()
        df_noisy["theta"] += np.random.normal(0, std_theta, size=len(df))
        df_noisy["v0"] += np.random.normal(0, std_v0, size=len(df))
        return df_noisy

    def dropout_features(self, df, dropout_rate=0.01):
        """Setzt zufällig Werte in x und y auf NaN (Dropout-Simulation)."""
        df_noisy = df.copy()
        mask = np.random.rand(*df[["x", "y"]].shape) < dropout_rate
        df_noisy[["x", "y"]] = df[["x", "y"]].mask(mask)
        return df_noisy

    def full_noise_pipeline(self, df):
        """
        Führt mehrere Rauschmethoden nacheinander aus.
        """
        df = self.add_position_noise(df, std_x=0.1, std_y=0.1)
        df = self.add_initial_param_noise(df, std_theta=0.1, std_v0=0.1)
        # Optional:
        # df = self.dropout_features(df, dropout_rate=0.005)
        return df
