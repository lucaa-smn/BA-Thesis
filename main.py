import pandas as pd
import plotly.express as px
from data_gen import DataGen

gen = DataGen()  # Standard-Parameter (Korbentfernung, Korbhöhe, etc.)

df = gen.generate_dataset_balanced(
    base_theta_deg=48.43,
    n_per_class=50,  # pro Klasse (0/1/2) so viele Würfe
    n_points=50,  # Zeitpunkte pro Wurf
    with_noise=True,  # noisy x/y zusätzlich
    noise_std=0.01,
)
