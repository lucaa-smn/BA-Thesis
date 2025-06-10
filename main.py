# Python-Umsetzung des ersten Modells aus dem Dokument
# "Modellierung des Freiwurfs beim Basketball"

import numpy as np
import plotly.graph_objects as go
from scipy.optimize import fmin

# Parameter
params = {
    "l": 4.115,  # horizontaler Abstand in m
    "hKorb": 3.048,  # Korbhöhe in m
    "Rr": 0.2286,  # Radius des Rings in m
    "Rb": 0.1219,  # Radius des Balls in m
    "g": 9.807,  # Erdbeschleunigung in m/s^2
}


def wurfgeschwindigkeit(theta, hS):
    l, h, g = params["l"], params["hKorb"] - 1.25 * hS, params["g"]
    return (l / np.cos(theta)) * np.sqrt(g / (2 * (l * np.tan(theta) - h)))


def wurftrajektorien(theta0, v0, t, y_offset):
    vx = v0 * np.cos(theta0)
    vy = v0 * np.sin(theta0)
    x = vx * t
    y = vy * t - 0.5 * params["g"] * t**2 + y_offset
    return x, y


def abstand_zum_ring(x, y):
    dx = x - (params["l"] - params["Rr"])
    dy = y - (params["hKorb"])
    return np.sqrt(dx**2 + dy**2)


def wurf_erlaubt(theta, v0, y_offset):
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    T = (
        vy + np.sqrt(vy**2 - 2 * params["g"] * (params["hKorb"] - y_offset))
    ) / params["g"]
    t0 = (params["l"] - params["Rr"] - params["Rb"]) / vx
    t_vals = np.linspace(t0, T, 100)
    x, y = wurftrajektorien(theta, v0, t_vals, y_offset)
    xT = x[-1]
    a_min = np.min(abstand_zum_ring(x, y))
    if xT + params["Rb"] > params["l"] + params["Rr"]:
        return False
    if xT - params["Rb"] < params["l"] - params["Rr"]:
        return False
    if a_min < params["Rb"]:
        return False
    return True


def fehlerwinkel(theta0, hS):
    y_offset = 1.25 * hS
    v0 = wurfgeschwindigkeit(theta0, hS)
    d_theta = np.deg2rad(0.5)
    theta_min = theta0
    while wurf_erlaubt(theta_min - d_theta, v0, y_offset):
        theta_min -= d_theta
    theta_max = theta0
    while wurf_erlaubt(theta_max + d_theta, v0, y_offset):
        theta_max += d_theta
    return min(theta0 - theta_min, theta_max - theta0)


def optimaler_winkel(hS):
    def zu_minimieren(theta):
        return -fehlerwinkel(np.deg2rad(theta[0]), hS)

    res = fmin(zu_minimieren, [45], disp=False)
    return res[0]


# Beispiel: Beste Wurfwinkel für Spielergröße 2.13 m
hs = 2.13
y_offset = 1.25 * hs
opt_winkel = optimaler_winkel(hs)
v_opt = wurfgeschwindigkeit(np.deg2rad(opt_winkel), hs)
print(f"Optimaler Winkel: {opt_winkel:.2f} Grad, v0: {v_opt:.2f} m/s")

# Visualisierung mit Plotly
T = params["l"] / (v_opt * np.cos(np.deg2rad(opt_winkel)))
t_vals = np.linspace(0, T, 500)
x, y = wurftrajektorien(np.deg2rad(opt_winkel), v_opt, t_vals, y_offset)

fig = go.Figure()
fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Wurfbahn"))

# Ring als dicke horizontale Linie
fig.add_shape(
    type="line",
    x0=params["l"] - params["Rr"],
    x1=params["l"] + params["Rr"],
    y0=params["hKorb"],
    y1=params["hKorb"],
    line=dict(color="orange", width=6),
)

fig.add_shape(
    type="line",
    x0=0,
    x1=max(x),
    y0=params["hKorb"],
    y1=params["hKorb"],
    line=dict(color="gray", dash="dash"),
)
fig.add_shape(
    type="line",
    x0=params["l"],
    x1=params["l"],
    y0=0,
    y1=max(y),
    line=dict(color="gray", dash="dash"),
)
fig.update_layout(
    title=f"Optimale Wurfbahn bei {opt_winkel:.2f}°",
    xaxis_title="x [m]",
    yaxis_title="y [m]",
    showlegend=True,
)
fig.show()
