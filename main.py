import numpy as np
import pandas as pd
import plotly.express as px


# Parameter
params = {
    "l": 4.115,  # horizontaler Abstand in m
    "hKorb": 3.048,  # Korbhöhe in m
    "Rr": 0.2286,  # Radius des Rings in m
    "Rb": 0.1219,  # Radius des Balls in m
    "g": 9.807,  # Erdbeschleunigung in m/s^2
}

# Spielergröße und Abwurfhöhe
hs = 2.13
wurf_hoehe = 1.25 * hs


def wurfgeschwindigkeit(theta):
    l, h, g = params["l"], params["hKorb"] - wurf_hoehe, params["g"]
    return (l / np.cos(theta)) * np.sqrt(g / (2 * (l * np.tan(theta) - h)))


def wurftrajektorien(theta, v0, t):
    vx = v0 * np.cos(theta)
    vy = v0 * np.sin(theta)
    x = vx * t
    y = vy * t - 0.5 * params["g"] * t**2 + wurf_hoehe
    return x, y


def wurf_klassifizieren(x, y):
    ziel_x = params["l"]
    ziel_y = params["hKorb"]
    idx = np.argmin(np.abs(y - ziel_y))
    xT = x[idx]
    if xT < ziel_x - params["Rr"]:
        return "short"
    elif xT > ziel_x + params["Rr"]:
        return "long"
    else:
        return "hit"


def generate_wurf(theta_base, v0_base, variation_range, label, n):
    data = []
    for _ in range(n):
        theta = theta_base + np.random.uniform(*variation_range["theta"])
        v0 = v0_base + np.random.uniform(*variation_range["v0"])
        T = 2 * v0 * np.sin(theta) / params["g"]
        t_vals = np.linspace(0, T, 50)
        x, y = wurftrajektorien(theta, v0, t_vals)
        for xi, yi in zip(x, y):
            data.append({"x": xi, "y": yi, "theta": theta, "v0": v0, "label": label})
    return data


# Basiswinkel und -geschwindigkeit
base_theta = np.deg2rad(48.43)
base_v0 = wurfgeschwindigkeit(base_theta)

# Parabeln erzeugen
shorts = generate_wurf(
    base_theta, base_v0, {"theta": (-0.15, -0.05), "v0": (-0.5, -0.2)}, "short", 100
)
hits = generate_wurf(
    base_theta, base_v0, {"theta": (-0.03, 0.03), "v0": (-0.1, 0.1)}, "hit", 100
)
longs = generate_wurf(
    base_theta, base_v0, {"theta": (0.05, 0.15), "v0": (0.2, 0.5)}, "long", 100
)

# In DataFrame und CSV schreiben
data = shorts + hits + longs
df = pd.DataFrame(data)
df.to_csv("wurfbahnen_labeled.csv", index=False)
# Winkel- und Geschwindigkeitsbereiche je Klasse berechnen
winkel_bereiche = (
    df.groupby("label")["theta"]
    .agg(["min", "max"])
    .applymap(lambda x: round(x * 180 / 3.14159, 2))
)
geschw_bereiche = df.groupby("label")["v0"].agg(["min", "max"]).round(2)

# Beschriftungstext vorbereiten
bereich_text = "".join(
    [
        f"{label}: Winkel {winkel_bereiche.loc[label, 'min']}°–{winkel_bereiche.loc[label, 'max']}°, v0 {geschw_bereiche.loc[label, 'min']}-{geschw_bereiche.loc[label, 'max']} m/s<br>"
        for label in df["label"].unique()
    ]
)

# Wurfbahnen visualisieren
fig = px.line(
    df,
    x="x",
    y="y",
    color="label",
    line_group=df.groupby(["theta", "v0"]).ngroup(),
    title="Simulierte Wurfparabeln nach Kategorie",
    labels={"x": "x [m]", "y": "y [m]"},
)

# Korb als Balken einzeichnen
korb_x0 = 4.115 - 0.2286
korb_x1 = 4.115 + 0.2286
korb_y = 3.048
fig.add_shape(
    type="line",
    x0=korb_x0,
    x1=korb_x1,
    y0=korb_y,
    y1=korb_y,
    line=dict(color="black", width=6),
    name="Korb",
)

# Annotation der Winkel- und Geschwindigkeitsbereiche
fig.add_annotation(
    x=0.01,
    y=0.99,
    xref="paper",
    yref="paper",
    text=f"<b>Winkel- und Geschwindigkeitsbereiche:</b><br>{bereich_text}",
    showarrow=False,
    align="left",
    bordercolor="black",
    borderwidth=1,
    bgcolor="white",
    opacity=0.8,
)

fig.update_layout(legend_title_text="Kategorie")
fig.show()
