import numpy as np
import plotly.graph_objects as go


def simulate_freiwurf(v0, angle_deg, h0=2.0, g=9.81):
    """
    Simuliert den Basketball-Freiwurf ohne Luftwiderstand.

    Parameters:
    - v0: Abwurfgeschwindigkeit in m/s
    - angle_deg: Abwurfwinkel in Grad
    - h0: Abwurfhöhe in Metern (Standard: 2.0 m)
    - g: Erdbeschleunigung (Standard: 9.81 m/s^2)

    Returns:
    - t: Zeitarray
    - x: x-Positionen
    - y: y-Positionen
    """
    angle_rad = np.radians(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)

    # Flugzeit berechnen (y(t) = 0): Lösung der quadratischen Gleichung
    t_flight = (vy + np.sqrt(vy**2 + 2 * g * h0)) / g

    t = np.linspace(0, t_flight, num=500)
    x = vx * t
    y = h0 + vy * t - 0.5 * g * t**2

    return t, x, y


# Beispielaufruf und Plot
if __name__ == "__main__":
    v0 = 7.5  # m/s
    angle_deg = 50  # Grad
    h0 = 2.0  # m (Spielerhöhe beim Abwurf)

    t, x, y = simulate_freiwurf(v0, angle_deg, h0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name=f"v0={v0} m/s, angle={angle_deg}°")
    )
    fig.add_trace(
        go.Scatter(
            x=[0, max(x)],
            y=[3.05, 3.05],
            mode="lines",
            name="Korb (3.05 m)",
            line=dict(dash="dash", color="red"),
        )
    )
    fig.update_layout(
        title="Freiwurfparabel ohne Luftwiderstand",
        xaxis_title="x-Distanz (m)",
        yaxis_title="y-Höhe (m)",
        showlegend=True,
        width=800,
        height=500,
    )
    fig.show()
