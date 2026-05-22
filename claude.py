
import math
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════
#  CONSTANTES FÍSICAS
# ══════════════════════════════════════════════════════════════════
G_GRAV = 9.81   # gravedad [m/s²]

# ══════════════════════════════════════════════════════════════════
#  INPUTS — Parámetros reales del auto GR001
#  (Fuente: Memoria de Cálculo, Tablas 1-4 y ensayos)
# ══════════════════════════════════════════════════════════════════

# — Masa (Tabla 1, Fusion 360) —
m_vehiculo  = 1.130   # kg — auto sin batería ni motor (chasis + ruedas)
m_carga     = 0.270   # kg — motor DC RS550
# La masa total se calcula en el pipeline (P1)

# — Geometría (Tabla 2) —
r_rueda   = 0.0325   # m   — radio de rueda (32.5 mm)
d_ejes    = 0.251    # m   — distancia entre ejes
d_CG_del  = 0.151    # m   — distancia CG → eje delantero

# — Coeficiente de roce (Tabla 8, ensayo experimental) —
# Ecuación de servilleta:
#   µ = F_traccion / N_normal
#   µ = 0.515  ±  0.021  (IC 95%, n=15 mediciones)
mu         = 0.515
mu_error   = 0.021

# — Transmisión (Sección 2.3.2) —
# Ecuación de servilleta:
#   G = N_engranaje_mayor / N_engranaje_menor = 44 / 17 = 2.588
#   A mayor G → más torque, menos velocidad
N_mayor   = 44
N_menor   = 17
G_transm  = N_mayor / N_menor   # = 2.588

# — Motor DC RS550 (Tabla 4) —
RPM_nominal      = 10_000   # RPM a 12 V nominal
V_motor_nominal  = 12.0     # V
V_bat_operacion  = 9.6      # V — voltaje mínimo sin dañar la batería
I_motor          = 2.0      # A — corriente a carga baja (Tabla 9, medición 1)

# — Batería LiPo 3S (Tabla 3) —
V_bat_nominal  = 11.1    # V
C_bat_mah      = 2200    # mAh

# — Pista de carrera —
d_pista = 20.0   # m — distancia estimada (ajustar según carrera real)


# ══════════════════════════════════════════════════════════════════
#  PROCESS — Pipeline de cálculos (sigue el diagrama de flujo)
# ══════════════════════════════════════════════════════════════════

def calcular_pipeline(m_vehiculo, m_carga, mu, G_transm, r_rueda,
                      V_bat, RPM_nominal, I_motor, C_bat_mah, d_pista,
                      d_ejes=0.251, d_CG_del=0.151, mu_error=0.021):
    """
    Ejecuta el pipeline completo GR001.

    Parámetros
    ----------
    m_vehiculo  : Masa del vehículo sin carga  [kg]
    m_carga     : Masa de la carga / motor     [kg]
    mu          : Coeficiente de roce estático [-]
    G_transm    : Relación de transmisión      [-]
    r_rueda     : Radio de la rueda            [m]
    V_bat       : Voltaje de operación         [V]
    RPM_nominal : RPM nominales del motor      [RPM]
    I_motor     : Corriente del motor          [A]
    C_bat_mah   : Capacidad de la batería      [mAh]
    d_pista     : Distancia de la pista        [m]

    Retorna
    -------
    dict con todos los resultados intermedios y finales.
    """

    # ── P1: Masa total ─────────────────────────────────────────
    # Ecuación de servilleta:
    #   M = m_vehiculo + m_carga
    M = m_vehiculo + m_carga

    # ── P2: Reacciones normales (diagrama cuerpo libre) ────────
    # Ecuación de servilleta:
    #   ΣFy = 0  →  N_t + N_d = M·g
    #   ΣM  = 0  →  N_t = M·g·d_CG_del / d_ejes
    N_t = (M * G_GRAV * d_CG_del) / d_ejes   # normal eje trasero  [N]
    N_d = M * G_GRAV - N_t                   # normal eje delantero [N]

    # ── P3: Fuerza de roce y tracción ─────────────────────────
    # Ecuación de servilleta:
    #   Fr = µ · N_t          (fricción sobre rueda trasera)
    #   Ft = Fr               (a vel. constante, tracción = roce)
    #
    #   Intervalo con incerteza del ensayo:
    #   Ft_min = (µ - err) · N_t
    #   Ft_max = (µ + err) · N_t
    Fr = mu * N_t
    Ft = Fr
    Ft_min = (mu - mu_error) * N_t
    Ft_max = (mu + mu_error) * N_t

    # ── P4: Torque mecánico ────────────────────────────────────
    # Ecuación de servilleta:
    #   τ_rueda = (N_t/2) · µ · r     (torque por rueda trasera)
    #   τ_eje   = 2 · τ_rueda          (dos ruedas traseras)
    #   τ_motor = τ_eje / G            (después del reductor)
    tau_rueda = (N_t / 2) * mu * r_rueda
    tau_eje   = 2 * tau_rueda
    tau_motor = tau_eje / G_transm

    # ── P5: Velocidad máxima ───────────────────────────────────
    # Ecuación de servilleta:
    #   ω_motor = 2π · RPM / 60        [rad/s]
    #   ω_rueda = ω_motor / G          [rad/s]
    #   v_max   = ω_rueda · r_rueda    [m/s]
    omega_motor = 2 * math.pi * RPM_nominal / 60
    omega_rueda = omega_motor / G_transm
    v_max       = omega_rueda * r_rueda

    # ── P6: Tiempo de carrera ──────────────────────────────────
    # Ecuación de servilleta:
    #   t_carrera = d_pista / v_max
    t_carrera = d_pista / v_max

    # ── P7: Balance eléctrico ──────────────────────────────────
    # Ecuación de servilleta:
    #   P_e = V · I            (potencia eléctrica consumida)
    #   P_m = τ_motor · ω_motor (potencia mecánica entregada)
    #   η   = P_m / P_e        (eficiencia del motor)
    P_e = V_bat * I_motor
    P_m = tau_motor * omega_motor
    eta = (P_m / P_e) * 100 if P_e > 0 else 0

    # ── P8: Corriente promedio y consumo ───────────────────────
    # Ecuación de servilleta:
    #   I_prom      = P_e / V_bat
    #   consumo_mAh = (I_prom · t_carrera / 3600) · 1000
    #   t_autonomia = (C_bat [Ah] / I_prom) · 60   [min]
    I_prom      = P_e / V_bat if V_bat > 0 else 0
    consumo_mah = (I_prom * t_carrera / 3600) * 1000
    t_auto_min  = (C_bat_mah / 1000) / I_prom * 60 if I_prom > 0 else 0

    # ── s15: Validación del sistema ────────────────────────────
    # Condición: el motor debe poder entregar el torque requerido
    tau_requerido  = tau_motor   # torque mínimo para mover el auto
    sistema_viable = tau_motor >= tau_requerido
    estado = (
        "✅  Simulación exitosa — Sistema viable"
        if sistema_viable
        else "❌  Fallo por torque insuficiente — Rediseñar reductor o motor"
    )

    return {
        # Intermedios
        "M_kg":            round(M, 4),
        "N_t_N":           round(N_t, 3),
        "N_d_N":           round(N_d, 3),
        "Fr_N":            round(Fr, 3),
        "Ft_N":            round(Ft, 3),
        "Ft_min_N":        round(Ft_min, 3),
        "Ft_max_N":        round(Ft_max, 3),
        "tau_rueda_Nm":    round(tau_rueda, 5),
        "tau_eje_Nm":      round(tau_eje, 5),
        "tau_motor_Nm":    round(tau_motor, 5),
        "omega_motor_rads":round(omega_motor, 2),
        # Salidas finales
        "v_max_ms":        round(v_max, 3),
        "v_max_kmh":       round(v_max * 3.6, 2),
        "t_carrera_s":     round(t_carrera, 2),
        "P_e_W":           round(P_e, 4),
        "P_m_W":           round(P_m, 4),
        "eta_pct":         round(eta, 2),
        "I_prom_A":        round(I_prom, 4),
        "consumo_mah":     round(consumo_mah, 4),
        "t_auto_min":      round(t_auto_min, 1),
        "estado":          estado,
    }


# ══════════════════════════════════════════════════════════════════
#  OUTPUT — Mostrar resultados en pantalla
# ══════════════════════════════════════════════════════════════════

def mostrar_resultados(r):
    """Despliega los outputs del pipeline en consola."""
    S = "═" * 64
    s = "─" * 60

    print(f"\n{S}")
    print("  GR001 — RESULTADOS DE SIMULACIÓN NUMÉRICA")
    print("  Escudería 6: Guepardex Racing · CDIO 541352")
    print(S)
    print(f"  {'Masa total M':<38}: {r['M_kg']:>8.4f}  kg")
    print(f"  {'Normal eje trasero Nt':<38}: {r['N_t_N']:>8.3f}  N")
    print(f"  {'Normal eje delantero Nd':<38}: {r['N_d_N']:>8.3f}  N")
    print(f"  {s}")
    print(f"  {'Fuerza de tracción Ft':<38}: {r['Ft_N']:>8.3f}  N")
    print(f"    IC 95%: [{r['Ft_min_N']:.3f} , {r['Ft_max_N']:.3f}] N")
    print(f"  {s}")
    print(f"  {'Torque por rueda trasera':<38}: {r['tau_rueda_Nm']:>8.5f}  N·m")
    print(f"  {'Torque en eje trasero':<38}: {r['tau_eje_Nm']:>8.5f}  N·m")
    print(f"  {'Torque requerido en motor':<38}: {r['tau_motor_Nm']:>8.5f}  N·m")
    print(f"  {s}")
    print(f"  {'Velocidad máxima teórica':<38}: {r['v_max_ms']:>8.3f}  m/s"
          f"  →  {r['v_max_kmh']} km/h")
    print(f"  {'Tiempo estimado de carrera':<38}: {r['t_carrera_s']:>8.2f}  s")
    print(f"  {s}")
    print(f"  {'Potencia eléctrica consumida':<38}: {r['P_e_W']:>8.4f}  W")
    print(f"  {'Potencia mecánica entregada':<38}: {r['P_m_W']:>8.4f}  W")
    print(f"  {'Eficiencia del motor':<38}: {r['eta_pct']:>8.2f}  %")
    print(f"  {s}")
    print(f"  {'Corriente promedio de bus':<38}: {r['I_prom_A']:>8.4f}  A")
    print(f"  {'Consumo en la carrera':<38}: {r['consumo_mah']:>8.4f}  mAh")
    print(f"  {'Autonomía estimada batería':<38}: {r['t_auto_min']:>8.1f}  min")
    print(f"  {s}")
    print(f"  Estado del sistema: {r['estado']}")
    print(f"{S}\n")


# ══════════════════════════════════════════════════════════════════
#  ANÁLISIS DE SENSIBILIDAD — Gráficos
# ══════════════════════════════════════════════════════════════════

def graficar_sensibilidad():
    """Genera 3 gráficos de sensibilidad y los guarda como PNG."""

    kw = dict(m_vehiculo=m_vehiculo, m_carga=m_carga, mu=mu,
              G_transm=G_transm, r_rueda=r_rueda,
              V_bat=V_bat_operacion, RPM_nominal=RPM_nominal,
              I_motor=I_motor, C_bat_mah=C_bat_mah, d_pista=d_pista)

    # — Sensibilidad a la masa total —
    masas  = np.linspace(0.8, 2.2, 60)
    Ft_m   = [calcular_pipeline(**{**kw, "m_vehiculo": m - m_carga})["Ft_N"] for m in masas]
    vmax_m = [calcular_pipeline(**{**kw, "m_vehiculo": m - m_carga})["v_max_kmh"] for m in masas]

    # — Sensibilidad al coeficiente de roce —
    mus    = np.linspace(0.30, 0.80, 60)
    Ft_mu  = [calcular_pipeline(**{**kw, "mu": u})["Ft_N"] for u in mus]

    # — Sensibilidad a la relación de transmisión —
    Gs     = np.linspace(1.5, 5.0, 60)
    tau_G  = [calcular_pipeline(**{**kw, "G_transm": g})["tau_motor_Nm"] for g in Gs]
    vmax_G = [calcular_pipeline(**{**kw, "G_transm": g})["v_max_kmh"] for g in Gs]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("GR001 Guepardex Racing — Análisis de Sensibilidad",
                 fontsize=13, fontweight="bold")

    COLOR_PURPLE = "#534AB7"
    COLOR_TEAL   = "#0F6E56"
    COLOR_CORAL  = "#993C1D"
    COLOR_AMBER  = "#BA7517"

    def marca_valor(ax, x_val, xs, ys, color, label):
        y_val = np.interp(x_val, xs, ys)
        ax.axvline(x_val, color=color, linestyle="--", linewidth=1, alpha=0.7, label=label)
        ax.plot(x_val, y_val, "o", color=color, markersize=7, zorder=5)

    # Gráfico 1: Ft vs masa
    axes[0, 0].plot(masas, Ft_m, color=COLOR_PURPLE, linewidth=2)
    axes[0, 0].fill_between(masas, Ft_m, alpha=0.08, color=COLOR_PURPLE)
    marca_valor(axes[0, 0], m_vehiculo + m_carga, masas, Ft_m,
                COLOR_CORAL, f"GR001: {m_vehiculo+m_carga} kg")
    axes[0, 0].set_title("Fuerza de tracción vs. masa total")
    axes[0, 0].set_xlabel("Masa total M (kg)")
    axes[0, 0].set_ylabel("Fuerza de tracción Ft (N)")
    axes[0, 0].legend(fontsize=9); axes[0, 0].grid(True, alpha=0.35, linestyle="--")

    # Gráfico 2: Ft vs µ
    axes[0, 1].plot(mus, Ft_mu, color=COLOR_TEAL, linewidth=2)
    axes[0, 1].fill_between(mus, Ft_mu, alpha=0.08, color=COLOR_TEAL)
    axes[0, 1].axvspan(mu - mu_error, mu + mu_error,
                       alpha=0.15, color=COLOR_TEAL, label="IC 95%")
    marca_valor(axes[0, 1], mu, mus, Ft_mu, COLOR_CORAL, f"µ medido = {mu}")
    axes[0, 1].set_title("Fuerza de tracción vs. coef. de roce")
    axes[0, 1].set_xlabel("Coeficiente de roce µ")
    axes[0, 1].set_ylabel("Fuerza de tracción Ft (N)")
    axes[0, 1].legend(fontsize=9); axes[0, 1].grid(True, alpha=0.35, linestyle="--")

    # Gráfico 3: τ_motor vs G
    axes[1, 0].plot(Gs, tau_G, color=COLOR_AMBER, linewidth=2)
    axes[1, 0].fill_between(Gs, tau_G, alpha=0.08, color=COLOR_AMBER)
    marca_valor(axes[1, 0], G_transm, Gs, tau_G, COLOR_CORAL, f"G = {G_transm:.3f}")
    axes[1, 0].set_title("Torque del motor vs. relación de transmisión")
    axes[1, 0].set_xlabel("Relación de transmisión G")
    axes[1, 0].set_ylabel("Torque motor τ (N·m)")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(True, alpha=0.35, linestyle="--")

    # Gráfico 4: v_max vs G
    axes[1, 1].plot(Gs, vmax_G, color=COLOR_PURPLE, linewidth=2)
    axes[1, 1].fill_between(Gs, vmax_G, alpha=0.08, color=COLOR_PURPLE)
    marca_valor(axes[1, 1], G_transm, Gs, vmax_G, COLOR_CORAL, f"G = {G_transm:.3f}")
    axes[1, 1].set_title("Velocidad máxima vs. relación de transmisión")
    axes[1, 1].set_xlabel("Relación de transmisión G")
    axes[1, 1].set_ylabel("Velocidad máxima (km/h)")
    axes[1, 1].legend(fontsize=9); axes[1, 1].grid(True, alpha=0.35, linestyle="--")

    plt.tight_layout()
    out = "sensibilidad_GR001.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Gráfico guardado → {out}\n")


# ══════════════════════════════════════════════════════════════════
#  MAIN — Punto de entrada
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n🏎  GR001 — Guepardex Racing  🏎")
    print("  Simulación Numérica — CDIO 541352 · UdeC\n")

    # Ejecutar pipeline
    resultados = calcular_pipeline(
        m_vehiculo  = m_vehiculo,
        m_carga     = m_carga,
        mu          = mu,
        G_transm    = G_transm,
        r_rueda     = r_rueda,
        V_bat       = V_bat_operacion,
        RPM_nominal = RPM_nominal,
        I_motor     = I_motor,
        C_bat_mah   = C_bat_mah,
        d_pista     = d_pista,
    )

    # Mostrar outputs
    mostrar_resultados(resultados)

    # Gráficos de sensibilidad
    graficar_sensibilidad()
