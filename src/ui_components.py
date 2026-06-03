"""
Composants de rendu (présentation) pour l'app Streamlit (app.py).

Fonctions d'affichage réutilisables : badges de sources, jauge de risque,
facteurs, waterfall SHAP. Dépendent de Streamlit/Plotly + du contenu
statique (src.ui_content). Extraites d'app.py (#10, palier 2).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ui_content import FEATURE_CATEGORIES, feature_label, get_feature_category


def risk_display(proba: float):
    if proba > 0.7:
        return "ÉLEVÉ", "#e74c3c", ""
    elif proba > 0.4:
        return "MOYEN", "#f39c12", ""
    else:
        return "FAIBLE", "#2ecc71", ""


def show_risk_result(
    proba: float,
    hours_away: float,
    duration: float,
    duration_note: str | None = None,
):
    """Bloc de résultat de risque (carte mise en avant, réutilisée partout)."""
    risk_level, risk_color, risk_icon = risk_display(proba)
    pct = int(proba * 100)
    when_str = f"{hours_away:.0f} h" if hours_away >= 1 else "< 1 h"

    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg,{risk_color}12,{risk_color}22);
                    border:1px solid {risk_color}55;
                    border-left:6px solid {risk_color};
                    border-radius:12px;padding:20px 26px;margin:8px 0 20px 0'>
            <div style='display:flex;justify-content:space-between;align-items:center;
                        flex-wrap:wrap;gap:16px'>
                <div>
                    <div style='font-size:12px;color:rgba(120,120,120,0.95);
                                text-transform:uppercase;letter-spacing:1.5px'>
                        Synthèse du risque
                    </div>
                    <div style='font-size:36px;font-weight:800;color:{risk_color};
                                line-height:1.1;margin-top:4px'>
                        {risk_level}
                    </div>
                </div>
                <div style='display:flex;gap:32px;flex-wrap:wrap'>
                    <div>
                        <div style='font-size:11px;color:rgba(120,120,120,0.95);text-transform:uppercase;
                                    letter-spacing:1.2px'>Probabilité</div>
                        <div style='font-size:32px;font-weight:700;color:{risk_color}'>
                            {pct}%
                        </div>
                    </div>
                    <div>
                        <div style='font-size:11px;color:rgba(120,120,120,0.95);text-transform:uppercase;
                                    letter-spacing:1.2px'>Délai estimé</div>
                        <div style='font-size:32px;font-weight:700;color:var(--text-color, #222)'>
                            {when_str}
                        </div>
                    </div>
                    <div>
                        <div style='font-size:11px;color:rgba(120,120,120,0.95);text-transform:uppercase;
                                    letter-spacing:1.2px'>Durée probable</div>
                        <div style='font-size:32px;font-weight:700;color:var(--text-color, #222)'>
                            {duration} h
                        </div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if duration_note:
        st.caption(duration_note)


def ui_step(title: str, detail: str = "") -> None:
    text = f"**{title}**"
    if detail:
        text += f" — {detail}"
    st.markdown(text)


def category_badge_html(cat_key: str) -> str:
    """HTML d'un badge coloré pour une catégorie de feature."""
    cat = FEATURE_CATEGORIES.get(cat_key, FEATURE_CATEGORIES["other"])
    return (
        f"<span style='background:{cat['color']}22;color:{cat['color']};"
        f"padding:2px 8px;border-radius:10px;font-size:11px;"
        f"font-weight:600;white-space:nowrap'>"
        f"{cat['label']}</span>"
    )


def show_factors(factors: list[dict]):
    """Affichage textuel groupé par catégorie pour les facteurs."""
    st.caption("Lecture rapide : contribution estimée des variables les plus influentes.")
    for f in factors:
        cat_key = get_feature_category(f["feature"])
        cat = FEATURE_CATEGORIES.get(cat_key, FEATURE_CATEGORIES["other"])
        pct = f["importance"] * 100
        st.markdown(
            f"<div style='border-left:3px solid {cat['color']};padding:6px 12px;"
            f"margin-bottom:8px'>"
            f"<div style='display:flex;justify-content:space-between;align-items:center'>"
            f"<b>{f['label']}</b>{category_badge_html(cat_key)}"
            f"</div>"
            f"<span style='color:#888;font-size:12px'>"
            f"Valeur : <code>{f['value']:.2f}</code> · "
            f"Importance : <b>{pct:.1f}%</b></span>"
            f"</div>",
            unsafe_allow_html=True,
        )


def show_shap_waterfall(shap_vals, expected_value, feature_cols: list[str], title: str = ""):
    """Affiche un waterfall SHAP via Plotly, avec préfixe emoji catégorie."""
    indices = np.argsort(np.abs(shap_vals))[::-1][:12]

    cat_keys = [get_feature_category(feature_cols[i]) for i in indices]
    features = [feature_label(feature_cols[i]) for i in indices]
    values = [shap_vals[i] for i in indices]

    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig = go.Figure(go.Bar(
        y=features[::-1],
        x=values[::-1],
        orientation="h",
        marker_color=colors[::-1],
        text=[f"{v:+.3f}" for v in values[::-1]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP : %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title or "Facteurs explicatifs (SHAP)",
                   font=dict(size=14)),
        xaxis_title="Impact (log-odds)",
        yaxis_title="",
        height=max(320, len(indices) * 32),
        margin=dict(l=240, r=70, t=50, b=40),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.add_vline(x=0, line_color="rgba(0,0,0,0.3)", line_width=1)
    fig.add_annotation(
        text=f"Base SHAP : {expected_value:.3f}",
        xref="paper", yref="paper",
        x=1.0, y=-0.08,
        showarrow=False,
        font=dict(size=11, color="#888"),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption("Rouge : augmente le risque · Vert : le réduit")


def show_live_grid_weather_risk(
    hospital_key: str,
    *,
    key_prefix: str = "em_risk",
    show_charts: bool = True,
) -> None:
    """Risque contextuel Electricity Maps (zone GPS) + météo — hors modèle Lacor."""
    from src.app_data import load_live_outage_risk

    st.caption(
        "Zone Electricity Maps au **point GPS** de l'hôpital + météo récente. "
        "Score **indicatif** (distinct du modèle 1/3/6 h dans l'autre sous-onglet)."
    )
    bust_key = f"{key_prefix}_bust"
    bust = st.session_state.get(bust_key, 0.0)
    if st.button("Rafraîchir réseau + météo", key=f"{key_prefix}_refresh"):
        st.session_state[bust_key] = bust + 1.0
        st.rerun()

    risk = load_live_outage_risk(hospital_key, bust)
    if risk is None:
        st.warning(
            "Contexte live indisponible. Définir `ELECTRICITY_MAPS_TOKEN` "
            "(electricitymaps.com) puis **Rafraîchir**."
        )
        return

    zm = risk.get("zone_meta") or {}
    zone_line = (
        f"**Zone** `{risk.get('zone', '—')}` "
        f"({risk.get('zone_source', '?')}, lat/lon hôpital"
    )
    if zm.get("country_name"):
        zone_line += f", {zm['country_name']}"
    zone_line += ")"
    st.markdown(zone_line)

    c1, c2, c3 = st.columns(3)
    lvl = risk.get("level", "—")
    delta_color = {"FAIBLE": "normal", "MOYEN": "off", "ÉLEVÉ": "inverse"}.get(lvl, "normal")
    c1.metric("Risque réseau+météo", f"{risk.get('score_pct', 0)} %", delta=lvl, delta_color=delta_color)

    grid = risk.get("grid")
    if grid is not None and not grid.empty and "em_total_load_mw" in grid.columns:
        c2.metric("Charge réseau", f"{float(grid['em_total_load_mw'].iloc[-1]):.0f} MW")
    wx = risk.get("weather")
    if wx is not None and not wx.empty and "temperature_2m" in wx.columns:
        last_wx = wx.sort_values("datetime").iloc[-1]
        c3.metric(
            "Météo site",
            f"{float(last_wx['temperature_2m']):.0f} °C",
            delta=f"vent {float(last_wx.get('wind_speed_10m', 0) or 0):.0f} km/h",
        )

    with st.expander("Détail des facteurs de risque", expanded=False):
        st.caption(risk.get("disclaimer", ""))
        for fac in risk.get("factors", []):
            bar = min(int(100 * float(fac.get("contrib", 0))), 100) / 100.0
            st.progress(
                bar,
                text=(
                    f"{fac.get('label')} — {fac.get('detail')} "
                    f"(poids {100 * float(fac.get('weight', 0)):.0f} %)"
                ),
            )

    if not show_charts:
        return

    gdf = risk.get("grid")
    if gdf is not None and not gdf.empty and "em_total_load_mw" in gdf.columns:
        fig_g = go.Figure()
        fig_g.add_trace(go.Scatter(
            x=gdf["datetime"],
            y=gdf["em_total_load_mw"],
            mode="lines",
            name="Charge zone (MW)",
            line=dict(color="#e67e22"),
        ))
        fig_g.update_layout(
            height=220,
            margin=dict(l=40, r=20, t=28, b=40),
            title="Réseau EM — 24 h",
            yaxis_title="MW",
        )
        st.plotly_chart(fig_g, width="stretch")

    wdf = risk.get("weather")
    if wdf is not None and not wdf.empty and "temperature_2m" in wdf.columns:
        fig_w = go.Figure()
        fig_w.add_trace(go.Scatter(
            x=wdf["datetime"],
            y=wdf["temperature_2m"],
            mode="lines",
            name="Température (°C)",
            line=dict(color="#3498db"),
        ))
        fig_w.update_layout(
            height=180,
            margin=dict(l=40, r=20, t=28, b=40),
            title="Météo récente (point hôpital)",
            yaxis_title="°C",
        )
        st.plotly_chart(fig_w, width="stretch")

    st.divider()
