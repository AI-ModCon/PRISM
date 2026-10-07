"""
PRISM x Aurora HPC -- Optimization Showcase
Auto-advancing dashboard for ~30-second demo video recording.

Usage:
    pip install streamlit plotly streamlit-autorefresh
    streamlit run src/ui/aurora_showcase.py

Controls:
    - Play/Pause toggles auto-advance (6s per panel)
    - Arrow buttons for manual navigation
    - 5 panels x 6 seconds = 30 seconds total
"""

import base64
import os
import subprocess

# Suppress h5py/wandb lazy-load error in Streamlit's file watcher
try:
    import h5py  # noqa: F401
except (ImportError, OSError):
    pass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from streamlit_autorefresh import st_autorefresh
except ImportError:
    st_autorefresh = None

# ── Constants ──────────────────────────────────────────────────
NUM_PANELS = 8
ADVANCE_MS = 10000

C = dict(
    teal="#00D4AA",
    blue="#4FC3F7",
    red="#FF6B6B",
    orange="#FFB74D",
    purple="#B39DDB",
    green="#81C784",
    pink="#F48FB1",
    text="#E6E6E6",
    muted="#8B949E",
    card="#161B22",
    card2="#1C2333",
    border="#30363D",
)

PANEL_TITLES = [
    "Unified Multimodal Architecture",
    "DDP Optimization: 28 to 131 samp/s",
    "DAOS Storage + Smart Batching",
    "FSDP: Scaling to 7B End-to-End",
    "Production-Ready on Aurora",
    "Launch a Training Job",
    "Live Training Monitor",
    "Multimodal Inference Demo",
]

PLOTLY_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6E6E6", size=13, family="monospace"),
    margin=dict(l=60, r=30, t=40, b=50),
)

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="PRISM x Aurora",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session State ──────────────────────────────────────────────
if "panel" not in st.session_state:
    st.session_state.panel = 0
if "playing" not in st.session_state:
    st.session_state.playing = True

# ── Auto-Advance ──────────────────────────────────────────────
if st_autorefresh is not None and st.session_state.playing:
    tick = st_autorefresh(interval=ADVANCE_MS, key="showcase_tick")
    prev = st.session_state.get("_tick", -1)
    if tick > 0 and tick != prev:
        st.session_state.panel = (st.session_state.panel + 1) % NUM_PANELS
    st.session_state._tick = tick

# ── CSS ────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 0; }
    header[data-testid="stHeader"] { background: rgba(13,17,23,0.95); }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { visibility: hidden; height: 0; }
    .stDeployButton { display: none; }
    .metric-card {
        background: #161B22;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #00D4AA;
        text-align: center;
        min-height: 100px;
    }
    .metric-label {
        color: #8B949E;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 0.2rem;
    }
    .metric-value {
        color: #E6E6E6;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.15rem;
    }
    .arch-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0.3rem 0;
    }
    .arch-box {
        border-radius: 8px;
        padding: 0.45rem 0.9rem;
        font-size: 0.82rem;
        font-weight: 600;
        text-align: center;
        min-width: 100px;
        white-space: nowrap;
    }
    .arch-input  { background: #1a2744; border: 1px solid #4FC3F7; color: #4FC3F7; }
    .arch-enc    { background: #1a3a2a; border: 1px solid #00D4AA; color: #00D4AA; }
    .arch-proj   { background: #2a1a3a; border: 1px solid #B39DDB; color: #B39DDB; }
    .arch-llm    { background: #3a2a1a; border: 1px solid #FFB74D; color: #FFB74D; }
    .arch-output { background: #3a1a1a; border: 1px solid #FF6B6B; color: #FF6B6B; }
    .arch-arrow  { color: #30363D; font-size: 1.2rem; margin: 0 0.1rem; }
    .panel-dots {
        display: flex;
        justify-content: center;
        gap: 0.6rem;
        margin-top: 0.8rem;
    }
    .dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        background: #30363D;
        transition: background 0.3s;
    }
    .dot.active { background: #00D4AA; box-shadow: 0 0 6px #00D4AA88; }
    .section-badge {
        display: inline-block;
        background: #1C2333;
        border: 1px solid #30363D;
        border-radius: 6px;
        padding: 0.2rem 0.7rem;
        font-size: 0.75rem;
        color: #8B949E;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stHorizontalBlock"] { gap: 0.6rem; }

    /* ── Staggered fade-in animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .anim-1 { animation: fadeInUp 0.55s ease-out 0.1s both; }
    .anim-2 { animation: fadeInUp 0.55s ease-out 0.35s both; }
    .anim-3 { animation: fadeInUp 0.55s ease-out 0.6s both; }
    .anim-4 { animation: fadeInUp 0.55s ease-out 0.85s both; }

    @keyframes countPulse {
        0%   { transform: scale(1); }
        50%  { transform: scale(1.06); }
        100% { transform: scale(1); }
    }
    .count-pulse { animation: countPulse 0.4s ease-out 1.8s both; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Helpers ────────────────────────────────────────────────────
def metric_card(label, value, delta=None, color=C["teal"]):
    delta_html = ""
    if delta:
        delta_html = f'<div class="metric-delta" style="color:{color};">{delta}</div>'
    return f"""
    <div class="metric-card" style="border-left-color:{color};">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def panel_header(idx):
    badge = f'<span class="section-badge">Panel {idx + 1} / {NUM_PANELS}</span>'
    title = PANEL_TITLES[idx]
    st.markdown(
        f"<div class='anim-1'>{badge}<br>"
        f"<h2 style='margin:0;'>{title}</h2></div>",
        unsafe_allow_html=True,
    )


# ── Panel 0: Architecture ─────────────────────────────────────
def _hex_to_rgba(hex_color, alpha=0.12):
    """Convert #RRGGBB to rgba(r,g,b,a) for Plotly fill."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _arch_box(fig, x0, y0, x1, y1, label, color, sublabel=None):
    """Add a rounded-rect box with label to a Plotly figure."""
    fig.add_shape(
        type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
        fillcolor=_hex_to_rgba(color, 0.12),
        line=dict(color=color, width=2),
        layer="below",
    )
    text = f"<b>{label}</b>"
    if sublabel:
        text += f"<br><span style='font-size:9px;color:#8B949E'>{sublabel}</span>"
    fig.add_annotation(
        x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=text,
        showarrow=False, font=dict(color=color, size=11),
    )


def _arch_arrow(fig, x0, y0, x1, y1):
    """Add an arrow between two points."""
    fig.add_annotation(
        x=x1, y=y1, ax=x0, ay=y0,
        xref="x", yref="y", axref="x", ayref="y",
        showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
        arrowcolor="#555D66",
    )


def render_architecture():
    panel_header(0)

    fig = go.Figure()

    # Layout coordinates
    # Inputs: x [0, 1.4], Encoders: x [2.2, 3.9], Projector: x [5, 7], LLM: x [8, 10], Output: x [11, 13]
    # Y rows: 5=Image, 4=Table, 3=TS, 2=Geo, 1=Graph, 0=Text
    box_h = 0.35  # half-height of a row box

    inputs = [
        (5, "Image",       C["blue"]),
        (4, "Table",        C["blue"]),
        (3, "Time Series",  C["blue"]),
        (2, "Geometry",     C["blue"]),
        (1, "Graph",        C["blue"]),
    ]
    encoders = [
        (5, "SigLIP2",   C["teal"]),
        (4, "TAPAS",      C["teal"]),
        (3, "Moirai",     C["teal"]),
        (2, "Walrus",     C["teal"]),
        (1, "GraphMAE2",  C["teal"]),
    ]

    # Draw input boxes and encoder boxes with arrows between them
    for y, label, color in inputs:
        _arch_box(fig, 0, y - box_h, 1.4, y + box_h, label, color)
    for y, label, color in encoders:
        _arch_box(fig, 1.9, y - box_h, 4.2, y + box_h, label, color)
    for y, _, _ in inputs:
        _arch_arrow(fig, 1.4, y, 1.9, y)

    # Text input (bypasses encoder, goes directly to LLM)
    _arch_box(fig, 0, 0 - box_h, 1.4, 0 + box_h, "Text", C["blue"])
    # Dashed line from Text to LLM
    fig.add_shape(
        type="line", x0=1.4, y0=0, x1=8, y1=0,
        line=dict(color="#555D66", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=4.7, y=-0.15, text="<i>tokenized directly</i>",
        showarrow=False, font=dict(color="#8B949E", size=10),
    )

    # Arrows from encoders to projector
    for y, _, _ in encoders:
        _arch_arrow(fig, 4.2, y, 5, y)

    # Projector (tall box spanning encoder rows)
    _arch_box(fig, 5, 0.5, 7, 5.5, "Modality<br>Projector", C["purple"], "MLP + LayerNorm")

    # Arrow projector -> LLM
    _arch_arrow(fig, 7, 3, 8, 3)

    # LLM backbone (tall box)
    _arch_box(fig, 8, -0.1, 10, 5.5, "OLMo-3 7B", C["orange"], "32 layers, 4096-d")

    # Arrow LLM -> Output
    _arch_arrow(fig, 10, 3, 11, 3)

    # Output
    _arch_box(fig, 11, 1.5, 13, 4.5, "Text<br>Output", C["red"], "CE loss on<br>text tokens")

    arch_layout = {k: v for k, v in PLOTLY_BASE.items() if k != "margin"}
    fig.update_layout(
        **arch_layout,
        xaxis=dict(visible=False, range=[-0.5, 13.5]),
        yaxis=dict(visible=False, range=[-1, 6.2], scaleanchor="x", scaleratio=0.45),
        height=340,
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='anim-2'>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="arch_diagram")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='anim-3'>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        cols = st.columns(2)
        cols[0].markdown(metric_card("Modalities", "6", "text, image, table, TS, geo, graph", C["blue"]), unsafe_allow_html=True)
        cols[1].markdown(metric_card("Backbone", "7.3B", "OLMo-3 parameters", C["orange"]), unsafe_allow_html=True)
        cols = st.columns(2)
        cols[0].markdown(metric_card("Datasets", "26+", "4 groups, 20M+ samples", C["purple"]), unsafe_allow_html=True)
        cols[1].markdown(metric_card("Platform", "Aurora", "Intel Max 1550 GPUs", C["green"]), unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#8B949E;font-size:0.8rem;margin-top:0.6rem;">'
            "Specialized encoders project each modality into a shared token space. "
            "The LLM backbone processes the concatenated sequence and generates text.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Panel 1: DDP Throughput Evolution ─────────────────────────
def _throughput_animated_bars():
    """Generate CSS-animated bar chart for DDP throughput evolution."""
    milestones = [
        ("Initial DDP", 28.5, "", "#8B949E"),
        ("+ CCL Tuning", 64, "+125%", "#4FC3F7"),
        ("+ Data Fix", 78.6, "+23%", "#4FC3F7"),
        ("+ static_graph", 98.4, "+25%", "#00D4AA"),
        ("+ Bucketing BS=3", 131, "4.6x total", "#FFB74D"),
    ]
    max_val = 160
    bars_html = ""
    for i, (label, val, gain, color) in enumerate(milestones):
        pct = val / max_val * 100
        delay = 300 + i * 400
        gain_html = f'<span style="color:{color};font-size:0.75rem;font-weight:700;margin-left:6px;">{gain}</span>' if gain else ""
        bars_html += f"""
        <div style="display:flex;align-items:center;margin:6px 0;opacity:0;
                    animation:fadeInUp 0.4s ease-out {delay}ms both;">
          <div style="width:110px;text-align:right;padding-right:10px;font-size:0.78rem;
                      color:#8B949E;white-space:nowrap;">{label}</div>
          <div style="flex:1;background:#161B22;border-radius:4px;height:28px;position:relative;overflow:hidden;">
            <div style="height:100%;width:0;background:{color};border-radius:4px;
                        animation:barGrow_{i} 0.8s ease-out {delay + 200}ms forwards;"></div>
            <span style="position:absolute;right:8px;top:4px;color:#E6E6E6;font-size:0.85rem;font-weight:700;">
              {val:.0f}</span>
          </div>
          {gain_html}
        </div>
        <style>
          @keyframes barGrow_{i} {{ from {{ width: 0; }} to {{ width: {pct:.1f}%; }} }}
        </style>
        """
    return f"""
    <style>
      @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
      }}
      body {{ margin: 0; background: transparent; }}
    </style>
    <div style="padding:0.5rem 0;">
      <div style="color:#8B949E;font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;
                  margin-bottom:8px;">Throughput (samp/s) &mdash; 2 nodes, 24 tiles</div>
      {bars_html}
    </div>
    """


def render_throughput():
    panel_header(1)

    left, right = st.columns([3, 2])
    with left:
        st.markdown("<div class='anim-2'>", unsafe_allow_html=True)
        from streamlit.components.v1 import html as _st_html
        _st_html(_throughput_animated_bars(), height=260)
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='anim-3'>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        st.markdown(
            metric_card(
                "Scaling Efficiency",
                "55% to 94%",
                "2-node DDP (24 tiles)",
                C["teal"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card(
                "Backward Overhead",
                "-51%",
                "9.34s to 4.56s (CCL ring)",
                C["blue"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card(
                "Key Insight",
                "static_graph",
                "30% loss from unused params -- fixed by modality match",
                C["orange"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Panel 2: DAOS + Bucketing ────────────────────────────────
def render_daos():
    panel_header(2)

    left, right = st.columns(2)

    with left:
        st.markdown("<div class='anim-2'>", unsafe_allow_html=True)
        # Startup time comparison
        fig1 = go.Figure()
        fig1.add_trace(
            go.Bar(
                x=["Lustre Staging", "DAOS Direct"],
                y=[240, 30],
                marker_color=[C["muted"], C["teal"]],
                marker_line=dict(width=1, color="#30363D"),
                text=["4 min", "30 sec"],
                textposition="outside",
                textfont=dict(size=14, color=C["text"]),
                width=0.5,
            )
        )
        fig1.add_annotation(
            x="DAOS Direct",
            y=100,
            text="<b>8x faster startup</b>",
            showarrow=False,
            font=dict(size=13, color=C["teal"]),
        )
        fig1.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Job Startup Time (8 nodes)", font=dict(size=14)),
            yaxis=dict(
                title="Seconds",
                gridcolor="#21262D",
                range=[0, 300],
                zeroline=False,
            ),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig1, use_container_width=True, key="daos_startup")

        # DAOS architecture summary
        cols = st.columns(2)
        cols[0].markdown(metric_card("DAOS Bandwidth", "30 TB/s", "pool-level aggregate", C["teal"]), unsafe_allow_html=True)
        cols[1].markdown(metric_card("Shard Discovery", "0.6s", "rank-0 broadcast (was 30min)", C["blue"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='anim-3'>", unsafe_allow_html=True)
        # Bucketing throughput comparison
        fig2 = go.Figure()
        configs = ["Mixed\nNo Bucketing", "Pixmo Only\nBaseline", "Mixed + BS=3\nBucketing"]
        tp_vals = [33.4, 97, 131]
        colors2 = [C["red"], C["muted"], C["teal"]]
        fig2.add_trace(
            go.Bar(
                x=configs,
                y=tp_vals,
                marker_color=colors2,
                marker_line=dict(width=1, color="#30363D"),
                text=[f"{v:.0f}" for v in tp_vals],
                textposition="outside",
                textfont=dict(size=14, color=C["text"]),
                width=0.5,
            )
        )
        fig2.add_annotation(
            x="Mixed + BS=3\nBucketing",
            y=163,
            text="<b>3.9x vs naive mixed</b>",
            showarrow=False,
            font=dict(size=13, color=C["teal"]),
        )
        fig2.update_layout(
            **PLOTLY_BASE,
            title=dict(text="Sequence Length Bucketing Impact", font=dict(size=14)),
            yaxis=dict(
                title="Throughput (samp/s)",
                gridcolor="#21262D",
                range=[0, 175],
                zeroline=False,
            ),
            height=280,
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True, key="bucket_chart")

        cols = st.columns(2)
        cols[0].markdown(
            metric_card(
                "Root Cause",
                "O(n^2)",
                "attention on variable seq lengths",
                C["orange"],
            ),
            unsafe_allow_html=True,
        )
        cols[1].markdown(
            metric_card("Buffer Size", "5000", "samples for length sorting", C["purple"]),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Panel 3: FSDP + Memory ───────────────────────────────────
def render_fsdp():
    panel_header(3)

    left, right = st.columns([3, 2])

    with left:
        st.markdown("<div class='anim-2'>", unsafe_allow_html=True)
        # Memory breakdown comparison
        strategies = ["DDP<br>(impossible)", "FSDP<br>BS=16", "FSDP+compile<br>BS=24"]

        # DDP: full replica on each tile
        ddp_params = 14.6
        ddp_grads = 14.7
        ddp_grad_buf = 14.7
        ddp_opt = 29.2
        ddp_act = 1.5
        ddp_tmp = 4.0

        # FSDP full_shard (24 ranks): 1/24 of params+grads+opt, plus activations
        fsdp_params = 14.6 / 24
        fsdp_grads = 14.7 / 24
        fsdp_opt = 29.2 / 24
        fsdp_gather = 14.6  # AllGather buffer (1 unit at a time)
        fsdp_act = 20.0  # BS=16, grad ckpt/2

        # FSDP + compile: same sharding but less activation memory
        comp_params = fsdp_params
        comp_grads = fsdp_grads
        comp_opt = fsdp_opt
        comp_gather = 14.6
        comp_act = 12.0  # 35% less activations

        components = ["Parameters", "Gradients", "Optimizer", "AllGather / DDP buf", "Activations", "Temporaries"]
        colors_stack = [C["blue"], C["teal"], C["purple"], C["orange"], C["pink"], C["muted"]]

        ddp_vals = [ddp_params, ddp_grads, ddp_opt, ddp_grad_buf, ddp_act, ddp_tmp]
        fsdp_vals = [fsdp_params, fsdp_grads, fsdp_opt, fsdp_gather, fsdp_act, 0]
        comp_vals = [comp_params, comp_grads, comp_opt, comp_gather, comp_act, 0]

        fig = go.Figure()
        for i, comp in enumerate(components):
            fig.add_trace(
                go.Bar(
                    name=comp,
                    x=strategies,
                    y=[ddp_vals[i], fsdp_vals[i], comp_vals[i]],
                    marker_color=colors_stack[i],
                    marker_line=dict(width=0.5, color="#30363D"),
                )
            )

        # Tile capacity line
        fig.add_hline(
            y=68.7,
            line_dash="dot",
            line_color="#FF4444",
            line_width=2.5,
            annotation_text="<b>Tile capacity: 68.7 GB</b>",
            annotation_position="top left",
            annotation_font=dict(color="#FF4444", size=13),
        )

        fig.update_layout(
            **PLOTLY_BASE,
            barmode="stack",
            yaxis=dict(
                title="GPU Memory (GB)",
                gridcolor="#21262D",
                range=[0, 90],
                zeroline=False,
            ),
            xaxis=dict(tickfont=dict(size=12)),
            height=350,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
        )

        # Add total labels on top
        totals = [sum(ddp_vals), sum(fsdp_vals), sum(comp_vals)]
        for _i, (strat, total) in enumerate(zip(strategies, totals, strict=False)):
            status = "OOM" if total > 68.7 else f"{total:.0f} GB"
            color = C["red"] if total > 68.7 else C["green"]
            fig.add_annotation(
                x=strat,
                y=total + 2,
                text=f"<b>{status}</b>",
                showarrow=False,
                font=dict(size=12, color=color),
            )

        st.plotly_chart(fig, use_container_width=True, key="mem_chart")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='anim-3'>", unsafe_allow_html=True)
        st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
        st.markdown(
            metric_card(
                "FSDP Strategy",
                "FULL_SHARD",
                "params + grads + optimizer all sharded",
                C["purple"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card(
                "Grad Checkpointing",
                "16/32",
                "every other layer (best tradeoff)",
                C["teal"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card(
                "torch.compile",
                "-35% mem",
                "50.5 to 33.0 GB at BS=16",
                C["blue"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card(
                "Compilation",
                "90s",
                "one-time, backbone-only (0 graph breaks)",
                C["orange"],
            ),
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Panel 4: Production Summary ──────────────────────────────
PIXMO_SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "pixmo_samples")


def _load_pixmo_b64():
    """Load sample images as base64 for JS cycling."""
    imgs = []
    if not os.path.isdir(PIXMO_SAMPLES_DIR):
        return imgs
    for fname in sorted(os.listdir(PIXMO_SAMPLES_DIR)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue
        fpath = os.path.join(PIXMO_SAMPLES_DIR, fname)
        with open(fpath, "rb") as f:
            data = f.read()
        ext = fname.rsplit(".", 1)[-1].lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
        imgs.append(f"data:image/{mime};base64,{base64.b64encode(data).decode()}")
        if len(imgs) >= 30:
            break
    return imgs


def _animated_counter_html(target, decimals, unit, elem_id, display_label, color, delay_ms):
    """Generate HTML/JS for an animated count-up number."""
    return f"""
    <div style="text-align:center;opacity:0;animation:fadeInUp 0.5s ease-out {delay_ms}ms both;">
      <div style="color:{color};font-weight:700;font-size:1.0rem;margin-bottom:0.3rem;">{display_label}</div>
      <div id="{elem_id}" style="color:#E6E6E6;font-size:2.8rem;font-weight:800;line-height:1;">0</div>
      <div style="color:#8B949E;font-size:0.85rem;">{unit}</div>
    </div>
    <script>
    (function() {{
      var el = document.getElementById("{elem_id}");
      var target = {target};
      var decimals = {decimals};
      var duration = 1600;
      var startTime = null;
      function ease(t) {{ return t < 0.5 ? 2*t*t : -1+(4-2*t)*t; }}
      function animate(ts) {{
        if (!startTime) startTime = ts;
        var p = Math.min((ts - startTime) / duration, 1);
        var val = target * ease(p);
        el.textContent = decimals > 0 ? val.toFixed(decimals) : Math.round(val).toString();
        if (p < 1) requestAnimationFrame(animate);
      }}
      setTimeout(function() {{ requestAnimationFrame(animate); }}, {delay_ms});
    }})();
    </script>
    """


def render_summary():
    panel_header(4)

    # ── Animated counters row ──
    counter_html = f"""
    <style>
      @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(18px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
      }}
      @keyframes countPulse {{
        0%   {{ transform: scale(1); }}
        50%  {{ transform: scale(1.06); }}
        100% {{ transform: scale(1); }}
      }}
      body {{ margin: 0; background: transparent; }}
    </style>
    <div style="display:flex;justify-content:space-around;align-items:flex-start;
                background:{C['card2']};border-radius:12px;padding:1.2rem 0.5rem;
                border:1px solid {C['border']};">
      <div style="flex:1;border-right:1px solid {C['border']};padding:0 0.5rem;">
        {_animated_counter_html(131, 0, "samp/s (2N, 24 tiles)", "ctr_ddp", "DDP Projector", C['teal'], 200)}
      </div>
      <div style="flex:1;border-right:1px solid {C['border']};padding:0 0.5rem;">
        {_animated_counter_html(13.0, 1, "samp/s (2N, 24 tiles)", "ctr_fsdp", "FSDP E2E", C['orange'], 500)}
      </div>
      <div style="flex:1;padding:0 0.5rem;">
        {_animated_counter_html(8.7, 1, "samp/s (1N, 12 tiles)", "ctr_compile", "FSDP + Compile", C['blue'], 800)}
      </div>
    </div>
    """
    from streamlit.components.v1 import html as _st_html
    _st_html(counter_html, height=140)

    # ── Config details + image feed ──
    left, right = st.columns([3, 2])

    with left:
        st.markdown("<div class='anim-3'>", unsafe_allow_html=True)
        # Three config cards side by side
        cc1, cc2, cc3 = st.columns(3)
        cc1.markdown(
            f"""<div style="background:{C['card']};border-radius:8px;padding:0.8rem;
                            border-top:3px solid {C['teal']};font-size:0.78rem;color:{C['muted']};line-height:1.6;">
                BS=3 | bucketing (5000 buf)<br>
                static_graph | 50MB buckets<br>
                26 datasets, 20M+ samples<br>
                <b style="color:{C['text']};">80 min convergence</b>
            </div>""",
            unsafe_allow_html=True,
        )
        cc2.markdown(
            f"""<div style="background:{C['card']};border-radius:8px;padding:0.8rem;
                            border-top:3px solid {C['orange']};font-size:0.78rem;color:{C['muted']};line-height:1.6;">
                BS=16 | FULL_SHARD<br>
                grad_ckpt every 2 layers<br>
                production mode (+6.6%)<br>
                <b style="color:{C['text']};">14.9h full epoch</b>
            </div>""",
            unsafe_allow_html=True,
        )
        cc3.markdown(
            f"""<div style="background:{C['card']};border-radius:8px;padding:0.8rem;
                            border-top:3px solid {C['blue']};font-size:0.78rem;color:{C['muted']};line-height:1.6;">
                BS=24 | backbone-only compile<br>
                35% memory reduction<br>
                +19% vs no-compile baseline<br>
                <b style="color:{C['text']};">First compile on Aurora</b>
            </div>""",
            unsafe_allow_html=True,
        )

        # Bottom row: key stats
        b1, b2, b3, b4 = st.columns(4)
        b1.markdown(metric_card("Issues Resolved", "8", "scaling blockers fixed", C["green"]), unsafe_allow_html=True)
        b2.markdown(metric_card("DDP Speedup", "4.7x", "28 to 131 samp/s", C["teal"]), unsafe_allow_html=True)
        b3.markdown(metric_card("FSDP Speedup", "26x", "0.5 to 13.0 samp/s", C["orange"]), unsafe_allow_html=True)
        b4.markdown(metric_card("Memory Saved", "35%", "via torch.compile", C["blue"]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        # ── Live training image feed ──
        st.markdown("<div class='anim-4'>", unsafe_allow_html=True)
        pixmo_imgs = _load_pixmo_b64()
        if pixmo_imgs:
            img_array_js = ",\n".join(f'"{u}"' for u in pixmo_imgs)
            feed_html = f"""
            <style>body {{ margin: 0; background: transparent; }}</style>
            <div style="text-align:center;">
              <div style="color:{C['muted']};font-size:0.75rem;text-transform:uppercase;
                          letter-spacing:0.06em;margin-bottom:0.4rem;">
                Training Image Feed (8.7 samp/s)
              </div>
              <div style="position:relative;width:100%;aspect-ratio:4/3;background:{C['card']};
                          border-radius:8px;overflow:hidden;border:1px solid {C['border']};">
                <img id="pixmo_feed" style="width:100%;height:100%;object-fit:cover;
                     transition:opacity 0.08s ease;" />
                <div style="position:absolute;bottom:6px;right:8px;background:rgba(0,0,0,0.7);
                     color:{C['teal']};font-size:0.7rem;padding:2px 6px;border-radius:4px;
                     font-family:monospace;" id="pixmo_counter">0 samples</div>
              </div>
            </div>
            <script>
            (function() {{
              var imgs = [{img_array_js}];
              var el = document.getElementById("pixmo_feed");
              var counter = document.getElementById("pixmo_counter");
              var idx = 0;
              var count = 0;
              var rate = 115;  // ~8.7 per second
              function next() {{
                el.src = imgs[idx % imgs.length];
                idx++;
                count++;
                counter.textContent = count + " samples";
              }}
              next();
              setInterval(next, rate);
            }})();
            </script>
            """
            _st_html(feed_html, height=320)
        else:
            st.info(
                "No pixmo_cap sample images found in src/ui/pixmo_samples/. "
                "These are ~20 MB of sample imagery that the installed wheel "
                "deliberately excludes (see [tool.setuptools.package-data] in "
                "pyproject.toml); run this showcase from a git checkout to see "
                "the image feed."
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ── Panel 5: Job Launcher ─────────────────────────────────────
LAUNCHER_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "tools", "launch_aurora_daos.py"
)

DESIGN_PRESETS = {
    "PRISM-IMAGE-ONLY-2N": dict(
        desc="Projector-only, OLMo-1B, 2 nodes (fast iteration)",
        defaults=dict(nodes=2, strategy="ddp", datasets="projector", backbone="OLMo-1B"),
    ),
    "PRISM-IMAGE-ONLY-7B": dict(
        desc="Projector-only, OLMo-7B, 1 node",
        defaults=dict(nodes=1, strategy="ddp", datasets="projector", backbone="OLMo-7B"),
    ),
    "PRISM-OLMO3-E2E-PROD": dict(
        desc="Full E2E, OLMo-3 7B, FSDP (production)",
        defaults=dict(nodes=2, strategy="fsdp", datasets="projector", backbone="OLMo-3 7B"),
    ),
    "PRISM-AGPT2B-PROJ": dict(
        desc="AuroraGPT-2B backbone, projector-only",
        defaults=dict(nodes=1, strategy="ddp", datasets="projector", backbone="AuroraGPT-2B"),
    ),
    "PRISM-OLMO3-MULTI-DATASET-DEBUG-2NODE": dict(
        desc="Debug run, OLMo-3 7B, 100 steps",
        defaults=dict(nodes=2, strategy="ddp", datasets="all", backbone="OLMo-3 7B"),
    ),
}


def render_launcher():
    panel_header(5)

    # Pause auto-advance when on the interactive panel
    st.session_state.playing = False

    left, right = st.columns([3, 2])

    with left:
        # Design preset selector
        design = st.selectbox(
            "Experiment Design",
            list(DESIGN_PRESETS.keys()),
            format_func=lambda d: f"{d}  —  {DESIGN_PRESETS[d]['desc']}",
            key="launch_design",
        )
        preset = DESIGN_PRESETS[design]

        col1, col2 = st.columns(2)
        with col1:
            run_id = st.text_input("Run ID", value=f"DEMO-{design.split('-')[-1]}", key="launch_id")
            nodes = st.number_input("Nodes", min_value=1, max_value=64, value=preset["defaults"]["nodes"], step=1, key="launch_nodes")
            strategy = st.selectbox(
                "Dist Strategy",
                ["ddp", "fsdp", "hsdp"],
                index=["ddp", "fsdp", "hsdp"].index(preset["defaults"]["strategy"]),
                key="launch_strategy",
            )
        with col2:
            queue = st.selectbox("Queue", ["debug", "debug-scaling", "prod", "next-eval", "capacity"], key="launch_queue")
            dataset_groups = st.selectbox(
                "Dataset Groups",
                ["projector", "all", "pixmo", "s1mmalign", "nemotron", "cosyn"],
                index=0,
                key="launch_datasets",
            )
            max_seq = st.selectbox("Max Seq Length", [512, 1024, 2048], index=1, key="launch_seq")

        # Advanced options
        with st.expander("Advanced Options"):
            adv1, adv2, adv3 = st.columns(3)
            with adv1:
                grad_ckpt = st.selectbox("Grad Ckpt Freq", [0, 1, 2, 4], index=2, key="launch_gc")
                use_bucketing = st.checkbox("Bucketing", value=True, key="launch_buck")
            with adv2:
                torch_compile = st.checkbox("torch.compile", value=False, key="launch_compile")
                prod_mode = st.checkbox("Production Mode", value=strategy == "fsdp", key="launch_prod")
            with adv3:
                fsdp_sharding = st.selectbox(
                    "FSDP Sharding",
                    ["full_shard", "shard_grad_op", "no_shard"],
                    index=0,
                    key="launch_shard",
                    disabled=strategy == "ddp",
                )
                no_pil4dfs = st.checkbox("--no-pil4dfs", value=True, key="launch_nopil")

        # Build the command
        cmd_parts = [
            "python", "tools/launch_aurora_daos.py",
            "--id", run_id,
            "--design", design,
            "--nodes", str(nodes),
            "--queue", queue,
            "--batch",
            "--dist-strategy", strategy,
            "--dataset-groups", dataset_groups,
            "--max-seq-length", str(max_seq),
            "--grad-ckpt-freq", str(grad_ckpt),
        ]
        if strategy in ("fsdp", "hsdp"):
            cmd_parts += ["--fsdp-sharding", fsdp_sharding]
        if use_bucketing:
            cmd_parts.append("--use-bucketing")
        if torch_compile:
            cmd_parts.append("--torch-compile")
        if prod_mode:
            cmd_parts.append("--fsdp-production-mode")
        if no_pil4dfs:
            cmd_parts.append("--no-pil4dfs")

        cmd_str = " \\\n    ".join(
            [" ".join(cmd_parts[:2])]
            + [" ".join(cmd_parts[i:i+2]) for i in range(2, len(cmd_parts), 2)]
        )

        st.code(cmd_str, language="bash")

        # Action buttons
        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            dry_run = st.button("Dry Run", use_container_width=True, key="btn_dry")
        with btn2:
            submit = st.button("Submit Job", type="primary", use_container_width=True, key="btn_submit")
        with btn3:
            check_q = st.button("Check Queue", use_container_width=True, key="btn_qstat")

    with right:
        st.markdown(
            metric_card("Backbone", preset["defaults"]["backbone"], design, C["orange"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card("Tiles", str(nodes * 12), f"{nodes} nodes x 12 XPU tiles", C["blue"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            metric_card("Strategy", strategy.upper(), f"sharding: {fsdp_sharding if strategy != 'ddp' else 'N/A'}", C["purple"]),
            unsafe_allow_html=True,
        )

        # Output area for command results
        output_container = st.container()

    # Handle button actions
    if dry_run:
        dry_cmd = cmd_parts.copy()
        # Replace --batch with --dry-run
        if "--batch" in dry_cmd:
            dry_cmd[dry_cmd.index("--batch")] = "--dry-run"
        with output_container:
            st.markdown("**Dry Run Output:**")
            with st.spinner("Generating PBS script..."):
                try:
                    result = subprocess.run(
                        dry_cmd,
                        capture_output=True, text=True, timeout=30,
                        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
                    )
                    output = result.stdout or result.stderr
                    # Show last 30 lines to fit the panel
                    lines = output.strip().split("\n")
                    st.code("\n".join(lines[-30:]), language="bash")
                except Exception as e:
                    st.error(f"Error: {e}")

    if submit:
        with output_container:
            st.markdown("**Submitting job...**")
            with st.spinner("Submitting to PBS..."):
                try:
                    result = subprocess.run(
                        cmd_parts,
                        capture_output=True, text=True, timeout=60,
                        cwd=os.path.join(os.path.dirname(__file__), "..", ".."),
                    )
                    output = result.stdout or result.stderr
                    if result.returncode == 0:
                        st.success(f"Job submitted!\n{output.strip()}")
                    else:
                        st.error(f"Submission failed:\n{output.strip()}")
                except Exception as e:
                    st.error(f"Error: {e}")

    if check_q:
        with output_container:
            st.markdown("**Queue Status:**")
            try:
                result = subprocess.run(
                    ["qstat", "-u", os.environ.get("USER", "<user>")],
                    capture_output=True, text=True, timeout=15,
                )
                output = result.stdout.strip()
                if output:
                    st.code(output, language="text")
                else:
                    st.info("No jobs currently in queue.")
            except Exception as e:
                st.error(f"Error: {e}")


# ── Panel 6: WandB Training Monitor ──────────────────────────
WANDB_PROJECTS = [
    "prism-proj-ablation-v2",
    "prism-encoder-proj-ablation",
    "prism-projector-4node",
    "prism-olmo3-training",
    "prism-zone-a-alignment",
    "prism-molmo-setup",
    "prism-debug",
]


def render_wandb():
    panel_header(6)

    # Pause auto-advance on interactive panel
    st.session_state.playing = False

    try:
        import wandb
    except ImportError:
        st.error("wandb not installed. Run: pip install wandb")
        return

    api = wandb.Api()

    # WandB entity/team that owns WANDB_PROJECTS above. Defaults to your own
    # W&B username unless WANDB_ENTITY is set (e.g. for a shared team entity).
    wandb_entity = os.environ.get("WANDB_ENTITY", "<your-wandb-entity>")

    # Project selector
    ctrl1, ctrl2, ctrl3 = st.columns([2, 3, 1])
    with ctrl1:
        project = st.selectbox("WandB Project", WANDB_PROJECTS, key="wb_project")
    with ctrl3:
        refresh = st.button("Refresh", key="wb_refresh", use_container_width=True)

    # Fetch runs for selected project
    cache_key = f"_wb_runs_{project}"
    if cache_key not in st.session_state or refresh:
        try:
            raw_runs = api.runs(f"{wandb_entity}/{project}", per_page=20, order="-created_at")
            st.session_state[cache_key] = [
                dict(id=r.id, name=r.name, state=r.state,
                     steps=r.lastHistoryStep or 0,
                     created=r.createdAt[:10] if r.createdAt else "?")
                for r in raw_runs
            ]
        except Exception as e:
            st.error(f"Failed to fetch runs: {e}")
            return

    runs_info = st.session_state[cache_key]
    if not runs_info:
        st.info("No runs found in this project.")
        return

    with ctrl2:
        run_options = {
            r["id"]: f"{r['name']}  ({r['state']}, {r['steps']} steps, {r['created']})"
            for r in runs_info
        }
        selected_id = st.selectbox(
            "Run",
            list(run_options.keys()),
            format_func=lambda x: run_options[x],
            key="wb_run",
        )

    selected_info = next(r for r in runs_info if r["id"] == selected_id)

    # Fetch history for selected run
    hist_key = f"_wb_hist_{selected_id}"
    if hist_key not in st.session_state or refresh:
        try:
            run = api.run(f"{wandb_entity}/{project}/{selected_id}")
            hist = run.history(samples=500)
            st.session_state[hist_key] = hist
            st.session_state[f"_wb_config_{selected_id}"] = dict(run.config)
            st.session_state[f"_wb_summary_{selected_id}"] = dict(run.summary)
        except Exception as e:
            st.error(f"Failed to fetch run history: {e}")
            return

    hist = st.session_state[hist_key]
    if hist.empty:
        st.warning("No history data for this run.")
        return

    # Top metrics cards
    m1, m2, m3, m4 = st.columns(4)
    state_color = C["green"] if selected_info["state"] == "running" else C["muted"]
    m1.markdown(
        metric_card("Status", selected_info["state"].upper(), selected_info["created"], state_color),
        unsafe_allow_html=True,
    )

    last_loss = hist["loss"].dropna().iloc[-1] if "loss" in hist.columns and not hist["loss"].dropna().empty else None
    m2.markdown(
        metric_card("Loss", f"{last_loss:.3f}" if last_loss is not None else "N/A",
                    f"step {selected_info['steps']}", C["teal"]),
        unsafe_allow_html=True,
    )

    sps = hist.get("perf/samples_per_sec", pd.Series(dtype=float)).dropna()
    avg_sps = sps.mean() if not sps.empty else None
    m3.markdown(
        metric_card("Throughput", f"{avg_sps:.0f}" if avg_sps else "N/A",
                    "avg samp/s", C["blue"]),
        unsafe_allow_html=True,
    )

    mem = hist.get("perf/mem_allocated_mb", pd.Series(dtype=float)).dropna()
    peak_mem = mem.max() / 1024 if not mem.empty else None
    m4.markdown(
        metric_card("Peak Memory", f"{peak_mem:.1f} GB" if peak_mem else "N/A",
                    "allocated", C["orange"]),
        unsafe_allow_html=True,
    )

    # Charts
    chart_left, chart_right = st.columns(2)

    with chart_left:
        # Loss curve
        if "loss" in hist.columns:
            loss_data = hist[["_step", "loss"]].dropna()
            if not loss_data.empty:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(
                    x=loss_data["_step"], y=loss_data["loss"],
                    mode="lines", line=dict(color=C["teal"], width=2),
                    name="loss",
                ))
                fig_loss.update_layout(
                    **PLOTLY_BASE,
                    title=dict(text="Training Loss", font=dict(size=14)),
                    xaxis=dict(title="Step", gridcolor="#21262D"),
                    yaxis=dict(title="Loss", gridcolor="#21262D"),
                    height=260, showlegend=False,
                )
                st.plotly_chart(fig_loss, use_container_width=True, key="wb_loss")

    with chart_right:
        # Throughput over time
        if "perf/samples_per_sec" in hist.columns:
            sps_data = hist[["_step", "perf/samples_per_sec"]].dropna()
            if not sps_data.empty:
                fig_sps = go.Figure()
                fig_sps.add_trace(go.Scatter(
                    x=sps_data["_step"], y=sps_data["perf/samples_per_sec"],
                    mode="lines+markers", line=dict(color=C["blue"], width=2),
                    marker=dict(size=4), name="samp/s",
                ))
                fig_sps.update_layout(
                    **PLOTLY_BASE,
                    title=dict(text="Throughput (samp/s)", font=dict(size=14)),
                    xaxis=dict(title="Step", gridcolor="#21262D"),
                    yaxis=dict(title="samp/s", gridcolor="#21262D"),
                    height=260, showlegend=False,
                )
                st.plotly_chart(fig_sps, use_container_width=True, key="wb_sps")
        elif "lr" in hist.columns:
            # Fallback: show LR schedule
            lr_data = hist[["_step", "lr"]].dropna()
            if not lr_data.empty:
                fig_lr = go.Figure()
                fig_lr.add_trace(go.Scatter(
                    x=lr_data["_step"], y=lr_data["lr"],
                    mode="lines", line=dict(color=C["purple"], width=2),
                    name="lr",
                ))
                fig_lr.update_layout(
                    **PLOTLY_BASE,
                    title=dict(text="Learning Rate Schedule", font=dict(size=14)),
                    xaxis=dict(title="Step", gridcolor="#21262D"),
                    yaxis=dict(title="LR", gridcolor="#21262D"),
                    height=260, showlegend=False,
                )
                st.plotly_chart(fig_lr, use_container_width=True, key="wb_lr")


# ── Panel 7: Multimodal Inference Demo ────────────────────────
TEST_IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "test_images")


def render_inference():
    panel_header(7)

    # Pause auto-advance on interactive panel
    st.session_state.playing = False

    # Server URL config
    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        server_url = st.text_input(
            "Gradio Server URL",
            value=st.session_state.get("inference_url", "http://localhost:7860"),
            key="inf_url",
            help="URL of the PRISM Gradio server running on a compute node",
        )
        st.session_state["inference_url"] = server_url
    with ctrl2:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        connect = st.button("Connect", use_container_width=True, key="btn_connect")

    # Embed the Gradio UI via iframe
    st.markdown(
        f'<p style="color:{C["muted"]};font-size:0.8rem;margin:0.5rem 0 0.2rem;">'
        "To start the server on a compute node:</p>",
        unsafe_allow_html=True,
    )
    st.code(
        "# Terminal 1 — on compute node (interactive allocation):\n"
        "cd /lus/flare/projects/<project>/<user>/BaseMM_PRISM\n"
        "module load frameworks\n"
        "python3 src/ui/app.py\n\n"
        "# Terminal 2 — on login node (tunnel to compute node):\n"
        "ssh -N -L 7860:localhost:7860 <compute-node-hostname>",
        language="bash",
    )

    # Check if server is reachable and embed
    import urllib.request

    from streamlit.components.v1 import iframe as _st_iframe

    if connect or st.session_state.get("_inf_connected"):
        url = server_url or "http://localhost:7860"
        try:
            urllib.request.urlopen(url, timeout=3)
            st.session_state["_inf_connected"] = True
            _st_iframe(url, height=1000, scrolling=False)
        except Exception:
            st.session_state["_inf_connected"] = False
            st.warning(
                f"Cannot reach {url}. Make sure the Gradio server is running "
                "on the compute node and the SSH tunnel is active:\n\n"
                "`ssh -N -L 7860:localhost:7860 <compute-node-hostname>`"
            )


# ── Main Layout ───────────────────────────────────────────────
# Header
hdr_left, hdr_right = st.columns([3, 1])
with hdr_left:
    st.markdown(
        "<h1 style='margin:0;font-size:1.8rem;'>"
        "PRISM <span style='color:#8B949E;font-weight:400;'>x</span> Aurora HPC"
        "</h1>"
        "<p style='color:#8B949E;margin:0;font-size:0.85rem;'>"
        "Poly-Reasoning Integrated Scientific Multimodal Model &mdash; "
        "Optimization Showcase on Intel Max 1550 GPUs</p>",
        unsafe_allow_html=True,
    )
with hdr_right:
    # Controls
    ctrl_cols = st.columns([1, 1, 1, 2])
    with ctrl_cols[0]:
        if st.button("Prev", use_container_width=True):
            st.session_state.panel = (st.session_state.panel - 1) % NUM_PANELS
            st.rerun()
    with ctrl_cols[1]:
        label = "Pause" if st.session_state.playing else "Play"
        if st.button(label, use_container_width=True):
            st.session_state.playing = not st.session_state.playing
            st.rerun()
    with ctrl_cols[2]:
        if st.button("Next", use_container_width=True):
            st.session_state.panel = (st.session_state.panel + 1) % NUM_PANELS
            st.rerun()
    with ctrl_cols[3]:
        if st_autorefresh is None:
            st.caption("pip install streamlit-autorefresh")

st.markdown("<hr style='margin:0.3rem 0;border-color:#21262D;'>", unsafe_allow_html=True)

# Render current panel
panel_idx = st.session_state.panel
if panel_idx == 0:
    render_architecture()
elif panel_idx == 1:
    render_throughput()
elif panel_idx == 2:
    render_daos()
elif panel_idx == 3:
    render_fsdp()
elif panel_idx == 4:
    render_summary()
elif panel_idx == 5:
    render_launcher()
elif panel_idx == 6:
    render_wandb()
elif panel_idx == 7:
    render_inference()

# Panel indicator dots
dots_html = '<div class="panel-dots">'
for i in range(NUM_PANELS):
    active = "active" if i == panel_idx else ""
    dots_html += f'<div class="dot {active}"></div>'
dots_html += "</div>"
st.markdown(dots_html, unsafe_allow_html=True)
