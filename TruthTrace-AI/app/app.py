import sys
import os
import time
import joblib
import difflib
 
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import networkx as nx
 
from datetime import datetime, timedelta
 
# =========================================================
# PROJECT PATH
# =========================================================
 
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)
 
from src.data_model import Message
from src.data_pipeline.dataset_parser import TwitterTreeParser
from src.ml.feature_engineer import FeatureEngineer
 
 
# =========================================================
# PAGE CONFIG
# =========================================================
 
st.set_page_config(
    page_title="TruthTrace AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
 
# =========================================================
# CUSTOM CSS
# NOTE: Streamlit's markdown renderer treats a blank line
# inside a raw HTML block as the end of that block — anything
# after the first blank line gets rendered as plain text
# instead of being parsed as CSS. To avoid that, this whole
# <style> block is written with NO blank lines and minimal
# (non 4-space) indentation, since 4+ leading spaces also
# trigger Markdown's "indented code block" rule.
# =========================================================
 
st.markdown(
    """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stApp {
background: #08111f;
color: #e5e7eb;
}
.block-container {
max-width: 1450px;
padding-top: 2rem;
padding-bottom: 4rem;
}
[data-testid="stSidebar"] {
background: #0d1728;
border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .block-container {
padding: 1.5rem 1rem;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
color: #ffffff;
}
.stButton > button {
border-radius: 9px;
font-weight: 700;
border: none;
min-height: 42px;
}
.hero-title {
font-size: 42px;
font-weight: 850;
letter-spacing: -1px;
margin-bottom: 5px;
color: #ffffff;
}
.hero-title span {
color: #8b7cff;
}
.hero-subtitle {
color: #94a3b8;
font-size: 14px;
line-height: 1.6;
max-width: 850px;
}
.status-box {
background: #062e2a;
border: 1px solid #047857;
color: #34d399;
border-radius: 999px;
padding: 8px 15px;
text-align: center;
font-size: 12px;
font-weight: 700;
}
.scenario-card {
background: #102a49;
border: 1px solid #1e4a78;
border-radius: 12px;
padding: 16px 20px;
margin: 20px 0;
}
.scenario-label {
color: #7dd3fc;
font-size: 11px;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 1px;
}
.scenario-value {
color: #ffffff;
font-size: 17px;
font-weight: 700;
margin-top: 4px;
}
.section-title {
color: #ffffff;
font-size: 20px;
font-weight: 800;
margin-top: 25px;
margin-bottom: 12px;
}
.metric-card {
background: #101b2d;
border: 1px solid #223149;
border-radius: 14px;
padding: 18px;
min-height: 125px;
}
.metric-title {
color: #94a3b8;
font-size: 12px;
font-weight: 700;
text-transform: uppercase;
letter-spacing: 0.5px;
}
.metric-value {
color: #ffffff;
font-size: 30px;
font-weight: 850;
margin-top: 8px;
}
.metric-description {
color: #64748b;
font-size: 11px;
margin-top: 5px;
}
.threat-card {
background: #101b2d;
border: 1px solid #223149;
border-radius: 16px;
padding: 25px;
}
.threat-number {
font-size: 52px;
font-weight: 900;
line-height: 1;
}
.threat-label {
color: #94a3b8;
margin-top: 8px;
font-size: 13px;
}
.safe { color: #34d399; }
.medium { color: #fbbf24; }
.high { color: #f87171; }
.ready-card {
background: #101b2d;
border: 1px solid #223149;
border-radius: 18px;
padding: 55px 30px;
text-align: center;
margin-top: 25px;
}
.ready-icon {
font-size: 52px;
margin-bottom: 10px;
}
.ready-title {
color: #ffffff;
font-size: 27px;
font-weight: 850;
}
.ready-text {
color: #94a3b8;
font-size: 14px;
margin-top: 10px;
line-height: 1.6;
}
.capability-card {
background: #12345a;
border: 1px solid #24517d;
border-radius: 10px;
padding: 20px;
min-height: 145px;
}
.capability-title {
color: #ffffff;
font-size: 17px;
font-weight: 800;
}
.capability-text {
color: #b8d4ed;
font-size: 12px;
line-height: 1.6;
margin-top: 10px;
}
.pipeline-box {
background: #101b2d;
border: 1px solid #223149;
border-radius: 12px;
padding: 16px;
}
.pipeline-item {
color: #dbeafe;
background: #16395f;
border-radius: 7px;
padding: 9px 10px;
margin-bottom: 7px;
font-size: 12px;
font-weight: 700;
}
.pipeline-arrow {
color: #60a5fa;
text-align: center;
font-size: 12px;
margin-bottom: 7px;
}
.chat-card {
background: #101b2d;
border: 1px solid #223149;
border-radius: 16px;
padding: 20px;
margin-bottom: 12px;
}
.message-source {
background: #12372f;
border: 1px solid #1d5b4e;
border-radius: 14px;
padding: 14px;
margin-left: 18%;
}
.message-forward {
background: #172033;
border: 1px solid #29364d;
border-radius: 14px;
padding: 14px;
margin-right: 18%;
}
.sender-source {
color: #34d399;
font-size: 12px;
font-weight: 800;
}
.sender-forward {
color: #f87171;
font-size: 12px;
font-weight: 800;
}
.message-text {
color: #e5e7eb;
font-size: 15px;
line-height: 1.6;
margin-top: 7px;
}
.message-time {
color: #64748b;
font-size: 10px;
text-align: right;
margin-top: 8px;
}
.diff {
color: #fecaca;
background: #541b1b;
border-radius: 4px;
padding: 2px 4px;
font-weight: 800;
}
.footer {
border-top: 1px solid #1e293b;
margin-top: 45px;
padding-top: 20px;
text-align: center;
color: #64748b;
font-size: 11px;
line-height: 1.7;
}
</style>
""",
    unsafe_allow_html=True
)
 
 
# =========================================================
# HELPER: HIGHLIGHT DIFFERENCES
# =========================================================
 
def highlight_diff(parent_text, child_text):
 
    if not parent_text:
        return child_text
 
    words1 = parent_text.split()
    words2 = child_text.split()
 
    matcher = difflib.SequenceMatcher(
        None,
        words1,
        words2
    )
 
    highlighted = []
 
    for opcode, a0, a1, b0, b1 in matcher.get_opcodes():
 
        if opcode == "equal":
 
            highlighted.append(
                " ".join(words1[a0:a1])
            )
 
        elif opcode in ("insert", "replace"):
 
            mutated = " ".join(
                words2[b0:b1]
            )
 
            highlighted.append(
                f"<span class='diff'>{mutated}</span>"
            )
 
    return " ".join(highlighted)
 
 
# =========================================================
# NETWORK GRAPH
# =========================================================
 
def draw_network_graph(
    messages_so_far,
    current_risk_prob
):
 
    G = nx.DiGraph()
 
    for message in messages_so_far:
 
        G.add_node(message.id)
 
        if message.parent_id:
 
            G.add_edge(
                message.parent_id,
                message.id
            )
 
    if len(G.nodes) == 0:
 
        return go.Figure()
 
    pos = nx.spring_layout(
        G,
        seed=42
    )
 
    node_colors = []
 
    nodes = list(G.nodes())
 
    for index, node in enumerate(nodes):
 
        if index == len(nodes) - 1:
 
            if current_risk_prob < 0.3:
                node_colors.append("#34d399")
 
            elif current_risk_prob < 0.7:
                node_colors.append("#fbbf24")
 
            else:
                node_colors.append("#ef4444")
 
        else:
 
            node_colors.append("#64748b")
 
    edge_x = []
    edge_y = []
 
    for edge in G.edges():
 
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
 
        edge_x.extend([
            x0,
            x1,
            None
        ])
 
        edge_y.extend([
            y0,
            y1,
            None
        ])
 
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(
            width=2,
            color="#475569"
        ),
        mode="lines",
        hoverinfo="none"
    )
 
    node_x = [
        pos[n][0]
        for n in nodes
    ]
 
    node_y = [
        pos[n][1]
        for n in nodes
    ]
 
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
 
        marker=dict(
            size=35,
            color=node_colors,
            line=dict(
                width=2,
                color="#e2e8f0"
            )
        ),
 
        text=nodes,
 
        textposition="bottom center",
 
        textfont=dict(
            color="#cbd5e1",
            size=11
        ),
 
        hoverinfo="text"
    )
 
    fig = go.Figure(
        data=[
            edge_trace,
            node_trace
        ]
    )
 
    fig.update_layout(
 
        showlegend=False,
 
        height=430,
 
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=30
        ),
 
        paper_bgcolor="#101b2d",
 
        plot_bgcolor="#101b2d",
 
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        ),
 
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            showticklabels=False
        )
    )
 
    return fig
 
 
# =========================================================
# LOAD BACKEND
# =========================================================
 
@st.cache_resource
def load_backend():
 
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../data/models/xgb_model.pkl"
        )
    )
 
    if not os.path.exists(model_path):
 
        st.error(
            "Model not found. "
            "Run `python src/ml/train_model.py` first."
        )
 
        st.stop()
 
    model = joblib.load(model_path)
 
    engineer = FeatureEngineer()
 
    return model, engineer
 
 
xgb_model, engineer = load_backend()
 
 
# =========================================================
# SIDEBAR
# =========================================================
 
with st.sidebar:
 
    st.markdown(
        """
<div style="font-size:25px;font-weight:850;color:white;margin-bottom:4px;">
🧠 TruthTrace <span style="color:#8b7cff;">AI</span>
</div>
""",
        unsafe_allow_html=True
    )
 
    st.caption(
        "Misinformation Drift Analyzer"
    )
 
    st.divider()
 
    st.subheader("Scenario")
 
    scenario = st.selectbox(
        "Choose propagation scenario",
        [
            "PHEME: Sydney Siege",
            "Financial Bank Scam",
            "Custom Input"
        ]
    )
 
    messages = []
    start_sim = False
 
    # =====================================================
    # CUSTOM INPUT
    # =====================================================
 
    if scenario == "Custom Input":
 
        st.markdown("### 📝 Create Rumor Chain")
 
        custom_m0 = st.text_area(
            "Original Fact",
            "The school will be closed tomorrow for maintenance."
        )
 
        custom_m1 = st.text_area(
            "Forward 1",
            "I heard the school is closed tomorrow because of an incident."
        )
 
        custom_m2 = st.text_area(
            "Forward 2",
            "URGENT! School closed indefinitely due to a toxic leak! Keep your kids inside!!"
        )
 
        if st.button(
            "▶ Analyze Custom Spread",
            type="primary",
            use_container_width=True
        ):
 
            base_time = datetime.now()
 
            messages = [
 
                Message(
                    "M0",
                    custom_m0,
                    base_time,
                    "Admin",
                    None
                ),
 
                Message(
                    "M1",
                    custom_m1,
                    base_time + timedelta(hours=1),
                    "ParentA",
                    "M0"
                ),
 
                Message(
                    "M2",
                    custom_m2,
                    base_time + timedelta(hours=2),
                    "ParentB",
                    "M1"
                )
 
            ]
 
            start_sim = True
 
    # =====================================================
    # DEFAULT SCENARIOS
    # =====================================================
 
    else:
 
        if scenario == "PHEME: Sydney Siege":
 
            root, replies = (
                TwitterTreeParser.get_pheme_mock_data()
            )
 
            messages = (
                TwitterTreeParser.parse_thread(
                    root,
                    replies
                )
            )
 
        elif scenario == "Financial Bank Scam":
 
            base_time = datetime.now()
 
            messages = [
 
                Message(
                    "M0",
                    "Notice: National Bank is updating its server infrastructure this weekend.",
                    base_time,
                    "BankOfficial",
                    None
                ),
 
                Message(
                    "M1",
                    "I heard National Bank is having server issues, might be a hack.",
                    base_time + timedelta(minutes=30),
                    "User_A",
                    "M0"
                ),
 
                Message(
                    "M2",
                    "My cousin works there, he said hackers breached the vault! ACCOUNTS ARE FROZEN!",
                    base_time + timedelta(minutes=60),
                    "User_B",
                    "M1"
                ),
 
                Message(
                    "M3",
                    "URGENT! National Bank is bankrupt! Withdraw all your cash NOW before the government takes it! 🚨💸",
                    base_time + timedelta(minutes=90),
                    "User_C",
                    "M2"
                )
 
            ]
 
        if st.button(
            "▶ Start Propagation Analysis",
            type="primary",
            use_container_width=True
        ):
 
            start_sim = True
 
    st.divider()
 
 
# =========================================================
# HEADER
# =========================================================
 
title_col, status_col = st.columns(
    [5, 1]
)
 
with title_col:
 
    st.markdown(
        """
<div class="hero-title">TruthTrace <span>AI</span></div>
""",
        unsafe_allow_html=True
    )
 
    st.markdown(
        """
<div class="hero-subtitle">
Track how factual information mutates as it propagates
through a social network using NLP, graph analysis,
and machine learning.
</div>
""",
        unsafe_allow_html=True
    )
 
with status_col:
 
    st.markdown(
        """
<div class="status-box">● AI ENGINE ONLINE</div>
""",
        unsafe_allow_html=True
    )
 
 
# =========================================================
# ACTIVE SCENARIO
# =========================================================
 
st.markdown(
    f"""
<div class="scenario-card">
<div class="scenario-label">Active Scenario</div>
<div class="scenario-value">{scenario}</div>
</div>
""",
    unsafe_allow_html=True
)
 
 
# =========================================================
# SESSION STATE
# =========================================================
 
if "last_features" not in st.session_state:
 
    st.session_state.last_features = None
 
if "last_risk" not in st.session_state:
 
    st.session_state.last_risk = 0.0
 
 
# =========================================================
# PLACEHOLDERS
# =========================================================
 
# Dashboard (metrics / threat card / graph) is redrawn in place
# every hop, since it represents the CURRENT live state.
dashboard_placeholder = st.empty()
 
# Chat log must NOT be wiped every hop - each new message should
# be appended below the previous ones so the full mutation chain
# stays visible. A plain st.container() (not st.empty()) does that:
# every st.markdown() call inside it adds new content instead of
# replacing what's already there.
if start_sim:
 
    st.markdown(
        '<div class="section-title">💬 Message Evolution</div>',
        unsafe_allow_html=True
    )
 
chat_placeholder = st.container()
 
 
# =========================================================
# SIMULATION
# =========================================================
 
if start_sim:
 
    messages_so_far = []
 
    for i, msg in enumerate(messages):
 
        messages_so_far.append(msg)
 
        # =================================================
        # FEATURE ENGINEERING
        # =================================================
 
        features_df = engineer.extract_features(
            messages_so_far
        )
 
        current_features = features_df.iloc[[-1]][
            [
                "semantic_drift",
                "sentiment_delta",
                "exaggeration_score",
                "graph_depth",
                "mutation_rate"
            ]
        ]
 
        # =================================================
        # XGBOOST
        # =================================================
 
        risk_prob = float(
            xgb_model.predict_proba(
                current_features
            )[0][1]
        )
 
        st.session_state.last_features = (
            features_df.iloc[-1]
        )
 
        st.session_state.last_risk = risk_prob
 
        # =================================================
        # FEATURES
        # =================================================
 
        semantic_drift = float(
            features_df.iloc[-1]["semantic_drift"]
        )
 
        exaggeration = float(
            features_df.iloc[-1]["exaggeration_score"]
        )
 
        mutation_rate = float(
            features_df.iloc[-1]["mutation_rate"]
        )
 
        graph_depth = float(
            features_df.iloc[-1]["graph_depth"]
        )
 
        # =================================================
        # THREAT STATUS
        # =================================================
 
        if risk_prob < 0.3:
 
            threat_status = "SAFE"
            threat_class = "safe"
 
        elif risk_prob < 0.7:
 
            threat_status = "SUSPICIOUS"
            threat_class = "medium"
 
        else:
 
            threat_status = "HIGH RISK"
            threat_class = "high"
 
        # =================================================
        # DASHBOARD (overwritten each hop - shows live state)
        # =================================================
 
        with dashboard_placeholder.container():
 
            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------
 
            c1, c2, c3, c4 = st.columns(4)
 
            with c1:
 
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">⚠️ Threat Level</div>
<div class="metric-value {threat_class}">{risk_prob * 100:.1f}%</div>
<div class="metric-description">{threat_status}</div>
</div>
""",
                    unsafe_allow_html=True
                )
 
            with c2:
 
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">🧬 Semantic Drift</div>
<div class="metric-value">{semantic_drift:.2f}</div>
<div class="metric-description">Meaning deviation</div>
</div>
""",
                    unsafe_allow_html=True
                )
 
            with c3:
 
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">🔀 Mutation Rate</div>
<div class="metric-value">{mutation_rate:.2f}</div>
<div class="metric-description">Change per propagation hop</div>
</div>
""",
                    unsafe_allow_html=True
                )
 
            with c4:
 
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">🌐 Graph Depth</div>
<div class="metric-value">{graph_depth:.0f}</div>
<div class="metric-description">Propagation hops</div>
</div>
""",
                    unsafe_allow_html=True
                )
 
            # =================================================
            # THREAT ASSESSMENT
            # =================================================
 
            st.markdown(
                '<div class="section-title">Live Threat Assessment</div>',
                unsafe_allow_html=True
            )
 
            st.markdown(
                f"""
<div class="threat-card">
<div class="threat-number {threat_class}">{risk_prob * 100:.1f}%</div>
<div class="threat-label">Current misinformation probability</div>
<div style="margin-top:12px;font-weight:800;" class="{threat_class}">{threat_status}</div>
</div>
""",
                unsafe_allow_html=True
            )
 
            st.progress(
                min(
                    max(risk_prob, 0.0),
                    1.0
                )
            )
 
            # =================================================
            # GRAPH + RISK
            # =================================================
 
            graph_col, risk_col = st.columns(
                [2.2, 1]
            )
 
            with graph_col:
 
                st.markdown(
                    '<div class="section-title">🌐 Propagation Network</div>',
                    unsafe_allow_html=True
                )
 
                st.plotly_chart(
                    draw_network_graph(
                        messages_so_far,
                        risk_prob
                    ),
                    use_container_width=True,
                    key=f"network_{i}"
                )
 
            with risk_col:
 
                st.markdown(
                    '<div class="section-title">📊 Risk Breakdown</div>',
                    unsafe_allow_html=True
                )
 
                st.markdown(
                    f"""
<div class="metric-card">
<div class="metric-title">Exaggeration</div>
<div class="metric-value">{exaggeration * 100:.1f}%</div>
<br>
<div class="metric-title">Semantic Drift</div>
<div class="metric-value">{semantic_drift * 100:.1f}%</div>
<br>
<div class="metric-title">Mutation</div>
<div class="metric-value">{mutation_rate * 100:.1f}%</div>
</div>
""",
                    unsafe_allow_html=True
                )
 
        # =================================================
        # MESSAGE EVOLUTION (appended to chat_placeholder -
        # NEVER overwritten, so the whole thread accumulates)
        # =================================================
 
        is_source = i == 0
 
        if is_source:
 
            sender_title = (
                f"~ {msg.sender_id} • Original Source"
            )
 
            message_class = "message-source"
            sender_class = "sender-source"
 
            display_text = msg.text
 
        else:
 
            depth = features_df.iloc[-1][
                "graph_depth"
            ]
 
            sender_title = (
                f"~ {msg.sender_id} • "
                f"Forwarded • Hop {depth}"
            )
 
            message_class = "message-forward"
            sender_class = "sender-forward"
 
            parent_text = (
                messages_so_far[i - 1].text
            )
 
            display_text = highlight_diff(
                parent_text,
                msg.text
            )
 
        time_str = msg.timestamp.strftime(
            "%H:%M"
        )
 
        with chat_placeholder:
 
            st.markdown(
                f"""
<div class="chat-card">
<div class="{message_class}">
<div class="{sender_class}">{sender_title}</div>
<div class="message-text">{display_text}</div>
<div class="message-time">{time_str}</div>
</div>
</div>
""",
                unsafe_allow_html=True
            )
 
            # =============================================
            # AI ANALYSIS
            # =============================================
 
            if not is_source:
 
                with st.expander(
                    f"🤖 View AI Threat Analysis — {msg.id}"
                ):
 
                    a, b, c = st.columns(3)
 
                    with a:
 
                        st.metric(
                            "Semantic Drift",
                            f"{semantic_drift:.2f}"
                        )
 
                    with b:
 
                        st.metric(
                            "Exaggeration",
                            f"{exaggeration:.2f}"
                        )
 
                    with c:
 
                        st.metric(
                            "Mutation Velocity",
                            f"{mutation_rate:.2f}"
                        )
 
                    st.caption(
                        "Red highlighted text represents "
                        "content introduced during propagation."
                    )
 
        # =====================================================
        # ANIMATION DELAY
        # =====================================================
 
        time.sleep(1.2)
 
    # =========================================================
    # FINAL RESULT
    # =========================================================
 
    st.divider()
 
    if st.session_state.last_risk >= 0.7:
 
        st.error(
            "🚨 High-risk misinformation detected. "
            "The latest message shows significant semantic "
            "mutation and propagation risk."
        )
 
    elif st.session_state.last_risk >= 0.3:
 
        st.warning(
            "⚠️ The message chain contains suspicious mutations. "
            "Further verification is recommended."
        )
 
    else:
 
        st.success(
            "✅ The message chain currently remains relatively low-risk."
        )
 
    st.markdown(
        '<div class="section-title">🔬 How the System Works</div>',
        unsafe_allow_html=True
    )
 
    st.info(
        """
        The system compares each forwarded message with
        previous messages, extracts NLP and graph-based
        features, and sends the resulting feature vector
        to an XGBoost classifier to estimate misinformation risk.
        """
    )
 
 
# =========================================================
# EMPTY DASHBOARD
# =========================================================
 
else:
 
    with dashboard_placeholder.container():
 
        st.markdown(
            """
<div class="ready-card">
<div class="ready-icon">🧠</div>
<div class="ready-title">Ready to Analyze</div>
<div class="ready-text">
Select a scenario from the sidebar and start
the propagation analysis to see how information
changes across the network.
</div>
</div>
""",
            unsafe_allow_html=True
        )
 
        st.markdown(
            '<div class="section-title">System Capabilities</div>',
            unsafe_allow_html=True
        )
 
        cap1, cap2, cap3 = st.columns(3)
 
        with cap1:
 
            st.markdown(
                """
<div class="capability-card">
<div class="capability-title">🧬 NLP Analysis</div>
<div class="capability-text">
Transformer models analyze semantic meaning,
sentiment, exaggeration and language patterns.
</div>
</div>
""",
                unsafe_allow_html=True
            )
 
        with cap2:
 
            st.markdown(
                """
<div class="capability-card">
<div class="capability-title">🌐 Graph Analysis</div>
<div class="capability-text">
NetworkX models propagation depth and
forwarding relationships between messages.
</div>
</div>
""",
                unsafe_allow_html=True
            )
 
        with cap3:
 
            st.markdown(
                """
<div class="capability-card">
<div class="capability-title">🤖 ML Risk Scoring</div>
<div class="capability-text">
XGBoost combines engineered features to
estimate misinformation propagation risk.
</div>
</div>
""",
                unsafe_allow_html=True
            )
 
 
# =========================================================
# FOOTER
# =========================================================
 
st.markdown(
    """
<div class="footer">
<b>TruthTrace AI</b> • Misinformation Drift Detection
<br>
Made by Puja Barman
<br><br>
NLP • Transformers • SentenceTransformers •
NetworkX • XGBoost • Streamlit
</div>
""",
    unsafe_allow_html=True
)
 
