import streamlit as st
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
import io

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="OPD Queue Simulator",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS  – clean hospital / medical feel
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

/* ── global ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

/* ── background ── */
.stApp { background: #f0f4f8; }

/* ── sidebar ── */
[data-testid="stSidebar"] {
    background: #0a2342 !important;
    border-right: 2px solid #1a3a5c;
}
[data-testid="stSidebar"] * { color: #c8d8e8 !important; }
[data-testid="stSidebar"] .stSlider > label { color: #7fb3d3 !important; font-weight: 500; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #ffffff !important; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0a2342 0%, #1a5276 60%, #1abc9c 100%);
    border-radius: 16px;
    padding: 2.4rem 2.8rem;
    color: #fff;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 32px rgba(10,35,66,.18);
}
.hero h1 { font-size: 2rem; font-weight: 600; margin: 0 0 .3rem; letter-spacing: -.5px; }
.hero p  { font-size: .95rem; opacity: .78; margin: 0; }

/* ── metric cards ── */
.metric-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 1.3rem 1.4rem;
    box-shadow: 0 2px 12px rgba(0,0,0,.07);
    border-left: 4px solid #1a5276;
    margin-bottom: .5rem;
}
.metric-label { font-size: .78rem; font-weight: 500; color: #6b7c93; text-transform: uppercase; letter-spacing: .8px; }
.metric-value { font-size: 2rem; font-weight: 600; color: #0a2342; font-family: 'DM Mono', monospace; }
.metric-unit  { font-size: .85rem; color: #6b7c93; }

/* accent colours per card */
.card-teal  { border-left-color: #1abc9c; }
.card-amber { border-left-color: #f39c12; }
.card-red   { border-left-color: #e74c3c; }
.card-blue  { border-left-color: #2980b9; }

/* ── section headers ── */
.section-title {
    font-size: 1.05rem; font-weight: 600; color: #0a2342;
    border-bottom: 2px solid #d0e3f0; padding-bottom: .45rem; margin: 1.6rem 0 1rem;
}

/* ── status banner ── */
.status-ok   { background:#d5f5e3; border-left:4px solid #1abc9c; color:#145a32; padding:.8rem 1rem; border-radius:8px; font-weight:500; }
.status-warn { background:#fef9e7; border-left:4px solid #f39c12; color:#7d6608; padding:.8rem 1rem; border-radius:8px; font-weight:500; }
.status-bad  { background:#fdedec; border-left:4px solid #e74c3c; color:#922b21; padding:.8rem 1rem; border-radius:8px; font-weight:500; }

/* ── run button ── */
.stButton > button {
    background: linear-gradient(90deg, #0a2342, #1a5276) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: .65rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    width: 100%;
    transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }

/* ── dataframe ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── chart containers ── */
.chart-box {
    background: #fff; border-radius: 12px;
    padding: 1.2rem; box-shadow: 0 2px 12px rgba(0,0,0,.07);
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  MATPLOTLIB THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "#ffffff",
    "figure.facecolor": "#ffffff",
    "axes.labelcolor": "#0a2342",
    "xtick.color": "#6b7c93",
    "ytick.color": "#6b7c93",
    "axes.titlecolor": "#0a2342",
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "grid.color": "#e8edf2",
    "grid.linewidth": 0.7,
})

COLORS = {
    "teal":  "#1abc9c",
    "blue":  "#2980b9",
    "navy":  "#0a2342",
    "amber": "#f39c12",
    "red":   "#e74c3c",
    "light": "#d0e3f0",
}

# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("#### ⚙️ Parameters")

    num_patients = st.slider("👥 Number of Patients", 20, 300, 80, step=10)
    st.markdown("")

    arrival_rate_hr = st.slider("🚶 Patient Arrival Rate λ (patients/hour)", 1, 60, 12)
    arrival_rate = arrival_rate_hr / 60

    service_rate_hr = st.slider("🩺 Service Rate μ (patients/hour per doctor)", 1, 60, 8)
    service_rate = service_rate_hr / 60
    service_interval = 60 / service_rate_hr

    num_doctors = st.slider("👨‍⚕️ Number of Doctors", 1, 8, 2)

    st.markdown("---")
    st.markdown(
        f"<small>Traffic intensity ρ = **{arrival_rate / (num_doctors * service_rate):.2f}**<br>"
        f"<i>(system overloads when ρ ≥ 1)</i></small>",
        unsafe_allow_html=True,
    )
    st.markdown("")
    run = st.button("▶  Run Simulation")

# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>🏥 Hospital Queue Simulation</h1>
  <p>Mathematical Modelling · M/M/c Queue Theory · S.Y. BSc IT — MMCA Project</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SIMULATION FUNCTION
# ─────────────────────────────────────────────
def simulate(num_patients, arrival_rate, service_rate, num_doctors):
    random.seed(42)

    # Step 1: generate all arrival times upfront
    arrival_times = []
    t = 0.0
    for _ in range(num_patients):
        t += random.expovariate(arrival_rate)
        arrival_times.append(t)

    # Step 2: assign each patient to a doctor
    doctor_free = [0.0] * num_doctors
    service_start_times = []
    waiting_times = []
    service_times = []
    departure_times = []

    for i in range(num_patients):
        arr = arrival_times[i]
        next_doc = doctor_free.index(min(doctor_free))
        start = max(arr, doctor_free[next_doc])
        wait = start - arr
        svc = random.expovariate(service_rate)
        depart = start + svc
        doctor_free[next_doc] = depart

        service_start_times.append(start)
        waiting_times.append(wait)
        service_times.append(svc)
        departure_times.append(depart)

    # Step 3: calculate true queue length at each arrival moment
    # Lq = patients who arrived before t but haven't started service yet
    queue_length_over_time = []
    for i in range(num_patients):
        t = arrival_times[i]
        lq = sum(
            1 for j in range(i)
            if arrival_times[j] <= t and service_start_times[j] > t
        )
        queue_length_over_time.append(lq)

    # Step 4: build dataframe — divide by 60 to convert minutes → hours
    records = []
    for i in range(num_patients):
        records.append({
            "Patient": i + 1,
            "Arrival (hr)": round(arrival_times[i] / 60, 4),
            "Service Start (hr)": round(service_start_times[i] / 60, 4),
            "Waiting Time (hr)": round(waiting_times[i] / 60, 4),
            "Consultation (hr)": round(service_times[i] / 60, 4),
            "Departure (hr)": round(departure_times[i] / 60, 4),
        })

    df = pd.DataFrame(records)
    total_time = max(departure_times)
    total_svc = sum(service_times)
    utilization = total_svc / (num_doctors * total_time)
    avg_wait = np.mean(waiting_times) / 60   # in hours
    max_wait = np.max(waiting_times) / 60    # in hours
    avg_queue = np.mean(queue_length_over_time)

    return df, avg_wait, max_wait, utilization, avg_queue, queue_length_over_time

# ─────────────────────────────────────────────
#  PLACEHOLDER BEFORE RUN
# ─────────────────────────────────────────────
if not run:
    st.info("👈  Adjust parameters in the sidebar and click **Run Simulation** to begin.")
    st.stop()

# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
df, avg_wait, max_wait, utilization, avg_queue, q_trend = simulate(
    num_patients, arrival_rate, service_rate, num_doctors
)

rho = arrival_rate / (num_doctors * service_rate)

# ─────────────────────────────────────────────
#  METRICS ROW
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Key Performance Indicators</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card card-teal">
      <div class="metric-label">Avg Waiting Time</div>
      <div class="metric-value">{avg_wait:.1f}</div>
      <div class="metric-unit">hours</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card card-red">
      <div class="metric-label">Max Waiting Time</div>
      <div class="metric-value">{max_wait:.1f}</div>
      <div class="metric-unit">hours</div>
    </div>""", unsafe_allow_html=True)

with c3:
    util_pct = utilization * 100
    card_cls = "card-teal" if util_pct < 70 else "card-amber" if util_pct < 90 else "card-red"
    st.markdown(f"""
    <div class="metric-card {card_cls}">
      <div class="metric-label">Doctor Utilization</div>
      <div class="metric-value">{util_pct:.0f}</div>
      <div class="metric-unit">percent</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card card-blue">
      <div class="metric-label">Avg Queue Length</div>
      <div class="metric-value">{avg_queue:.1f}</div>
      <div class="metric-unit">patients</div>
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  STATUS BANNER
# ─────────────────────────────────────────────
st.markdown("")
if rho >= 1:
    st.markdown('<div class="status-bad">⚠️ System Overloaded — Traffic intensity ρ ≥ 1. Queue grows indefinitely. Add more doctors immediately.</div>', unsafe_allow_html=True)
elif util_pct > 80:
    st.markdown('<div class="status-warn">⚡ High Load — Doctors are heavily utilised. Consider adding 1 more doctor during peak hours.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-ok">✅ System Healthy — Waiting times and utilisation are within acceptable limits.</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  CHARTS  (2 × 2 grid)
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📈 Simulation Graphs</div>', unsafe_allow_html=True)

left, right = st.columns(2)

# ── Chart 1: Waiting time per patient ──
with left:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.4))
    ax1.fill_between(df["Patient"], df["Waiting Time (hr)"], alpha=.18, color=COLORS["teal"])
    ax1.plot(df["Patient"], df["Waiting Time (hr)"], color=COLORS["teal"], linewidth=1.6)
    ax1.axhline(avg_wait, color=COLORS["amber"], linewidth=1.2, linestyle="--", label=f"Avg {avg_wait:.2f} hr")
    ax1.set_title("Waiting Time per Patient")
    ax1.set_xlabel("Patient Number")
    ax1.set_ylabel("Wait (hr)")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig1)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Chart 2: Doctor utilization bar ──
with right:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.4))
    cats = ["Utilized", "Idle"]
    vals = [util_pct, 100 - util_pct]
    bar_colors = [COLORS["blue"], COLORS["light"]]
    bars = ax2.bar(cats, vals, color=bar_colors, width=0.45, edgecolor="white")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                 f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="600", color=COLORS["navy"])
    ax2.set_ylim(0, 115)
    ax2.set_title(f"Doctor Utilization  ({num_doctors} Doctor{'s' if num_doctors>1 else ''})")
    ax2.set_ylabel("Percentage (%)")
    ax2.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

left2, right2 = st.columns(2)

# ── Chart 3: Queue length over time ──
with left2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.4))
    ax3.fill_between(range(1, num_patients + 1), q_trend, alpha=.2, color=COLORS["amber"])
    ax3.plot(q_trend, color=COLORS["amber"], linewidth=1.6)
    ax3.set_title("Queue Length Over Time")
    ax3.set_xlabel("Patient Sequence")
    ax3.set_ylabel("Patients Waiting")
    ax3.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig3)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Chart 4: Arrival vs Service timeline ──
with right2:
    st.markdown('<div class="chart-box">', unsafe_allow_html=True)
    fig4, ax4 = plt.subplots(figsize=(5.5, 3.4))
    ax4.plot(df["Patient"], df["Arrival (hr)"], label="Arrival", color=COLORS["navy"], linewidth=1.5)
    ax4.plot(df["Patient"], df["Service Start (hr)"], label="Service Start",
             color=COLORS["teal"], linewidth=1.5, linestyle="--")
    ax4.fill_between(df["Patient"], df["Arrival (hr)"], df["Service Start (hr)"],
                     alpha=.12, color=COLORS["red"], label="Gap = Wait")
    ax4.set_title("Arrival vs Service Start Timeline")
    ax4.set_xlabel("Patient Number")
    ax4.set_ylabel("Time (hr)")
    ax4.legend(fontsize=8)
    ax4.grid(axis="y")
    plt.tight_layout()
    st.pyplot(fig4)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DATA TABLE
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📋 Patient Data Table</div>', unsafe_allow_html=True)

with st.expander("Show full patient table", expanded=False):
    st.dataframe(
        df,
        use_container_width=True,
        height=320,
        hide_index=True,
        column_config={col: st.column_config.NumberColumn(col, help=None) for col in df.columns if col != "Patient"},
    )

# ─────────────────────────────────────────────
#  DOWNLOAD BUTTON
# ─────────────────────────────────────────────
csv_buffer = io.StringIO()
df.to_csv(csv_buffer, index=False)
st.download_button(
    label="⬇️  Download Results as CSV",
    data=csv_buffer.getvalue(),
    file_name="opd_simulation_results.csv",
    mime="text/csv",
)

# ─────────────────────────────────────────────
#  ANALYSIS SECTION
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">🧠 System Analysis & Insights</div>', unsafe_allow_html=True)

ia, ib = st.columns(2)

with ia:
    st.markdown("**What the model tells us**")
    st.markdown(f"""
- **Traffic intensity (ρ)** = `{rho:.3f}` — {'system is stable ✅' if rho < 1 else 'system is unstable ⚠️'}
- Each doctor handles **{service_rate * 60:.1f} patients/hour**
- Total system capacity = **{num_doctors * service_rate * 60:.1f} patients/hour**
- Patients arrive at **{arrival_rate * 60:.1f} patients/hour**
- {num_patients} patients simulated — avg wait = **{avg_wait:.2f} hr**
""")

with ib:
    st.markdown("**Recommendation**")
    if rho >= 1:
        needed = int(np.ceil(arrival_rate / service_rate)) + 1
        st.markdown(f"""
❌ With **{num_doctors} doctor(s)**, the system cannot keep up.  
Minimum doctors needed = **{needed}** to stabilise.  
Increase doctors or reduce patient load during peak hours.
""")
    elif util_pct > 80:
        st.markdown(f"""
⚡ Doctors are under significant load ({util_pct:.0f}% utilized).  
Adding **1 more doctor** would reduce waiting time substantially.  
Peak-hour staffing should be reviewed.
""")
    else:
        st.markdown(f"""
✅ Current setup is efficient.  
Doctors are **{util_pct:.0f}% utilized** — a healthy balance.  
No immediate changes needed. Monitor during peak periods.
""")

# ─────────────────────────────────────────────
#  MATHEMATICAL MODEL
# ─────────────────────────────────────────────
st.markdown('<div class="section-title">📐 Mathematical Model</div>', unsafe_allow_html=True)

with st.expander("📐 M/M/c Queue — Formulas & Explanation", expanded=True):
    st.markdown("""
This simulation is based on the **M/M/c Queue Model** — a standard mathematical model used to study waiting lines where:
- Patients arrive **randomly** (Poisson process)
- Each consultation takes a **random** amount of time (Exponential distribution)
- There are **c doctors** serving patients simultaneously
""")

    f1, f2, f3 = st.columns(3)

    with f1:
        st.markdown("""
<div style="background:#f8fbff; border:1.5px solid #d0e3f0; border-radius:10px; padding:1.1rem 1.2rem; text-align:center;">
  <div style="font-size:.78rem; font-weight:600; color:#6b7c93; text-transform:uppercase; letter-spacing:.8px; margin-bottom:.5rem;">Traffic Intensity</div>
  <div style="font-size:1.6rem; color:#0a2342; font-family:'Georgia', serif; margin-bottom:.6rem;">ρ = λ / (c × μ)</div>
  <div style="font-size:.82rem; color:#6b7c93;">Measures how loaded the system is.<br>Must be &lt; 1 for a stable queue.<br><b>λ</b> = arrival rate &nbsp;|&nbsp; <b>μ</b> = service rate &nbsp;|&nbsp; <b>c</b> = doctors</div>
</div>
""", unsafe_allow_html=True)

    with f2:
        st.markdown("""
<div style="background:#f8fbff; border:1.5px solid #d0e3f0; border-radius:10px; padding:1.1rem 1.2rem; text-align:center;">
  <div style="font-size:.78rem; font-weight:600; color:#6b7c93; text-transform:uppercase; letter-spacing:.8px; margin-bottom:.5rem;">Avg Waiting Time in Queue</div>
  <div style="font-size:1.6rem; color:#0a2342; font-family:'Georgia', serif; margin-bottom:.6rem;">W<sub>q</sub> = L<sub>q</sub> / λ</div>
  <div style="font-size:.82rem; color:#6b7c93;">Average time a patient waits <i>before</i> seeing a doctor.<br><b>L<sub>q</sub></b> = avg number of patients in queue</div>
</div>
""", unsafe_allow_html=True)

    with f3:
        st.markdown("""
<div style="background:#f8fbff; border:1.5px solid #d0e3f0; border-radius:10px; padding:1.1rem 1.2rem; text-align:center;">
  <div style="font-size:.78rem; font-weight:600; color:#6b7c93; text-transform:uppercase; letter-spacing:.8px; margin-bottom:.5rem;">Total Time in System</div>
  <div style="font-size:1.6rem; color:#0a2342; font-family:'Georgia', serif; margin-bottom:.6rem;">W = W<sub>q</sub> + 1/μ</div>
  <div style="font-size:.82rem; color:#6b7c93;">Total time from arrival to leaving.<br><b>1/μ</b> = avg consultation duration</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
**Current simulation values:**
- λ = `{arrival_rate_hr}` patients/hr &nbsp;|&nbsp; μ = `{service_rate_hr}` patients/hr &nbsp;|&nbsp; c = `{num_doctors}` doctors
- ρ = `{rho:.3f}` → {"✅ Stable system" if rho < 1 else "⚠️ Unstable — queue grows without bound"}
- Avg waiting time (Wq) = **{avg_wait:.2f} hr** &nbsp;|&nbsp; Total time (W) = **{(avg_wait + service_interval/60):.2f} hr**
""")

# ─────────────────────────────────────────────
#  KEY TERMS
# ─────────────────────────────────────────────
with st.expander("📖 Key Terms & Definitions"):
    st.markdown("""
| Term | Meaning |
|---|---|
| **M/M/c** | Queue model — Markov arrivals / Markov service / c servers |
| **λ (lambda)** | Patient arrival rate (patients per hour) |
| **μ (mu)** | Doctor service rate (patients served per hour) |
| **ρ (rho)** | Traffic intensity = λ / (c × μ) — measures system load |
| **Wq** | Average time a patient spends waiting before being seen |
| **Lq** | Average number of patients waiting in queue at any moment |
| **Poisson distribution** | Statistical model for random arrival events |
| **Exponential distribution** | Statistical model for random service durations |
| **Server** | A doctor or staff member providing service |
| **Queue** | The waiting line formed when all doctors are busy |
""")

st.markdown("<br><small style='color:#aaa;'>MMCA Project · S.Y. BSc IT Sem IV · M/M/c Queue Simulation</small>", unsafe_allow_html=True)
