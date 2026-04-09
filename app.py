import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="IDS System", layout="wide")

# 🪖 Military Theme Styling
import base64

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("background.jpg")

st.markdown(f"""
<style>

/* 🔥 REMOVE WHITE HEADER + TOOLBAR */
header {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0) !important;
}}

[data-testid="stToolbar"] {{
    background: rgba(0,0,0,0) !important;
}}

[data-testid="stAppViewContainer"] {{
    background: none;
}}

/* 🌍 BACKGROUND */
.stApp {{
    background: linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)),
                url("data:image/png;base64,{img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* 🟢 HEADINGS */
h1, h2, h3 {{
    color: #00ff9c !important;
    text-shadow: 0 0 10px #00ff9c;
}}

/* TEXT */
p, label, span {{
    color: #e0ffe0 !important;
    font-weight: 500;
}}

div {{
    color: inherit !important;
}}

label {{
    color: #b6ffb6 !important;
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background: rgba(0, 20, 0, 0.95);
    border-right: 2px solid #00ff9c;
}}

/* BUTTONS */
.stButton>button {{
    background: linear-gradient(135deg, #00ff9c, #004d2b);
    color: black;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 0 15px #00ff9c;
}}

/* TABLE */
.stDataFrame {{
    background: rgba(0,0,0,0.6) !important;
    color: #00ff9c !important;
    border: 1px solid #00ff9c;
    border-radius: 10px;
}}

thead tr th {{
    background-color: rgba(0, 255, 156, 0.2) !important;
    color: #00ff9c !important;
    font-weight: bold;
}}

tbody tr td {{
    background-color: rgba(0,0,0,0.5) !important;
    color: #a8ffdb !important;
}}

tbody tr:hover {{
    background-color: rgba(0,255,156,0.1) !important;
}}

/* DOWNLOAD BUTTON */
.stDownloadButton>button {{
    background: linear-gradient(135deg, #00ff9c, #004d2b);
    color: black !important;
    font-weight: bold;
    border-radius: 10px;
    box-shadow: 0 0 10px #00ff9c;
}}

/* UPLOAD */
[data-testid="stFileUploader"] {{
    background: rgba(0,0,0,0.25);
    border: 1px solid #00ff9c;
    border-radius: 12px;
    backdrop-filter: blur(10px);
}}

[data-testid="stFileUploader"] label {{
    color: #ffffff !important;
    font-weight: 600;
}}

[data-testid="stFileUploader"] small {{
    color: #b6ffb6 !important;
}}

[data-testid="stFileUploader"] button {{
    background: linear-gradient(135deg, #00ff9c, #004d2b) !important;
    color: black !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    padding: 10px 20px !important;
    border: none !important;
    box-shadow: 0 0 10px #00ff9c, 0 0 20px #00ff9c;
    transition: all 0.3s ease;
}}

[data-testid="stFileUploader"] button:hover {{
    box-shadow: 0 0 20px #00ff9c, 0 0 40px #00ff9c;
    transform: scale(1.05);
}}

[data-testid="stFileUploader"] div {{
    background: transparent !important;
}}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD MODELS
# -------------------------------
rf_model = pickle.load(open("rf_model.pkl", "rb"))
bilstm_model = load_model("bilstm_model.h5")


# -------------------------------
# MODEL ACCURACY (ADD THIS)
# -------------------------------
rf_accuracy = 0.9985
  # replace with your actual RF accuracy
bilstm_accuracy = 0.99  # replace with your BiLSTM accuracy
scaler = pickle.load(open("scaler.pkl", "rb"))
encoders = pickle.load(open("encoders.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))



# -------------------------------
# 🆕 ATTACK INFO (NEW)
# -------------------------------
attack_info = {
    "normal": {
        "desc": "Normal network traffic with no malicious behavior. All activities follow expected patterns and no anomaly is detected.",
        "solution": "No action required. Continue monitoring system performance and logs regularly."
    },

    "dos": {
        "desc": "Denial of Service (DoS) attack floods the target system with excessive traffic, exhausting resources like bandwidth, CPU, or memory. This makes the system unavailable to legitimate users.",
        "solution": "Use firewalls, enable rate limiting, deploy intrusion prevention systems (IPS), and block suspicious IP addresses."
    },

    "probe": {
        "desc": "Probe attacks are reconnaissance attacks where attackers scan networks to identify open ports, services, and vulnerabilities. This is usually a preparation step before launching a major attack.",
        "solution": "Enable network monitoring, block suspicious scanning IPs, use IDS alerts, and disable unused ports/services."
    },

    "r2l": {
        "desc": "Remote to Local (R2L) attack occurs when an attacker tries to gain access to a system remotely without having an account, often using password guessing or exploiting vulnerabilities.",
        "solution": "Use strong authentication, enable multi-factor authentication (MFA), monitor login attempts, and disable unused services."
    },

    "u2r": {
        "desc": "User to Root (U2R) attack occurs when a normal user gains root/admin privileges by exploiting system vulnerabilities. This gives full control over the system.",
        "solution": "Apply security patches regularly, restrict root access, monitor privilege escalation, and audit system logs."
    },

    "other": {
        "desc": "This category includes unknown or rare attack patterns that do not match predefined attack types. These could represent new or evolving threats.",
        "solution": "Investigate logs, update detection rules, and apply adaptive security mechanisms."
    }
}

# -------------------------------
# TITLE
# -------------------------------
st.title("🛡️ Military Intrusion Detection System")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("⚙️ Control Panel")

model_choice = st.sidebar.selectbox(
    "Select Detection Model",
    ["Random Forest", "BiLSTM"]
)

st.sidebar.markdown("## 🧠 System Info")
st.sidebar.info("""
- 🪖 Defense-grade monitoring  
- 🌐 Network intrusion detection  
- 🤖 AI-powered classification  
""")

# -------------------------------
# USER INPUT
# -------------------------------
def user_input():
    data = {}

    data['protocol_type'] = st.sidebar.selectbox(
        "Protocol", encoders['protocol_type'].classes_
    )
    data['service'] = st.sidebar.selectbox(
        "Service", encoders['service'].classes_
    )
    data['flag'] = st.sidebar.selectbox(
        "Flag", encoders['flag'].classes_
    )

    for feature in [
        "duration","src_bytes","dst_bytes","land","wrong_fragment","urgent",
        "hot","num_failed_logins","logged_in","num_compromised","root_shell",
        "su_attempted","num_root","num_file_creations","num_shells",
        "num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]:
        data[feature] = st.sidebar.number_input(feature, value=0.0)

    return data

input_data = user_input()

# -------------------------------
# PREPROCESS
# -------------------------------
def preprocess(data):
    row = []

    for col in [
        "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
        "wrong_fragment","urgent","hot","num_failed_logins","logged_in",
        "num_compromised","root_shell","su_attempted","num_root","num_file_creations",
        "num_shells","num_access_files","num_outbound_cmds","is_host_login",
        "is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
        "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate",
        "srv_diff_host_rate","dst_host_count","dst_host_srv_count",
        "dst_host_same_srv_rate","dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
        "dst_host_serror_rate","dst_host_srv_serror_rate",
        "dst_host_rerror_rate","dst_host_srv_rerror_rate"
    ]:
        if col in encoders:
            row.append(encoders[col].transform([data[col]])[0])
        else:
            row.append(data[col])

    return np.array(row).reshape(1, -1)

# Show Accuracy on Main Screen
st.markdown("### 📊 Model Performance")

if model_choice == "Random Forest":
    st.metric("📈 Accuracy", f"{rf_accuracy * 100:.2f}%")
else:
    st.metric("📈 Accuracy", f"{bilstm_accuracy * 100:.2f}%")

# -------------------------------
# DETECTION
# -------------------------------
if st.button("🎯 Run Threat Detection"):
    processed = preprocess(input_data)
    scaled = scaler.transform(processed)

    if model_choice == "Random Forest":
        prediction = rf_model.predict(scaled)[0]
        probs = rf_model.predict_proba(scaled)[0]
    else:
        scaled = scaled.reshape(1, 1, scaled.shape[1])
        probs = bilstm_model.predict(scaled)[0]
        prediction = np.argmax(probs)

    attack_type = label_encoder.inverse_transform([prediction])[0]

    col1, col2 = st.columns(2)
    col1.metric("🧠 Model", model_choice)
    col2.metric("🎯 Result", attack_type.upper())

    if attack_type == "normal":
        st.success("✅ System Secure")
    else:
        st.error(f"⚠️ Threat Detected: {attack_type.upper()}")

    prob_df = pd.DataFrame({
        "Attack Type": label_encoder.classes_,
        "Probability": probs
    })

    st.markdown("### 📊 Threat Probability Analysis")
    st.bar_chart(prob_df.set_index("Attack Type"))

# -------------------------------
# BULK DETECTION
# -------------------------------
st.markdown("## 📁 Bulk Threat Analysis")

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.dataframe(df.head())

    try:
        for col in ['protocol_type', 'service', 'flag']:
            df[col] = encoders[col].transform(df[col])

        scaled = scaler.transform(df)

        if model_choice == "Random Forest":
            preds = rf_model.predict(scaled)
        else:
            scaled = scaled.reshape(df.shape[0], 1, df.shape[1])
            preds = np.argmax(bilstm_model.predict(scaled), axis=1)

        df['Attack Type'] = label_encoder.inverse_transform(preds)
        
        
        df['Status'] = df['Attack Type'].apply(
            lambda x: "⚠️ Threat" if x != "normal" else "✅ Secure"
        )

        st.markdown("### 🔍 Detection Results")
        st.dataframe(df)

        summary = df['Attack Type'].value_counts()

        st.markdown("### 📊 Attack Distribution")
        cols = st.columns(len(summary))

        for i, (attack, count) in enumerate(summary.items()):
            with cols[i]:
                st.metric(attack.upper(), count)

        st.bar_chart(summary)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Report", csv, "report.csv", "text/csv")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# -------------------------------
# 🆕 ATTACK DASHBOARD (NEW)
# -------------------------------
st.markdown("## 📊 Attack Intelligence Panel")

cols = st.columns(3)

for i, (attack, info) in enumerate(attack_info.items()):
    with cols[i % 3]:
        st.markdown(f"""
        <div style="background: rgba(0,0,0,0.6); padding: 20px; border-radius: 15px; margin-bottom: 25px;">
            <h3>⚠️ {attack.upper()}</h3>
            <p><b>Description:</b> {info['desc']}</p>
            <p><b>Solution:</b> {info['solution']}</p>
        </div>
        """, unsafe_allow_html=True)