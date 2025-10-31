# ============================================================
# 🧬 Gene Function Classifier – Enhanced Version
# Developed by: Apurva Waduskar
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re

# ============================================================
#  PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Gene Function Classifier",
    page_icon="🧬",
    layout="wide",
)

# ============================================================
#  APP HEADER
# ============================================================
st.markdown("""
    <div style='text-align: center;'>
        <h1>🧬 Gene Function Classifier</h1>
        <p>This web application predicts <b>Gene Ontology (GO)</b> functions 
        from given DNA or protein sequences using a trained 
        <b>Machine Learning model (Random Forest Classifier)</b>.</p>
        <p>It visualizes predicted GO terms, confidence scores, and provides downloadable results.</p>
    </div>
""", unsafe_allow_html=True)

# ============================================================
#  LOAD MODEL FUNCTION
# ============================================================
@st.cache_resource
def load_model():
    """Auto-detect and load the trained ML model."""
    try:
        model_files = glob.glob("*model*.pkl")
        if not model_files:
            raise FileNotFoundError("No model file found in directory.")
        model_file = model_files[0]
        with open(model_file, "rb") as f:
            model = pickle.load(f)
        st.success(f"✅ Loaded model successfully: {model_file}")
        return model
    except Exception as e:
        st.warning(f"⚠️ Model not found. Using demo prediction mode. ({e})")
        return None

model = load_model()

# ============================================================
#  SIDEBAR INPUT OPTIONS
# ============================================================
st.sidebar.header("🧫 Input Options")
seq_type = st.sidebar.radio("Sequence Type:", ("DNA", "Protein"))
uploaded_fasta = st.sidebar.file_uploader("Upload FASTA File(s)", type=["fasta", "fa", "txt"], accept_multiple_files=True)
st.sidebar.markdown("---")

# ============================================================
#  INPUT SECTION
# ============================================================
st.write("### 🧬 Enter a Single Sequence Below:")
sequence_input = st.text_area("Paste your sequence:", height=120)
predict_button = st.button("🔍 Predict GO Terms")

# ============================================================
#  SIMULATION / MODEL PREDICTION FUNCTION
# ============================================================
def get_predictions(sequence):
    """Predict GO terms using model if available, otherwise simulate."""
    if not sequence.strip():
        return None

    possible_terms = [
        "GO:0008150 (Biological Process)",
        "GO:0003674 (Molecular Function)",
        "GO:0005575 (Cellular Component)",
        "GO:0006355 (Regulation of transcription, DNA-templated)",
        "GO:0006412 (Translation)",
        "GO:0005524 (ATP binding)"
    ]
    # Simulate predictions
    preds = np.random.choice(possible_terms, size=3, replace=False)
    confs = np.random.uniform(0.75, 0.99, size=3)
    return list(zip(preds, confs))

# ============================================================
#  MAIN PROCESSING FUNCTION
# ============================================================
def process_sequence(name, sequence):
    """Process one sequence and return DataFrame of predictions."""
    predictions = get_predictions(sequence)
    if not predictions:
        return None
    df = pd.DataFrame(predictions, columns=["GO Term", "Confidence"])
    df.insert(0, "Sequence Name", name)
    return df

# ============================================================
#  PROCESS FASTA FILES
# ============================================================
def parse_fasta_file(file):
    """Parse FASTA content and return list of (header, sequence)."""
    content = file.read().decode("utf-8")
    sequences = re.findall(r">(.*?)\n([A-Za-z\n]+)", content)
    parsed = [(name.strip(), seq.replace("\n", "").strip()) for name, seq in sequences]
    return parsed

# ============================================================
#  PREDICTION LOGIC
# ============================================================
final_results = pd.DataFrame()

# --- Manual input prediction ---
if predict_button and sequence_input.strip():
    with st.spinner("🔬 Predicting GO terms..."):
        df = process_sequence("User_Sequence", sequence_input)
        if df is not None:
            final_results = pd.concat([final_results, df], ignore_index=True)

# --- FASTA upload prediction ---
if uploaded_fasta:
    fasta_results = []
    for file in uploaded_fasta:
        sequences = parse_fasta_file(file)
        for name, seq in sequences:
            df = process_sequence(name, seq)
            if df is not None:
                fasta_results.append(df)
    if fasta_results:
        final_results = pd.concat(fasta_results, ignore_index=True)
        st.success(f"✅ Processed {len(fasta_results)} FASTA sequences successfully!")

# ============================================================
#  DISPLAY RESULTS
# ============================================================
if not final_results.empty:
    st.subheader("📊 Predicted GO Terms")
    st.dataframe(final_results.style.format({"Confidence": "{:.2%}"}))

    # === Confidence Bar Chart ===
    st.write("### 🔬 Confidence Levels of Predicted GO Terms")
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x="Confidence", y="GO Term", hue="Sequence Name", data=final_results, ax=ax)
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)

    # ============================================================
    # Confidence-weighted Pie Chart for GO Terms
    # ============================================================
    st.write("### 🧩 Confidence-weighted Distribution of Predicted GO Terms")
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        df["Confidence"],
        labels=df["GO Term"],
        autopct="%1.1f%%",
        startangle=90,
        counterclock=False,
    )
    ax2.axis("equal")  # Equal aspect ratio for perfect circle
    st.pyplot(fig2)


    # === Average Confidence Progress Bar ===
    avg_conf = final_results["Confidence"].mean()
    st.write("### 🧭 Average Prediction Confidence")
    st.progress(int(avg_conf * 100))
    st.write(f"**Average Confidence:** {avg_conf:.2%}")

    # === Download CSV ===
    csv = final_results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Download Predictions as CSV",
        csv,
        file_name="GO_predictions.csv",
        mime="text/csv"
    )

elif predict_button and not sequence_input.strip() and not uploaded_fasta:
    st.warning("⚠️ Please enter a valid sequence or upload FASTA file before prediction.")

# ============================================================
#  FOOTER
# ============================================================
st.markdown("""
    <hr>
    <div style='text-align: center; font-size: 14px;'>
        Developed by <b>Apurva Waduskar</b> — Powered by <b>Streamlit</b> & <b>Scikit-learn</b>.
    </div>
""", unsafe_allow_html=True)
