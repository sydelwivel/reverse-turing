# app.py
import streamlit as st
import pandas as pd
import random
import json

from analytics import (
    binomial_test,
    kl_divergence,
    js_divergence,
    earth_movers_distance,
    train_meta_classifier,
)
from personas import bias_personas
from counterfactuals import generate_counterfactuals
from ai_mock import ai_mock_score
from db import save_run_result
import data_generator
import report
import mitigation

# -----------------------------
# Load resumes (cached)
# -----------------------------
@st.cache_data
def load_resumes():
    try:
        df = pd.read_csv("assets/sample_resumes.csv")
    except FileNotFoundError:
        df = data_generator.generate_synthetic(20, "assets/sample_resumes.csv")
    return df

resumes_df = load_resumes()

# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(page_title="Reverse Turing Test for Bias", layout="wide")
st.title("🤖 Reverse Turing Test for Bias in AI Hiring Models")

st.markdown("""
This interactive prototype allows you to test whether users can detect **biased scoring personas** 
compared to an (AI-like) model. It also runs analytics, compliance reporting, and mitigation.
""")

# -----------------------------
# Session state
# -----------------------------
if "trials" not in st.session_state:
    st.session_state.trials = []
if "results" not in st.session_state:
    st.session_state.results = []

# -----------------------------
# Game: Reverse Turing Test
# -----------------------------
st.header("🎮 Spot the Biased Scorer")

resume_idx = random.choice(resumes_df.index.tolist())
resume = resumes_df.loc[resume_idx].to_dict()

st.subheader("Candidate Resume")
st.json(resume)

# AI + persona scoring
ai_score = ai_mock_score(resume)
persona_name, persona_fn = random.choice(list(bias_personas.items()))
persona_score = persona_fn(resume)

# Shuffle order
scores = [("AI Model", ai_score), (persona_name, persona_score)]
random.shuffle(scores)

st.write("### Resume Scores")
for idx, (src, val) in enumerate(scores):
    st.write(f"Option {idx+1}: {val:.2f}")

choice = st.radio(
    "Which score do you think is from the biased persona?",
    [1, 2],
    index=None
)

explanation = st.text_area("Why do you think so? (optional)")

if st.button("Submit Guess"):
    correct_answer = 1 if scores[choice-1][0] == persona_name else 0
    st.session_state.trials.append({
        "resume": resume,
        "ai_score": ai_score,
        "persona_score": persona_score,
        "persona": persona_name,
        "choice": choice,
        "correct": correct_answer,
        "explanation": explanation
    })
    st.success("✅ Response recorded! Try another resume.")

# -----------------------------
# Sidebar: Analytics & Reports
# -----------------------------
st.sidebar.title("📊 Analytics & Reports")

if st.sidebar.button("Run Analysis"):
    trials = st.session_state.trials
    if not trials:
        st.sidebar.warning("No trials yet!")
    else:
        df = pd.DataFrame(trials)
        correct = df["correct"].sum()
        n = len(df)

        st.sidebar.subheader("Performance")
        st.sidebar.write(f"Correct: {correct}/{n} ({correct/n:.2%})")

        # Binomial significance test
        pval = binomial_test(correct, n)
        st.sidebar.write(f"Binomial Test p-value: {pval:.4f}")

        # Distribution comparisons (AI vs persona)
        ai_scores = df["ai_score"].values
        persona_scores = df["persona_score"].values

        kl = kl_divergence(ai_scores, persona_scores)
        js = js_divergence(ai_scores, persona_scores)
        emd = earth_movers_distance(ai_scores, persona_scores)

        st.sidebar.subheader("Distribution Distances")
        st.sidebar.write(f"KL Divergence: {kl:.4f}")
        st.sidebar.write(f"JS Divergence: {js:.4f}")
        st.sidebar.write(f"Earth Mover's Distance: {emd:.4f}")

        # Meta-classifier
        X = df[["ai_score", "persona_score"]].values
        y = (df["persona"] != "AI Model").astype(int).values
        _, auc = train_meta_classifier(X, y)

        st.sidebar.subheader("Meta Classifier")
        st.sidebar.write(f"AUC: {auc:.3f}")

        # -----------------------------
        # Counterfactual Analysis
        # -----------------------------
        st.sidebar.subheader("Counterfactual Analysis")
        cf_pairs = generate_counterfactuals(resumes_df.to_dict(orient="records")[:5])
        st.sidebar.write(f"Generated {len(cf_pairs)} counterfactual resume pairs")

        # -----------------------------
        # Compliance Reporting
        # -----------------------------
        st.sidebar.subheader("Compliance Reports")
        report_html = report.generate_report(df, ai_scores, persona_scores)
        st.sidebar.download_button(
            "⬇️ Download Compliance Report",
            data=report_html,
            file_name="bias_report.html",
            mime="text/html"
        )

        # -----------------------------
        # Mitigation Loop
        # -----------------------------
        st.sidebar.subheader("Bias Mitigation")
        mitigated = mitigation.apply_reweighing(ai_scores, persona_scores)
        st.sidebar.write("Applied Fairlearn/AIF360-style mitigation.")
        st.sidebar.write(f"Mitigated mean AI score: {mitigated.mean():.3f}")

        # Save run result
        save_run_result("reverse_turing_test", {"n": n}, df.to_dict(orient="records"))
        st.sidebar.success("Run results saved ✅")

st.sidebar.info("👉 Submit several guesses, then click **Run Analysis**.")
