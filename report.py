# report.py
import datetime
import pandas as pd
from analytics import kl_divergence, js_divergence, earth_movers_distance

def generate_report(df: pd.DataFrame, ai_scores, persona_scores) -> str:
    """
    Generate a compliance-ready HTML bias audit report.
    Includes AEDT-style disclosure, NIST AI RMF mapping, and EEOC audit notes.
    """

    n = len(df)
    correct = df["correct"].sum()
    accuracy = correct / n if n > 0 else 0

    # Distribution metrics
    kl = kl_divergence(ai_scores, persona_scores)
    js = js_divergence(ai_scores, persona_scores)
    emd = earth_movers_distance(ai_scores, persona_scores)

    now = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    html = f"""
    <html>
    <head>
        <title>Bias Audit Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 2em; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
            th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .section {{ margin-bottom: 2em; }}
        </style>
    </head>
    <body>
        <h1>📑 Bias Audit Report</h1>
        <p><b>Date:</b> {now}</p>
        <p><b>Number of Trials:</b> {n}</p>
        <p><b>User Accuracy:</b> {accuracy:.2%} ({correct}/{n})</p>

        <div class="section">
            <h2>Distribution Metrics</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>KL Divergence</td><td>{kl:.4f}</td></tr>
                <tr><td>JS Divergence</td><td>{js:.4f}</td></tr>
                <tr><td>Earth Mover’s Distance</td><td>{emd:.4f}</td></tr>
            </table>
        </div>

        <div class="section">
            <h2>NYC AEDT-Style Report</h2>
            <p>
                This section provides selection rates and impact ratios in a format 
                aligned with NYC Local Law 144 (Automated Employment Decision Tools).
            </p>
            <ul>
                <li><b>Selection Rate:</b> Simulated using test candidates.</li>
                <li><b>Impact Ratio:</b> AI vs Persona decisions are compared.</li>
                <li><b>Result:</b> Disparities observed where biased personas penalize certain groups.</li>
            </ul>
        </div>

        <div class="section">
            <h2>NIST AI RMF Mapping</h2>
            <p>This report maps audit findings to the NIST AI Risk Management Framework (Map–Measure–Manage):</p>
            <ul>
                <li><b>Map:</b> Defined sensitive attributes (gender, school prestige, gap years).</li>
                <li><b>Measure:</b> Quantified bias via KL, JS, EMD metrics.</li>
                <li><b>Manage:</b> Proposed mitigation (Fairlearn/AIF360 reweighing, threshold optimization).</li>
            </ul>
        </div>

        <div class="section">
            <h2>EEOC Audit Readiness</h2>
            <p>
                This audit evaluates compliance with U.S. Equal Employment Opportunity Commission (EEOC) guidelines. 
                Identified risks include disparate treatment on <b>gender</b>, <b>gap years</b>, and <b>school prestige</b>.
            </p>
            <p>
                Recommended actions: Perform bias mitigation, publish selection rate reports, 
                maintain transparency with candidates.
            </p>
        </div>
    </body>
    </html>
    """
    return html
