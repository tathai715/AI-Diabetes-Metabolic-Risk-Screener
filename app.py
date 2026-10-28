import numpy as np
import gradio as gr
# ==================================================
# SAMPLE DATA — EDUCATIONAL DEMONSTRATION ONLY
# Features: Fasting glucose, diastolic BP, age
# Labels: 0 = Low Risk, 1 = High Risk
# ==================================================
X = np.array([
    [175, 92, 56],
    [155, 84, 48],
    [162, 90, 62],
    [92, 70, 24],
    [104, 74, 31],
    [88, 68, 22],
    [182, 95, 60],
    [149, 86, 44],
    [98, 72, 29],
    [110, 76, 58],
], dtype=float)
y = np.array([1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
K = 3
# ==================================================
# KNN PREDICTION
# ==================================================
def screen_risk(glucose, diastolic_bp, age):
    try:
        glucose = float(glucose)
        diastolic_bp = float(diastolic_bp)
        age = float(age)
    except (ValueError, TypeError):
        raise gr.Error("Please enter a valid number in every field.")
    values = [glucose, diastolic_bp, age]
    if not all(np.isfinite(v) for v in values):
        raise gr.Error("Please enter finite numeric values.")
    if not 70 <= glucose <= 200:
        raise gr.Error("Fasting glucose must be between 70 and 200 mg/dL.")
    if not 40 <= diastolic_bp <= 130:
        raise gr.Error("Diastolic blood pressure must be between 40 and 130 mm Hg.")
    if not 1 <= age <= 120:
        raise gr.Error("Age must be between 1 and 120 years.")
    patient = np.array(values, dtype=float)
    # Min-max feature scaling
    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    feature_range = np.where(
        feature_max - feature_min == 0,
        1,
        feature_max - feature_min
    )
    X_scaled = (X - feature_min) / feature_range
    patient_scaled = (patient - feature_min) / feature_range
    # Euclidean distances
    distances = np.sqrt(
        np.sum((X_scaled - patient_scaled) ** 2, axis=1)
    )
    # Select K nearest sample records
    nearest_indices = np.argsort(distances)[:K]
    neighbor_labels = y[nearest_indices]
    high_votes = int(np.sum(neighbor_labels == 1))
    low_votes = int(np.sum(neighbor_labels == 0))
    prediction = 1 if high_votes > low_votes else 0
    nearest_distance = float(distances[nearest_indices[0]])
    # Result display
    if prediction == 1:
        risk_title = "HIGH RISK"
        risk_icon = "⚠"
        risk_class = "high-risk"
        explanation = (
            "The nearest example records vote for the high-risk "
            "category in this demonstration."
        )
    else:
        risk_title = "LOW RISK"
        risk_icon = "✓"
        risk_class = "low-risk"
        explanation = (
            "The nearest example records vote for the low-risk "
            "category in this demonstration."
        )

    result_html = f"""
    <div class="result-shell">
        <div class="result-topline">
            <span class="live-dot"></span>
            <span>MODEL OUTPUT</span>
            <span class="result-chip">KNN · k={K}</span>
        </div>

        <div class="risk-panel {risk_class}">
            <div class="risk-symbol">{risk_icon}</div>
            <div>
                <div class="risk-eyebrow">DEMO CLASSIFICATION</div>
                <h2 class="risk-title">{risk_title}</h2>
            </div>
        </div>
        <p class="result-description">{explanation}</p>

        <div class="metrics-grid">
            <div class="metric-card">
                <span class="metric-label">NEAREST DISTANCE</span>
                <span class="metric-value">{nearest_distance:.3f}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">HIGH-RISK VOTES</span>
                <span class="metric-value">{high_votes}/{K}</span>
            </div>
            <div class="metric-card">
                <span class="metric-label">LOW-RISK VOTES</span>
                <span class="metric-value">{low_votes}/{K}</span>
            </div>
        </div>
        <div class="disclaimer">
            <span class="disclaimer-icon">ⓘ</span>
            <div>
                <strong>EDUCATIONAL PROTOTYPE — NOT A MEDICAL DIAGNOSIS</strong>
                <p>This demo uses a small, illustrative sample dataset with
                manually assigned labels. It has not been clinically validated
                and must not be used to diagnose diabetes, estimate actual
                personal risk, or guide medical decisions. Consult a qualified
                healthcare professional for medical advice.</p>
            </div>
        </div>
    </div>
    """

    return result_html
# ==================================================
# FUTURISTIC NAVY + CYAN CSS
# ==================================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #07111f;
    --surface: #0c1b2e;
    --surface-2: #10243a;
    --cyan: #35e0f5;
    --cyan-soft: #8cf2ff;
    --text: #e8f5ff;
    --muted: #91a9c2;
    --border: rgba(74, 208, 237, 0.20);
}
.gradio-container {
    background:
        radial-gradient(ellipse at 12% 0%, rgba(0, 191, 224, 0.12), transparent 38%),
        radial-gradient(ellipse at 95% 30%, rgba(34, 109, 212, 0.10), transparent 35%),
        var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}
.gradio-container,
.gradio-container * {
    box-sizing: border-box;
}
.main-title {
    color: var(--text) !important;
    text-align: center;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(26px, 4vw, 42px) !important;
    font-weight: 700 !important;
    letter-spacing: -1.2px;
    margin: 18px 0 8px !important;
}
.subtitle {
    color: var(--muted) !important;
    text-align: center;
    max-width: 720px;
    margin: 0 auto 26px !important;
    line-height: 1.7;
}
.hero-badge {
    display: table;
    margin: 0 auto 14px;
    padding: 7px 13px;
    border: 1px solid rgba(53, 224, 245, 0.35);
    border-radius: 999px;
    background: rgba(53, 224, 245, 0.08);
    color: var(--cyan) !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
}
.panel {
    background: linear-gradient(
        145deg,
        rgba(16, 36, 58, 0.97),
        rgba(9, 24, 41, 0.97)
    ) !important;
    border: 1px solid var(--border) !important;
    border-radius: 22px !important;
    padding: 24px !important;
    box-shadow: 0 14px 45px rgba(0, 0, 0, 0.18);
    min-width: 0;
}
.panel h2 {
    color: var(--text) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
.section-caption {
    color: var(--cyan) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 2px;
}
.section-description {
    color: var(--muted) !important;
    line-height: 1.7;
    font-size: 14px !important;
    margin-bottom: 22px;
}
/* Rounded, readable input cards */
.gradio-container label {
    color: #d9efff !important;
    font-weight: 600 !important;
}
.gradio-container input,
.gradio-container textarea,
.gradio-container select {
    color: #f0fbff !important;
    background: #081625 !important;
    border: 1px solid rgba(86, 200, 228, 0.26) !important;
    border-radius: 13px !important;
    min-height: 50px;
    font-size: 16px !important;
    box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.12);
    opacity: 1 !important;
}
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
    color: #8da6bd !important;
    opacity: 1 !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    border-color: var(--cyan) !important;
    box-shadow: 0 0 0 3px rgba(53, 224, 245, 0.10) !important;
}
.gradio-container .wrap,
.gradio-container .form {
    border-color: var(--border) !important;
}
.input-note {
    color: #8da6bd !important;
    font-size: 12px !important;
    margin-top: -7px;
    margin-bottom: 9px;
}
/* Main action button */
.run-button {
    background: linear-gradient(100deg, #087fba, #16c6dc) !important;
    color: #04111d !important;
    border: 1px solid rgba(140, 242, 255, 0.65) !important;
    border-radius: 14px !important;
    min-height: 55px !important;
    font-weight: 800 !important;
    font-size: 15px !important;
    letter-spacing: 0.2px;
    box-shadow: 0 5px 22px rgba(20, 199, 222, 0.16);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.run-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(20, 199, 222, 0.28);
}
/* Result panel */
.result-shell {
    background: #081625;
    border: 1px solid rgba(53, 224, 245, 0.20);
    border-radius: 18px;
    padding: 20px;
    color: var(--text);
    overflow-wrap: anywhere;
}
.result-topline {
    display: flex;
    align-items: center;
    gap: 9px;
    color: var(--cyan);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.6px;
    flex-wrap: wrap;
}
.live-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--cyan);
    box-shadow: 0 0 12px rgba(53, 224, 245, 0.85);
}
.result-chip {
    margin-left: auto;
    padding: 6px 10px;
    border: 1px solid var(--border);
    border-radius: 999px;
    background: rgba(53, 224, 245, 0.06);
    color: #b9eefa;
    letter-spacing: 0.5px;
}
.risk-panel {
    display: flex;
    align-items: center;
    gap: 15px;
    margin: 22px 0 17px;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid var(--border);
    background: rgba(255, 255, 255, 0.025);
}
.risk-symbol {
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    width: 44px;
    height: 44px;
    border-radius: 13px;
    font-size: 25px;
    font-weight: 800;
}

.low-risk {
    border-color: rgba(54, 211, 153, 0.30);
}

.low-risk .risk-symbol {
    color: #5cf0bd;
    background: rgba(54, 211, 153, 0.12);
}

.high-risk {
    border-color: rgba(255, 105, 105, 0.35);
}

.high-risk .risk-symbol {
    color: #ff9b9b;
    background: rgba(255, 105, 105, 0.12);
}

.risk-eyebrow {
    color: var(--muted);
    font-size: 10px;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 5px;
}

.risk-title {
    margin: 0 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 26px !important;
    letter-spacing: 0.4px;
}

.low-risk .risk-title {
    color: #5cf0bd !important;
}

.high-risk .risk-title {
    color: #ff9b9b !important;
}

.result-description {
    color: #b8cce0;
    line-height: 1.8;
    font-size: 14px;
    margin: 18px 0;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin: 20px 0;
}

.metric-card {
    background: #10243a;
    border: 1px solid rgba(74, 208, 237, 0.14);
    border-radius: 13px;
    padding: 14px 12px;
    min-width: 0;
}

.metric-label {
    display: block;
    color: #8da6bd;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.8px;
    line-height: 1.5;
    margin-bottom: 9px;
}
.metric-value {
    display: block;
    color: var(--cyan-soft);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
}
.disclaimer {
    display: flex;
    gap: 12px;
    margin-top: 23px;
    padding: 15px;
    background: rgba(255, 190, 80, 0.06);
    border: 1px solid rgba(255, 190, 80, 0.25);
    border-radius: 13px;
    color: #ffe0a1;
}
.disclaimer-icon {
    flex-shrink: 0;
    font-size: 20px;
    color: #ffd071;
}
.disclaimer strong {
    display: block;
    font-size: 10px;
    letter-spacing: 0.7px;
    line-height: 1.7;
}

.disclaimer p {
    color: #c6bda9 !important;
    font-size: 12px;
    line-height: 1.8;
    margin: 8px 0 0;
}

.footer-note {
    text-align: center;
    color: #7f9bb3 !important;
    font-size: 12px !important;
    line-height: 1.8;
    margin: 18px auto;
    max-width: 800px;
}

/* Mobile and tablet layout */
@media (max-width: 760px) {
    .main-title {
        font-size: 28px !important;
        letter-spacing: -0.6px;
    }

    .panel {
        padding: 17px !important;
        border-radius: 17px !important;
    }

    .result-shell {
        padding: 15px;
    }

    .metrics-grid {
        grid-template-columns: 1fr;
    }

    .metric-card {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
    }

    .metric-label {
        margin-bottom: 0;
    }

    .metric-value {
        font-size: 20px;
    }

    .risk-title {
        font-size: 23px !important;
    }
}
"""
# ==================================================
# GRADIO INTERFACE
# ==================================================

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate"
    ),
    css=custom_css,
    title="AI Diabetes & Metabolic Risk Screener"
) as app:

    gr.Markdown(
        '<div class="hero-badge">✦ AI HEALTH LAB / PROTOTYPE</div>'
    )

    gr.Markdown(
        "# AI Diabetes & Metabolic Risk Screener",
        elem_classes=["main-title"]
    )

    gr.Markdown(
        "An interactive machine-learning demonstration powered by "
        "NumPy and the K-Nearest Neighbors algorithm.",
        elem_classes=["subtitle"]
    )

    with gr.Row(equal_height=False):

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown(
                '<div class="section-caption">01 / INPUT PARAMETERS</div>'
            )
            gr.Markdown("## Patient Information")

            gr.Markdown(
                "Enter sample values to explore how the demo model works.",
                elem_classes=["section-description"]
            )

            glucose = gr.Number(
                label="Fasting Glucose (mg/dL)",
                value=106,
                precision=1
            )
            gr.Markdown(
                "Illustrative input range: 70–200 mg/dL",
                elem_classes=["input-note"]
            )

            diastolic_bp = gr.Number(
                label="Diastolic Blood Pressure (mm Hg)",
                value=72,
                precision=1
            )
            gr.Markdown(
                "Enter the lower blood pressure number.",
                elem_classes=["input-note"]
            )

            age = gr.Number(
                label="Age (years)",
                value=30,
                precision=1
            )
            gr.Markdown(
                "Enter age in years.",
                elem_classes=["input-note"]
            )

            run_button = gr.Button(
                "▶  RUN SCREENING ",
                elem_classes=["run-button"]
            )

        with gr.Column(scale=1, elem_classes=["panel"]):
            gr.Markdown(
                '<div class="section-caption">02 / ANALYSIS OUTPUT</div>'
            )
            gr.Markdown("## Screening Result")

            result = gr.HTML(
                """
                <div class="result-shell">
                    <div class="result-topline">
                        <span class="live-dot"></span>
                        MODEL STANDBY
                    </div>
                    <div class="risk-panel">
                        <div class="risk-symbol">⌁</div>
                        <div>
                            <div class="risk-eyebrow">SYSTEM STATUS</div>
                            <h2 class="risk-title" style="color:#8cf2ff">
                                Ready to analyze
                            </h2>
                        </div>
                    </div>
                    <p class="result-description">
                        Enter the example values and run the demo to view
                        the KNN classification output.
                    </p>
                    <div class="disclaimer">
                        <span class="disclaimer-icon">ⓘ</span>
                        <div>
                            <strong>EDUCATIONAL USE ONLY</strong>
                            <p>This prototype is not a medical diagnosis and
                            has not been clinically validated.</p>
                        </div>
                    </div>
                </div>
                """
            )

    gr.Markdown(
        "**HOW IT WORKS**  ·  Inputs are scaled, Euclidean distances are "
        "calculated, and the three nearest sample records vote on the output.",
        elem_classes=["footer-note"]
    )

    run_button.click(
        fn=screen_risk,
        inputs=[glucose, diastolic_bp, age],
        outputs=result
    )
# share=True creates a temporary public Gradio URL while the app is running.
if __name__ == "__main__":
    app.launch(share=True)