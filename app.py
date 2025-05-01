import streamlit as st
import pandas as pd
from fuzyengine import compute_fuzzy_risk
from qnada import generate_risk_assessment


st.set_page_config(page_title="AI-Powered Construction Risk Tool", layout="wide")
st.title("🏗️ GPT-RAG Risk Intelligence for Construction Projects")
st.markdown("This tool uses GPT-4 + Retrieval-Augmented Generation (RAG) and fuzzy logic to identify project risks.")

st.sidebar.header("📋 Project Brief Input")


# Project Overview
project_name = st.sidebar.text_input("Project Name", "Lower Thames Corridor Expansion Project")
project_type = st.sidebar.selectbox("Project Type", ["Bridge", "Road", "Tunnel", "Railway", "School", "Hospital", "Data Center", "Housing"])
project_description = st.sidebar.text_area("Project Description", "A new 3.2-km dual-span cable-stayed bridge crossing the River Thames, including 11 km of new highway to relieve congestion at the Dartford Crossing.")
project_objectives = st.sidebar.text_area("Key Objectives", "Reduce Dartford Crossing congestion, improve logistics access, and stimulate economic growth.")
project_scope = st.sidebar.text_area("Scope of Work", "Bridge superstructure, 11 km new highway, 4 tunnels, ITS integration, utility relocation")

# Funding & Procurement
project_budget = st.sidebar.text_input("Estimated Budget", "£6.8 Billion")
funding_model = st.sidebar.selectbox("Funding Model", ["Public", "Private", "Public-Private Partnership (PPP)"])
procurement_strategy = st.sidebar.selectbox("Procurement Strategy", ["Design-Build", "EPC", "Lump Sum", "Cost Plus"])
contingency = st.sidebar.slider("Contingency Allocation (%)", 0, 30, 15)

# Schedule
project_duration = st.sidebar.slider("Project Duration (months)", 6, 72, 60)
milestones = st.sidebar.text_area("Major Milestones", "Q1 2024: Enabling Works Complete\nQ3 2025: Main Bridge Piling\nQ2 2027: Road Deck Completion\nQ1 2029: Final Testing")
schedule_risks = st.sidebar.text_area("Schedule Risks or Constraints", "Environmental permit delays, supply chain disruptions, adverse weather")
history_delays = st.sidebar.selectbox("Previous Delays on Similar Projects?", ["Yes", "No"])

# Technical Considerations
site_conditions = st.sidebar.selectbox("Site Conditions", ["Urban", "Rural", "Coastal", "Mountainous", "Riverbank"])
key_systems = st.sidebar.text_area("Key Systems and Components", "Bridge towers, marine piling, seismic foundations, smart transport systems")
lead_times = st.sidebar.text_area("Expected Lead Times", "Steel Cabling: 26 weeks, Bearings: 14 weeks, ITS Systems: 18 weeks")
technical_challenges = st.sidebar.text_area("Potential Technical Challenges", "Tidal constraints, wind load design, underground utility conflicts")

# Legal & Stakeholders
stakeholders = st.sidebar.slider("Number of Stakeholders Involved", 1, 50, 28)
stakeholder_influence = st.sidebar.selectbox("Stakeholder Influence", ["Low", "Medium", "High"])
contract_type = st.sidebar.selectbox("Main Contract Type", ["NEC4", "FIDIC", "Design-Build", "PPP"])
dispute_resolution = st.sidebar.selectbox("Formal Dispute Resolution in Place?", ["Yes", "No"])
legal_disputes = st.sidebar.selectbox("Any Previous Legal Disputes?", ["Yes", "No"])

# Environment & Regulation
weather_risks = st.sidebar.text_input("Weather & Climate Risks", "Storm surges, heavy rainfall")
env_constraints = st.sidebar.text_area("Environmental or Permitting Constraints", "Designated wetlands, restricted working hours, noise limits")
reg_frameworks = st.sidebar.text_area("Applicable Regulations", "Planning Act 2008, UK Highways Act, Environmental Management Act")
political_stability = st.sidebar.selectbox("Current Political / Economic Stability", ["Stable", "Volatile"])

# Risk & Performance Expectations
risk_appetite = st.sidebar.selectbox("Organization's Risk Appetite", ["Low", "Medium", "High"])
risk_monitoring = st.sidebar.text_input("Risk Monitoring Approach", "Monthly audits, ISO 31000 compliance")
expected_benefits = st.sidebar.text_area("Expected Project Benefits", "Reduced congestion, improved freight access, economic uplift in Essex and Kent")
kpis = st.sidebar.text_area("Key Performance Indicators (KPIs)", "Schedule adherence >95%, Budget variance <10%, Zero fatalities")


if st.sidebar.button("🔍 Generate Risk Assessment"):
    project_details = f"""
    ### Project Overview
    - Name: {project_name}
    - Type: {project_type}
    - Description: {project_description}
    - Objectives: {project_objectives}
    - Scope: {project_scope}

    ### Funding & Procurement
    - Budget: {project_budget}
    - Funding: {funding_model}
    - Procurement: {procurement_strategy}
    - Contingency Allocation: {contingency}%

    ### Schedule
    - Duration: {project_duration} months
    - Major Milestones: {milestones}
    - Schedule Risks: {schedule_risks}
    - History of Delays: {history_delays}

    ### Technical Considerations
    - Site Conditions: {site_conditions}
    - Key Systems: {key_systems}
    - Lead Times: {lead_times}
    - Technical Challenges: {technical_challenges}

    ### Legal & Stakeholders
    - Stakeholder Count: {stakeholders}
    - Stakeholder Influence: {stakeholder_influence}
    - Contract Type: {contract_type}
    - Dispute Resolution: {dispute_resolution}
    - Past Legal Disputes: {legal_disputes}

    ### Environmental & Regulatory
    - Weather Risks: {weather_risks}
    - Environmental Constraints: {env_constraints}
    - Regulatory Frameworks: {reg_frameworks}
    - Stability: {political_stability}

    ### Risk & Performance
    - Risk Appetite: {risk_appetite}
    - Monitoring Approach: {risk_monitoring}
    - Expected Benefits: {expected_benefits}
    - KPIs: {kpis}
    """


    st.info("⏳ Generating risk assessment using GPT-RAG and fuzzy logic...")
    output_text = generate_risk_assessment(project_details)

    import io
    import csv

    rows = []
    csv_file = io.StringIO(output_text.strip())
    reader = csv.reader(csv_file)

    header = next(reader, None)  # Skip header row

    for row in reader:
        if len(row) == 11:
            try:
                ae = int(row[3])
                lc = int(row[4])
                ic = int(row[5])
                is_ = int(row[6])
                iq = int(row[7])
                rmf = int(row[8])
                cfc = int(row[9])
                fuzzy_score = compute_fuzzy_risk(ae, lc, ic, is_, iq, rmf, cfc)
                row.append(round(fuzzy_score, 2))
                rows.append(row)
            except Exception as e:
                st.warning(f"⚠️ Skipping row due to error: {e}")
        else:
            st.warning(f"⚠️ Skipping malformed row: {row}")

    if rows:
        df = pd.DataFrame(rows, columns=[
            "Risk Factor", "Risk Category", "Risk Detail",
            "Availability of Expertise", "Level of Contingencies",
            "Impact on Cost", "Impact on Schedule", "Impact on Quality",
            "Risk Monitoring Frequency", "Cross-Functional Collaboration",
            "Mitigation Strategy", "Fuzzy Risk Score"
        ])

        st.success("✅ Risk assessment generated!")
        st.markdown("### 🧾 Risk Assessment Table")

        # Format mitigation strategy column with line wrapping for better visualization
        df["Mitigation Strategy"] = df["Mitigation Strategy"].apply(
            lambda x: "\n".join(x[i:i + 70] for i in range(0, len(x), 70)))

        st.dataframe(df.style.set_properties(**{
            'white-space': 'pre-wrap',
            'overflow-wrap': 'break-word'
        }), use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download CSV", csv, "risk_assessment.csv", "text/csv")
    else:
        st.error("❌ No valid risk rows found. Please check the project details and try again.")




