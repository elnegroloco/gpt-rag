import os
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata4")

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create retriever for document search
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Create GPT-4 model
llm = ChatOpenAI(model="gpt-4o")

# Refined GPT-4 Prompt for Risk Assessment
risk_assessment_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert in construction risk management. Based on the provided project details, generate a structured risk assessment table.

    ### **Risk Assessment Criteria**
    - **Probability**: Based on [Availability of Expertise, Level of Contingencies]
    - **Impact**: Based on [Impact on Cost, Schedule, and Quality]
    - **Detection**: Based on [Risk Monitoring Frequency, Cross-Functional Collaboration]

    ### **Output Instructions**
    - Output should be **purely tabular**, in **CSV format**.
    - **No bullet points, explanations, or numbering.**
    - Use the format:

    ```
    Risk Factor, Risk Category, Risk Detail, Availability of Expertise (1-5), Level of Contingencies (1-5), Impact on Cost (1-5), Impact on Schedule (1-5), Impact on Quality (1-5), Frequency of Risk Monitoring (1-5), Cross-Functional Team Collaboration (1-5), Mitigation Strategy
    Structural Failure, Safety, Poor material quality may cause collapse, 3, 4, 5, 5, 4, 3, 4, Ensure strict material quality checks and compliance with safety codes.
    Budget Overruns, Financial, Project exceeding allocated costs due to mismanagement, 2, 3, 5, 4, 3, 2, 3, Implement real-time budget tracking and cost forecasting.
    ```

    {context}
    """
)

# Create retrieval chain
risk_chain = create_stuff_documents_chain(llm, risk_assessment_prompt)
rag_chain = create_retrieval_chain(retriever, risk_chain)

# Define Fuzzy Logic System
def setup_fuzzy_system():
    # Inputs
    ae = ctrl.Antecedent(np.arange(0, 11, 1), 'Availability of Expertise')
    lc = ctrl.Antecedent(np.arange(0, 11, 1), 'Level of Contingencies')
    ic = ctrl.Antecedent(np.arange(0, 11, 1), 'Impact on Cost')
    is_ = ctrl.Antecedent(np.arange(0, 11, 1), 'Impact on Schedule')
    iq = ctrl.Antecedent(np.arange(0, 11, 1), 'Impact on Quality')
    rmf = ctrl.Antecedent(np.arange(0, 11, 1), 'Risk Monitoring Frequency')
    cfc = ctrl.Antecedent(np.arange(0, 11, 1), 'Cross-Functional Collaboration')

    # Outputs
    probability = ctrl.Consequent(np.arange(0, 11, 1), 'probability')
    impact = ctrl.Consequent(np.arange(0, 11, 1), 'impact')
    detection = ctrl.Consequent(np.arange(0, 11, 1), 'detection')
    risk_level = ctrl.Consequent(np.arange(0, 11, 1), 'risk_level')

    # Membership Functions
    for var in [ae, lc, ic, is_, iq, rmf, cfc]:
        var['Low'] = fuzz.trimf(var.universe, [0, 0, 5])
        var['Medium'] = fuzz.trimf(var.universe, [0, 5, 10])
        var['High'] = fuzz.trimf(var.universe, [5, 10, 10])

    for var in [probability, impact, detection, risk_level]:
        var['Low'] = fuzz.trimf(var.universe, [0, 0, 5])
        var['Medium'] = fuzz.trimf(var.universe, [0, 5, 10])
        var['High'] = fuzz.trimf(var.universe, [5, 10, 10])

    # Rules for Probability
    prob_rules = [
        ctrl.Rule(ae['Low'] & lc['Low'], probability['High']),
        ctrl.Rule(ae['Low'] & lc['Medium'], probability['High']),
        ctrl.Rule(ae['Low'] & lc['High'], probability['Medium']),
        ctrl.Rule(ae['Medium'] & lc['Low'], probability['High']),
        ctrl.Rule(ae['Medium'] & lc['Medium'], probability['Medium']),
        ctrl.Rule(ae['Medium'] & lc['High'], probability['Low']),
        ctrl.Rule(ae['High'] & lc['Low'], probability['Medium']),
        ctrl.Rule(ae['High'] & lc['Medium'], probability['Low']),
        ctrl.Rule(ae['High'] & lc['High'], probability['Low'])
    ]

    # Rules for Impact
    impact_rules = [
        ctrl.Rule(ic['Low'] & is_['Low'] & iq['Low'], impact['Low']),
        ctrl.Rule(ic['Low'] & is_['Low'] & iq['Medium'], impact['Low']),
        ctrl.Rule(ic['Low'] & is_['Low'] & iq['High'], impact['Medium']),
        # Add all 27 rules for Impact...
    ]

    # Rules for Detection
    detection_rules = [
        ctrl.Rule(rmf['Low'] & cfc['Low'], detection['Low']),
        ctrl.Rule(rmf['Low'] & cfc['Medium'], detection['Low']),
        ctrl.Rule(rmf['Low'] & cfc['High'], detection['Medium']),
        # Add all 9 rules for Detection...
    ]

    # Rules for Risk Level
    risk_rules = [
        ctrl.Rule(probability['Low'] & impact['Low'] & detection['High'], risk_level['Low']),
        ctrl.Rule(probability['Low'] & impact['Low'] & detection['Medium'], risk_level['Low']),
        ctrl.Rule(probability['Low'] & impact['Low'] & detection['Low'], risk_level['Medium']),
        # Add all 27 rules for Risk Level...
    ]

    # Control Systems
    prob_ctrl = ctrl.ControlSystem(prob_rules)
    impact_ctrl = ctrl.ControlSystem(impact_rules)
    detection_ctrl = ctrl.ControlSystem(detection_rules)
    risk_ctrl = ctrl.ControlSystem(risk_rules)

    # Simulations
    prob_sim = ctrl.ControlSystemSimulation(prob_ctrl)
    impact_sim = ctrl.ControlSystemSimulation(impact_ctrl)
    detection_sim = ctrl.ControlSystemSimulation(detection_ctrl)
    risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

    return prob_sim, impact_sim, detection_sim, risk_sim

# Initialize Fuzzy Logic System
prob_sim, impact_sim, detection_sim, risk_sim = setup_fuzzy_system()

# Function to generate risk assessment
def generate_risk_assessment(project_details):
    """
    Retrieves risk-related context from RAG, formats it properly, and generates a structured risk assessment.
    """
    # Step 1: Retrieve relevant risk-related documents
    retrieved_docs = retriever.invoke(project_details)

    # Step 2: Ensure retrieved documents are formatted correctly
    context_texts = []
    for i, doc in enumerate(retrieved_docs):
        if isinstance(doc, str):  # If it's already a plain string, use it directly
            context_texts.append(doc)
        elif hasattr(doc, "page_content"):  # If it's a LangChain document object, extract page content
            context_texts.append(doc.page_content)
        elif isinstance(doc, dict) and "page_content" in doc:  # If it's a dictionary, extract page content
            context_texts.append(doc["page_content"])
        else:
            print(f"⚠️ Skipping document {i+1} - Unrecognized format: {type(doc)}")

    # Step 3: Combine formatted retrieved texts
    context_combined = "\n".join(context_texts) if context_texts else "No additional context retrieved."

    # Step 4: Pass BOTH project details and the retrieved context to GPT for risk assessment
    response = rag_chain.invoke({
        "input": project_details,
        "context": context_combined  # Injects retrieved knowledge!
    })

    return response["answer"]

# Function to save output as CSV
def save_to_csv(output_text, filename="risk_assessment.csv"):
    data = []
    lines = output_text.strip().split("\n")

    for line in lines:
        fields = line.split(",")  # Ensure correct CSV parsing
        fields = [f.strip() for f in fields]  # Remove unnecessary whitespace
        if len(fields) == 11:  # Ensure it matches the expected format (11 columns)
            # Extract values for Probability, Impact, and Detection
            ae = int(fields[3])  # Availability of Expertise (1-5)
            lc = int(fields[4])  # Level of Contingencies (1-5)
            ic = int(fields[5])  # Impact on Cost (1-5)
            is_ = int(fields[6])  # Impact on Schedule (1-5)
            iq = int(fields[7])  # Impact on Quality (1-5)
            rmf = int(fields[8])  # Risk Monitoring Frequency (1-5)
            cfc = int(fields[9])  # Cross-Functional Collaboration (1-5)

            # Compute Probability, Impact, and Detection
            prob_sim.input['Availability of Expertise'] = ae
            prob_sim.input['Level of Contingencies'] = lc
            prob_sim.compute()
            probability_val = prob_sim.output['probability']

            impact_sim.input['Impact on Cost'] = ic
            impact_sim.input['Impact on Schedule'] = is_
            impact_sim.input['Impact on Quality'] = iq
            impact_sim.compute()
            impact_val = impact_sim.output['impact']

            detection_sim.input['Risk Monitoring Frequency'] = rmf
            detection_sim.input['Cross-Functional Collaboration'] = cfc
            detection_sim.compute()
            detection_val = detection_sim.output['detection']

            # Compute Risk Level
            risk_sim.input['probability'] = probability_val
            risk_sim.input['impact'] = impact_val
            risk_sim.input['detection'] = detection_val
            risk_sim.compute()
            fuzzy_risk = risk_sim.output['risk_level']

            # Add Fuzzy Risk Number to the row
            fields.append(fuzzy_risk)
            data.append(fields)

    if data:
        df = pd.DataFrame(data, columns=[
            "Risk Factor", "Risk Category", "Risk Detail",
            "Availability of Expertise (1-5)", "Level of Contingencies (1-5)",
            "Impact on Cost (1-5)", "Impact on Schedule (1-5)", "Impact on Quality (1-5)",
            "Frequency of Risk Monitoring (1-5)", "Cross-Functional Team Collaboration (1-5)",
            "Mitigation Strategy", "Fuzz Risk Number"
        ])
        df.to_csv(filename, index=False)
        print(f"✅ Risk assessment saved to {filename}")
    else:
        print("⚠️ No valid risk data found! Check GPT output formatting.")

# Function to collect structured project details
def collect_project_details():
    print("\n🚀 Welcome to the AI-Powered Project Risk Assessment Tool")
    print("Generating risk assessment for a real-world project: **UK Bridge Construction**\n")

    # 📌 **Basic Project Information**
    project_type = "Bridge"
    budget = "High"
    complexity = "Complex"
    project_duration = "48"  # in months
    stakeholders = "15"

    # 📌 **Workforce & Expertise**
    labor_availability = "Sufficient"
    team_experience = "Highly Experienced"
    subcontractor_dependence = "50"  # Percentage

    # 📌 **Financial & Contingency Planning**
    project_budget = "£500M"
    contingency_budget = "12"  # Percentage
    funding_stability = "Mixed (Public-Private Partnership)"

    # 📌 **Risk Monitoring & Safety Compliance**
    risk_framework = "ISO 31000"
    safety_audits = "Monthly"
    cross_team_collab = "High"
    compliance = "UK Highways Act, Health & Safety at Work Act"

    # 📌 **Project Schedule & Timeline Sensitivity**
    schedule_criticality = "Fixed Deadlines"
    dependencies = "High"
    history_delays = "No"

    # 📌 **Contractual & Legal Risks**
    contract_type = "PPP (Public-Private Partnership)"
    dispute_resolution = "Yes"
    legal_disputes = "Yes"

    # 📌 **Environmental & External Factors**
    site_conditions = "Coastal"
    weather_risks = "Storms, Flooding"
    political_stability = "Stable"
    supply_chain = "Imported Materials"

    # 📌 **Project-Specific Technical Factors**
    project_goals = "2-km long dual carriageway bridge over the River Thames"
    key_milestones = "Foundation Piling by Q2 2025, Main Span Completion by Q3 2027, Final Commissioning by Q1 2028"
    resource_lead_times = "Steel Cables: 20 weeks, Concrete: 6 weeks, Electrical Systems: 10 weeks"
    technical_specs = "Seismic-resistant foundation, Wind Load Protection"
    risk_tolerance = "Medium"

    # 🔹 **Formatted Output for Risk Analysis**
    project_details = f"""
    ### Basic Project Information
    - Project Type: {project_type}
    - Budget Constraints: {budget}
    - Execution Complexity: {complexity}
    - Project Duration: {project_duration} months
    - Number of Stakeholders: {stakeholders}

    ### Workforce & Expertise
    - Availability of Skilled Labor: {labor_availability}
    - Team Experience: {team_experience}
    - Subcontractor Dependence: {subcontractor_dependence}%

    ### Financial & Contingency Planning
    - Total Project Budget: {project_budget}
    - Contingency Budget: {contingency_budget}%
    - Funding Stability: {funding_stability}

    ### Risk Monitoring & Safety Compliance
    - Risk Framework: {risk_framework}
    - Safety Audits Frequency: {safety_audits}
    - Cross-Team Collaboration: {cross_team_collab}
    - Regulatory Compliance: {compliance}

    ### Project Schedule & Timeline Sensitivity
    - Schedule Criticality: {schedule_criticality}
    - Task Dependencies: {dependencies}
    - History of Delays: {history_delays}

    ### Contractual & Legal Risks
    - Contract Type: {contract_type}
    - Dispute Resolution Mechanism: {dispute_resolution}
    - Previous Legal Disputes: {legal_disputes}

    ### Environmental & External Factors
    - Site Conditions: {site_conditions}
    - Weather & Climate Risks: {weather_risks}
    - Political & Economic Stability: {political_stability}
    - Supply Chain Reliability: {supply_chain}

    ### Project-Specific Technical Factors
    - Project Goals: {project_goals}
    - Key Milestones: {key_milestones}
    - Resource Lead Times: {resource_lead_times}
    - Technical Specifications: {technical_specs}
    - Organization's Risk Tolerance: {risk_tolerance}
    """

    return project_details

# Main execution
if __name__ == "__main__":
    print("🚀 Welcome to the AI-Powered Construction Risk Assessment Tool")

    while True:
        project_details = collect_project_details()
        result = generate_risk_assessment(project_details)
        print("\n🔍 Generated Risk Assessment:\n")
        print(result)
        save_to_csv(result)

        another = input("\nWould you like to analyze another project? (yes/no): ").lower()
        if another != "yes":
            print("✅ Exiting program. Goodbye!")
            break