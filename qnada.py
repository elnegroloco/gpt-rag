import os
import csv
import json
import pandas as pd
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain



from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fuzyengine import compute_fuzzy_risk, compute_fuzzy_detection, compute_fuzzy_impact, compute_fuzzy_probability
from fuzyengine import show_surface_plot
from fuzyengine import prob_ctrl, impact_ctrl, detection_ctrl, risk_ctrl
from skfuzzy import control as ctrl
from fuzyengine import visualize_fuzzy_scores
from langchain.retrievers.multi_query import MultiQueryRetriever
import io

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define directories
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db_with_metadata5")

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Load the existing vector store
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)



# # Create retriever for document search
# retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})

# Create GPT-4 model
llm = ChatOpenAI(model="gpt-4o")

# Refined GPT-4 Prompt for Risk Assessment
risk_assessment_prompt = ChatPromptTemplate.from_template(
    """
You are a senior construction risk manager specializing in large-scale infrastructure projects. Your task is to generate a high-quality, structured risk assessment table using the provided project details and retrieved contextual documents.

### OBJECTIVE
Identify and clearly describe the most relevant risks associated with the project.  
✅ You must **always generate a minimum of 10 distinct risks**.  
✅ If project information is insufficient, use your professional knowledge to extrapolate realistic risks based on typical construction projects.  
✅ Never provide fewer than 10 risks.

### RISK DIMENSIONS
Assess each risk across:
- **Probability**: [Availability of Expertise, Level of Contingencies]
- **Impact**: [Impact on Cost, Schedule, and Quality]
- **Detection Difficulty**: [Risk Monitoring Frequency, Cross-Functional Collaboration]

### OUTPUT FORMAT
- Output must be in **strict CSV format** — no titles, no commentary, no explanations outside the table.
- Each risk must be a separate row with these columns: Risk Factor, Risk Category, Risk Detail, Availability of Expertise (1-5), Level of Contingencies (1-5), Impact on Cost (1-5), Impact on Schedule (1-5), Impact on Quality (1-5), Frequency of Risk Monitoring (1-5), Cross-Functional Team Collaboration (1-5), Mitigation Strategy
- Enclose the Risk Detail field in double quotes ("...") to ensure clean CSV format.
- Enclose the Mitigation strategy field in double quotes ("...") to ensure clean CSV format.


### FIELD GUIDANCE
- **Risk Factor**: Short, descriptive title (e.g., “Material Shortages”, “Design Errors”).
- **Risk Category**: Choose from: Safety, Financial, Legal, Operational, Environmental, Scheduling, Technical, Stakeholder.
- **Risk Detail**: 1–2 professional sentences describing the risk's cause, effect, and potential project impact.
- **Mitigation Strategy**: 2–3 detailed, actionable sentences explaining how to reduce, transfer, or control the risk.

### ADDITIONAL INSTRUCTIONS
- Risks must cover **different categories** (do not cluster too many in one domain).
- Mitigation strategies must be specific and non-repetitive.
- If context is insufficient, you are expected to **expand the risks intelligently** based on typical project risks.
- Maintain high technical language standards, referencing construction best practices (e.g., ISO 31000, FIDIC clauses, NEC4 principles) when appropriate.

{context}

    """
)


# Wrap retriever in MultiQueryRetriever
llm = ChatOpenAI(model="gpt-4")



# Function to extract metadata filter from project_details using GPT
def extract_metadata_filters(project_details):
    prompt = f"""
    From the following construction project description, extract metadata as JSON with:
    - project_type
    - sector
    - country_or_region

    If not found, set value to "unknown".

    Text:
    {project_details[:2000]}
    """
    try:
        response = llm.invoke(prompt)
        print("\n🧠 GPT Extracted Metadata Filters:", response.content.strip())
        return json.loads(response.content.strip())
    except Exception as e:
        print(f"❌ Failed to extract metadata filters: {e}")
        return {}


# Function to generate risk assessment
def generate_risk_assessment(project_details):
    """
    Retrieves risk-related context from RAG using MultiQueryRetriever,
    formats it properly, and generates a structured risk assessment.
    """
    metadata_filter = extract_metadata_filters(project_details)
    # ✅ Only use project_type for now to avoid Chroma limitations
    filter_key = "project_type"
    filter_value = metadata_filter.get(filter_key, "unknown")

    # ✅ Use $eq operator for proper Chroma filtering
    if filter_value.lower() != "unknown":
        search_filter = {

                filter_key: {"$eq": filter_value.lower()}

        }
        print(f"📎 Using metadata filter: {search_filter}")
    else:
        search_filter = {}
        print("⚠️ No valid metadata filter found. Proceeding without filter.")

    # ✅ MultiQueryRetriever with correct filter structure
    retriever = MultiQueryRetriever.from_llm(
        retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 15, "filter": search_filter}),
        llm=llm
    )

    # Create retrieval chain
    risk_chain = create_stuff_documents_chain(llm, risk_assessment_prompt)
    rag_chain = create_retrieval_chain(retriever, risk_chain)

    # Step 1: Retrieve relevant risk-related documents
    retrieved_docs = retriever.get_relevant_documents(project_details)

    print("\n🔍 Retrieved Documents (Debug):")
    for i, doc in enumerate(retrieved_docs):
        print(f"--- Document {i+1} ---")
        print(f"Type: {type(doc)}")
        print(f"Metadata: {doc.metadata}")
        print(f"Content: {doc.page_content[:300]}...\n")

    # Step 2: Extract and clean page content
    context_texts = []
    for i, doc in enumerate(retrieved_docs):
        if hasattr(doc, "page_content"):
            context_texts.append(doc.page_content)
        elif isinstance(doc, dict) and "page_content" in doc:
            context_texts.append(doc["page_content"])
        else:
            print(f"⚠️ Skipping document {i+1} - Unrecognized format: {type(doc)}")

    # Step 3: Combine formatted context
    context_combined = "\n".join(context_texts) if context_texts else "No additional context retrieved."

    print("\n📄 Combined RAG Context (first 500 chars):")
    print(context_combined[:500], "\n...")

    # Step 4: Inject context into GPT risk prompt
    response = rag_chain.invoke({
        "input": project_details,
        "context": context_combined
    })

    return response["answer"]



def save_to_csv(output_text, filename="risk_assessment2.csv"):
    data = []
    csv_file = io.StringIO(output_text.strip())
    reader = csv.reader(csv_file)

    header = next(reader, None)  # Skip header line
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
                prob = compute_fuzzy_probability(ae, lc)
                impact = compute_fuzzy_impact(ic, is_, iq)
                detect = compute_fuzzy_detection(rmf, cfc)
                row.append(str(fuzzy_score))
                row.append(str(prob))
                row.append(str(impact))
                row.append(str(detect))
                data.append(row)
            except Exception as e:
                print(f"⚠️ Skipping row due to error: {e}")
        else:
            print(f"⚠️ Skipping row with incorrect column count: {row}")

    if data:
        df = pd.DataFrame(data, columns=[
            "Risk Factor", "Risk Category", "Risk Detail",
            "Availability of Expertise (1-5)", "Level of Contingencies (1-5)",
            "Impact on Cost (1-5)", "Impact on Schedule (1-5)", "Impact on Quality (1-5)",
            "Frequency of Risk Monitoring (1-5)", "Cross-Functional Team Collaboration (1-5)",
            "Mitigation Strategy", "Fuzzy Risk Number" , "Prob", "Impa", "Dete"
        ])
        df.to_csv(filename, index=False)
        print(f"✅ Risk assessment saved to {filename}")
    else:
        print("⚠️ No valid risk data found! Check GPT output formatting.")



def collect_project_details():
    print("\n🚀 Welcome to the AI-Powered Project Risk Assessment Tool")
    print("Generating risk assessment for a real-world project: **Lower Thames Corridor Expansion Project**\n")

    # 📌 **Basic Project Information**
    project_name = "Lower Thames Corridor Expansion Project"
    project_type = "Bridge"
    project_description = "A new 3.2-km dual-span cable-stayed bridge crossing the River Thames, including 11 km of new highway to relieve congestion at the Dartford Crossing."
    project_objectives = "Reduce Dartford Crossing congestion, improve logistics access, and stimulate economic growth."
    project_scope = "Bridge superstructure, 11 km new highway, 4 tunnels, ITS integration, utility relocation"
    budget = "Very High"
    complexity = "Very Complex"
    project_duration = "60"  # in months
    stakeholders = "28"

    # 📌 **Workforce & Expertise**
    labor_availability = "Sufficient"
    team_experience = "Mixed Experience"
    subcontractor_dependence = "60"  # Percentage

    # 📌 **Financial & Contingency Planning**
    project_budget = "£6.8 Billion"
    contingency_budget = "15"  # Percentage
    funding_stability = "Public-Private Partnership (PPP)"
    procurement_strategy = "Design-Build"

    # 📌 **Risk Monitoring & Safety Compliance**
    risk_framework = "ISO 31000"
    safety_audits = "Monthly audits"
    cross_team_collab = "High"
    compliance = "Planning Act 2008, UK Highways Act, Environmental Management Act"

    # 📌 **Project Schedule & Timeline Sensitivity**
    schedule_criticality = "High"
    dependencies = "High"
    history_delays = "No"
    key_milestones = "Q1 2024: Enabling Works, Q3 2025: Main Bridge Piling, Q2 2027: Road Deck Completion, Q1 2029: Final Testing"
    schedule_risks = "Environmental permit delays, supply chain disruptions, adverse weather"

    # 📌 **Contractual & Legal Risks**
    contract_type = "PPP"
    dispute_resolution = "Yes"
    legal_disputes = "No"
    stakeholder_influence = "High"

    # 📌 **Environmental & External Factors**
    site_conditions = "Riverbank"
    weather_risks = "Storm surges, heavy rainfall"
    env_constraints = "Designated wetlands, restricted working hours, noise limits"
    political_stability = "Stable"
    supply_chain = "Mixed Domestic and Imported"

    # 📌 **Project-Specific Technical Factors**
    key_systems = "Bridge towers, marine piling, seismic foundations, smart transport systems"
    resource_lead_times = "Steel Cabling: 26 weeks, Bearings: 14 weeks, ITS Systems: 18 weeks"
    technical_specs = "Seismic foundations, smart transport systems, tidal & wind resilience"
    technical_challenges = "Tidal constraints, wind load design, underground utility conflicts"
    risk_tolerance = "Medium"

    # 📌 **Performance Expectations**
    expected_benefits = "Reduced congestion, improved freight access, economic uplift in Essex and Kent"
    kpis = "Schedule adherence >95%, Budget variance <10%, Zero fatalities"

    # 🔹 **Formatted Output for Risk Analysis**
    project_details = f"""
    ### Basic Project Information
    - Project Name: {project_name}
    - Project Type: {project_type}
    - Description: {project_description}
    - Objectives: {project_objectives}
    - Scope of Work: {project_scope}
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
    - Procurement Strategy: {procurement_strategy}

    ### Risk Monitoring & Safety Compliance
    - Risk Framework: {risk_framework}
    - Safety Audits Frequency: {safety_audits}
    - Cross-Team Collaboration: {cross_team_collab}
    - Regulatory Compliance: {compliance}

    ### Project Schedule & Timeline Sensitivity
    - Schedule Criticality: {schedule_criticality}
    - Task Dependencies: {dependencies}
    - History of Delays: {history_delays}
    - Major Milestones: {key_milestones}
    - Schedule Risks: {schedule_risks}

    ### Contractual & Legal Risks
    - Contract Type: {contract_type}
    - Dispute Resolution Mechanism: {dispute_resolution}
    - Previous Legal Disputes: {legal_disputes}
    - Stakeholder Influence: {stakeholder_influence}

    ### Environmental & External Factors
    - Site Conditions: {site_conditions}
    - Weather & Climate Risks: {weather_risks}
    - Environmental Constraints: {env_constraints}
    - Political & Economic Stability: {political_stability}
    - Supply Chain Reliability: {supply_chain}

    ### Project-Specific Technical Factors
    - Key Systems: {key_systems}
    - Resource Lead Times: {resource_lead_times}
    - Technical Specifications: {technical_specs}
    - Technical Challenges: {technical_challenges}
    - Organization's Risk Tolerance: {risk_tolerance}

    ### Performance Expectations
    - Expected Benefits: {expected_benefits}
    - Key Performance Indicators (KPIs): {kpis}
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

        # Simulators to visualize surface
        prob_sim = ctrl.ControlSystemSimulation(prob_ctrl)
        impact_sim = ctrl.ControlSystemSimulation(impact_ctrl)
        detection_sim = ctrl.ControlSystemSimulation(detection_ctrl)
        risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

        # Surface plot: AE vs LC → Probability
        show_surface_plot(prob_sim, 'AE', 'LC', 'Probability')

        # Surface plot: IC vs IS → Impact
        show_surface_plot(impact_sim, 'IC', 'IS', 'Impact')

        # Surface plot: RMF vs CFC → Detection
        show_surface_plot(detection_sim, 'RMF', 'CFC', 'Detection')

        # Surface plot: P vs I → Risk Level
        show_surface_plot(risk_sim, 'P', 'I', 'Risk')



        another = input("\nWould you like to analyze another project? (yes/no): ").lower()
        if another != "yes":
            print("✅ Exiting program. Goodbye!")
            break

