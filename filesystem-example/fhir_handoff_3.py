from dotenv import load_dotenv
import os
import asyncio
import re
from loguru import logger

# Load environment variables from .env file
load_dotenv()

# Now you can access them via os.environ
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in the .env file.")


from agents import Agent, Runner,function_tool, trace, gen_trace_id


# --- Tool Stubs ---
# These simulate the actual FHIR API calls

@function_tool
def get_fhir_immunizations(vaccine_description: str = None) -> str:
    """
    Queries simulated FHIR Immunization resources based on a vaccine description.
    Returns a string listing subjects (patients) found in 'Patient/<id>' format.
    If no description is given, might return a general list or error.
    """
    print(f"Tool: get_fhir_immunizations called with vaccine_description='{vaccine_description}'")
    # --- Simulation Logic ---
    if "flu shot" in vaccine_description.lower():
        # Simulate finding some immunizations for patients with these IDs
        return "Simulated Immunizations Found. Subjects: Patient/patient-abc, Patient/patient-def, Patient/patient-ghi."
    elif "measles vaccine" in vaccine_description.lower():
         # Simulate finding some immunizations for different patients
         return "Simulated Immunizations Found. Subjects: Patient/patient-abc, Patient/patient-xyz."
    else:
        return "Simulated Immunizations Found. Subjects: None found for that description."
    # --- End Simulation Logic ---

@function_tool
def get_fhir_patients_by_id(patient_ids: list[str]) -> str:
    """
    Queries simulated FHIR Patient resources by a list of patient reference strings (e.g., 'Patient/patient-id').
    Returns a simulated string listing patient names for the found IDs.
    """
    print(f"Tool: get_fhir_patients_by_id called with patient_ids={patient_ids}")
    # --- Simulation Logic ---
    if not patient_ids:
        return "Tool: get_fhir_patients_by_id received no patient IDs."

    # Map simulated IDs to names
    simulated_patients = {
        "patient-abc": "Alice Smith",
        "patient-def": "Bob Johnson",
        "patient-ghi": "Charlie Brown",
        "patient-xyz": "David Green"
    }

    found_names = []
    for pid_ref in patient_ids:
        # Extract the ID part (after 'Patient/')
        parts = pid_ref.split('/')
        if len(parts) == 2 and parts[0] == 'Patient':
            patient_id = parts[1]
            name = simulated_patients.get(patient_id)
            if name:
                found_names.append(name)
            else:
                found_names.append(f"Unknown Patient (ID: {patient_id})") # Indicate if ID not found in sim
        else:
             found_names.append(f"Invalid ID format: {pid_ref}") # Indicate if format is wrong


    if not found_names:
         return f"Simulated Patients Found for IDs {patient_ids}: None."
    else:
         return f"Simulated Patients Found for IDs {patient_ids}: {', '.join(found_names)}."
    # --- End Simulation Logic ---


# --- Specialized Agents ---

FHIR_Immunization_Agent = Agent(
    name="FHIR-Immunization-Agent",
    instructions="""
    You are a healthcare data agent specializing in searching FHIR Immunization resources.
    Your primary function is to use the `get_fhir_immunizations` tool to find immunization records based on a vaccine description provided as input.
    After using the tool, report the subjects (patients) associated with the found immunizations based on the tool's output.
    """,
    tools=[get_fhir_immunizations],
    model="gpt-4o"
)

FHIR_Patient_Agent = Agent(
    name="FHIR-Patient-Agent",
    instructions="""
    You are a healthcare data agent specializing in retrieving FHIR Patient details.
    Your primary function is to use the `get_fhir_patients_by_id` tool to find patient information based on a list of patient IDs provided as input.
    After using the tool, report the names of the patients found based on the tool's output.
    """,
    tools=[get_fhir_patients_by_id],
    model="gpt-4o"
)


# --- Coordinator Agent ---

ImmunizationPatient_Coordinator = Agent(
    name="ImmunizationPatient-Coordinator",
    instructions="""
You are a coordinator agent for clinical workflows. Your primary task is to find patients based on their immunization records.

Given a user query about patients who received a specific immunization (e.g., "list of patients who had the flu shot"):

1.  **Identify the specific immunization** (e.g., vaccine name or type, like "flu shot" or "measles vaccine") mentioned in the user query.
2.  **Hand off to the FHIR-Immunization-Agent**. Instruct it to query immunizations using the identified immunization description. Call the `get_fhir_immunizations` tool provided by the FHIR-Immunization-Agent, passing the identified immunization description as the value for the `vaccine_description` argument.
3.  When you receive the response from the FHIR-Immunization-Agent, **carefully parse and extract ALL patient reference strings** that are in the format 'Patient/<id>'. Collect these extracted strings into a Python **list of strings**. Make sure to get the full 'Patient/<id>' string for each subject found.
4.  Hand off to the **FHIR-Patient-Agent**. Instruct it to retrieve patient details using the list of patient references you just extracted. Call the `get_fhir_patients_by_id` tool provided by the FHIR-Patient-Agent, passing the list of extracted patient reference strings as the value for the `patient_ids` argument.
5.  When you receive the response from the FHIR-Patient-Agent (which will contain patient names), summarize the list of patients who received the immunization for the user based on the retrieved patient names.

You must perform these steps in order. Only return the final summary after successfully obtaining the list of patients from the FHIR-Patient-Agent. If no immunizations or patients are found at any step, report that to the user in the final summary.
""",
    handoffs=[FHIR_Immunization_Agent, FHIR_Patient_Agent],
    model="gpt-4o" # You can adjust the model if needed
)

#Example runner
async def main():
    query = "Get FHIR Patients who have taken Immunization shots for 'flu shot'"
    with trace("FHIR workflow"):
        result = await Runner.run(starting_agent=ImmunizationPatient_Coordinator, input=query)
        print("\n🩺 Final Output:\n", result.final_output)
# Run the full coordination
if __name__ == "__main__":
    asyncio.run(main())


# The above code is a simulation of a healthcare workflow using FHIR agents to retrieve immunization and patient data.
# ➜  filesystem-example git:(main) ✗ python fhir_handoff_3.py
# Tool: get_fhir_immunizations called with vaccine_description='flu shot'

# 🩺 Final Output:
#  Here are the patients who have taken flu shots:

# - Patient/patient-abc
# - Patient/patient-def
# - Patient/patient-ghi
# ➜