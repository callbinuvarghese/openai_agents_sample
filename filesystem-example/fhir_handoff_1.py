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

# Agent to map terms to ICD codes
ICD_Mapping_Agent = Agent(
    name="ICD-Mapper",
    instructions="""
    You are a medical assistant that converts symptom names or common medical terms into appropriate ICD-10 codes.
    Given a phrase like 'heart beat', return the most relevant ICD-10 code and a short explanation.
    Respond in the format: ICD code: <code>, Description: <desc>.
    """,
    model="gpt-4o"
)

@function_tool
def get_fhir_observations(icd_code: str) -> str:
    # This is a stub for real FHIR interaction
    return f"Retrieved 3 recent FHIR observations for ICD code: {icd_code}."

# Agent to fetch FHIR observations
FHIR_Observation_Agent = Agent(
    name="FHIR-Observer",
    instructions="""
    You are a healthcare data specialist. Given an ICD-10 code, retrieve matching FHIR Observations from a medical database.
    Use the `get_fhir_observations` tool to fetch FHIR observations.
    For now, simulate the observation by returning a summary such as:
    'Retrieved 3 recent observations for ICD code: R00.1 (Bradycardia)'.
    """,
    tools=[get_fhir_observations],
    model="gpt-4o"
)

# Coordinator agent with handoffs
Coordinator_Agent = Agent(
    name="FHIR-Coordinator",
    instructions="""
    You are a coordinator agent. Given a medical term like 'Heart Beat',
    your goal is to:
    1. Handoff to the ICD-Mapper agent to retrieve the ICD code.
    2. Then handoff to the FHIR-Observer to fetch observations using that code.
    Return the full observation summary to the user.
    """,
    handoffs=[ICD_Mapping_Agent, FHIR_Observation_Agent],
    model="gpt-4o"
)



# Example runner
async def main():
    query = "Get FHIR observations for ICD code of 'Heart Beat'"
    with trace("FHIR workflow"):
        result = await Runner.run(starting_agent=Coordinator_Agent, input=query)
        print("\n🩺 Final Output:\n", result.final_output)
        icd_code_match = re.search(r"ICD code:\s*(\w+\.\w+|\w+)", result.final_output)
        logger.debug(f"ICD code match: {icd_code_match}")
        if icd_code_match:
            icd_code = icd_code_match.group(1)
            logger.debug(f"Extracted ICD code: {icd_code}")
            # Then hand off
            result = await Runner.run(
                starting_agent=FHIR_Observation_Agent,
                input=f"ICD code: {icd_code}"
            )
            print("\n🩺 Final Output:\n", result.final_output)
        else:
            logger.error("ICD code not found in mapping agent output.")
            raise ValueError("ICD code not found in mapping agent output.")
# Run the full coordination
if __name__ == "__main__":
    asyncio.run(main())
