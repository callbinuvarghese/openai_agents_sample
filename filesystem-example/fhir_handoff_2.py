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
    Respond ONLY in the format: ICD code: <code>, Description: <desc>.
    """,
    model="gpt-4o"
)

@function_tool
def get_fhir_observations(icd_code: str) -> str:
    # This is a stub for real FHIR interaction
    return f"Retrieved 3 recent FHIR observations for ICD code: {icd_code}."

# Agent to fetch FHIR observations
# Assume get_fhir_observations is defined elsewhere and decorated with @function_tool
FHIR_Observation_Agent = Agent(
    name="FHIR-Observer",
    instructions="""
    You are a healthcare data specialist. Given an ICD-10 code, retrieve matching FHIR Observations from a medical database.
    Use the `get_fhir_observations` tool to fetch FHIR observations.
    For now, simulate the observation by returning a summary such as:
    'Retrieved 3 recent observations for ICD code: R00.1 (Bradycardia)'.
    """,
    tools=[get_fhir_observations], # Make sure get_fhir_observations is available in scope
    model="gpt-4o"
)


Coordinator_Agent = Agent(
    name="FHIR-Coordinator",
    instructions="""
You are a coordinator agent for clinical workflows.

Given a user query, you must follow this full process:

1. First, hand off to the **ICD-Mapper Agent** to get an ICD-10 code for the medical term in the user query.
2. When you receive the response from the ICD-Mapper Agent, **carefully extract ONLY the ICD-10 code value**. It will be in the format 'ICD code: <code>, Description: <desc>'. You need just the <code> part.
3. Then, use the extracted ICD-10 code value to hand off to the **FHIR-Observer Agent**. Explicitly call the `get_fhir_observations` tool provided by the FHIR-Observer Agent, passing the extracted code as the value for the `icd_code` argument.
4. After the `get_fhir_observations` tool returns the observation result, return a **final combined summary** to the user based on the retrieved observations.

Only return the final response **after both steps are complete**. You must always perform both handoffs and tool calls in this exact order.
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
