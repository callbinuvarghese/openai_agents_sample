from dotenv import load_dotenv
import os
import asyncio

# Load environment variables from .env file
load_dotenv()

# Now you can access them via os.environ
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise RuntimeError("OPENAI_API_KEY is not set. Please set it in the .env file.")

from agents import Agent, Runner

"""
 A simple example of an agent handoff:
   (1) A poet writes a poem.
   (2) A triage agent selects the right analyst to provide feedback based on the style.
   (3) There are three analysts each a specialist in a type of poem.
   (4) The selected analyst provides feedback.
"""

useModel = "gpt-4o"

Poet_Agent = Agent(
    name="Poet",
    instructions="You are an expert at writing poetry.",
    model=useModel,
)

Analyst_Agent1 = Agent(
    name="Haiku-Analyst",
    instructions="""Always introduce yourself as 'Helen the Haiku Whisperer'. You are an expert in analyzing poetry. 
You specialize in Haikus. Your output is poetic but concise.""",
    model=useModel,
)

Analyst_Agent2 = Agent(
    name="Free-Verse-Analyst",
    instructions="""Always introduce yourself as 'Fred the Free Verse Expert'. 
You are an expert in analyzing poetry. You specialize in non-rhyming free verse poems. 
Your output is quirky, fun, and kind.""",
    model=useModel,
)

Analyst_Agent3 = Agent(
    name="Rhyming-Poem-Analyst",
    instructions="""Always introduce yourself as 'Ron the Rhyme Master'. 
You are an expert in analyzing poetry. You specialize in classic rhyming poems. 
Your analysis is direct and to the point. You spare no one’s feelings and can be quite verbose.""",
    model=useModel,
)

triage_agent = Agent(
    name="Triage-Agent",
    instructions="""We are going to write a poem. There are three agents that serve as analysts. 
The Poet will write a poem, and based on the style of poem, you will hand it off to the appropriate analyst.""",
    handoffs=[Analyst_Agent1, Analyst_Agent2, Analyst_Agent3],
    model=useModel,
)

# This is your async wrapper
async def main():
    style_of_Poem = "Ballad with dark themes"

    # Step 1: Poet writes the poem
    result = await Runner.run(
        starting_agent=Poet_Agent,
        input=f"""Identify yourself as the Poet. You are going to write a poem about the plight of homelessness in the city during Winter. 
You will write this poem as a {style_of_Poem}. 
Think about this step by step. First describe your approach to the poem, then write the poem."""
    )
    print('\n📜 Poem Draft:\n', result.final_output)

    Draft_Poem = result.final_output

    # Step 2: Triage agent decides who should analyze it
    result = await Runner.run(
        starting_agent=triage_agent,
        input=f"""Here is the first draft from the poet:\n\n{Draft_Poem}\n\n
Determine the best analyst to hand this poem off to for analysis."""
    )
    print('\n🔍 Triage Agent Decision:\n', result.final_output)

    print("\n✅ All done. Carry on.")

# Run the agent chain
if __name__ == "__main__":
    asyncio.run(main())
