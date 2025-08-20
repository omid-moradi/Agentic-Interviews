from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console

from dotenv import load_dotenv
import asyncio
import os


# Load api key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# define model
model_client = OpenAIChatCompletionClient(
    model="gemini-2.0-flash",
    api_key=api_key,
)

# define agents
# 1. Interviewer Agent -> AssistantAgent
job_position = "Data Scientist"

interviewer_system_message = f'''
You are a professional interviewer for a {job_position} position. Your goal is to conduct the interview by asking a total of 3 questions.

**Interview Flow:**
1.  Ask your first question and wait for the candidate's answer.
2.  After the candidate answers, a 'Career_Coach' agent will provide feedback to the candidate.
3.  After the coach speaks, you will ask your next question.
4.  Repeat this process for all 3 questions.
5.  After the candidate's final answer and the coach's final feedback, end the conversation with the word "TERMINATE".

**--- CRITICAL RULE ---**
**You MUST completely ignore any and all messages from the 'Career_Coach'. The coach's feedback is NOT for you. Your entire focus is on the candidate's direct answers. Do NOT comment on, refer to, or acknowledge the coach's messages in any way. Proceed with your next question as if the coach never spoke.**
'''

career_coach_system_message = f'''
You are a world-class Career Coach specializing in {job_position}.
Your role is to provide **immediate, concise, and actionable feedback** directly to the candidate after each of their answers.

**Your Task:**
1.  After the candidate answers a question from the interviewer, it is your turn to speak.
2.  Address the candidate directly (e.g., "That was a solid answer...").
3.  Provide **one key strength** and **one specific area for improvement** for that single answer.
4.  Keep your feedback brief and to the point (5-10 sentences). Do not summarize the whole interview.
'''

interviewer_agent = AssistantAgent(
    name="Interviewer",
    model_client=model_client,
    description=f"an AI Agent that is interviewing for a {job_position} position",
    system_message=interviewer_system_message
)
# 2. Interviewee Agent -> UserProxyAgent
interviewee = UserProxyAgent(
    name="Interviewee",
    description=f"an Agent that simulates a candidate for a {job_position} position",
    input_func=input
)
# 3. Career Coach Agent -> AssistantAgent
career_coach = AssistantAgent(
    name="CareerCoach",
    model_client=model_client,
    description=f"an AI Agent that provides feedback and advice to a {job_position} candidate",
    system_message=career_coach_system_message
)

# define group chat
termination_condition = TextMentionTermination(text="TERMINATE")

team = RoundRobinGroupChat(
    participants=[interviewer_agent, interviewee, career_coach],
    termination_condition=termination_condition
)
# running the interview
stream = team.run_stream(task=f"conducting an interview for a {job_position} position")

# run the console
async def main():
    await Console(stream=stream)

# run
if __name__ == "__main__":
    asyncio.run(main())