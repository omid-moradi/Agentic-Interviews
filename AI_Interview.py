from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination

from dotenv import load_dotenv
import os


# Load api key
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if api_key is None:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# define model
model_client = OpenAIChatCompletionClient(
    model="gemini-2.5-flash",
    api_key=api_key,
)

# define agents
# 1. Interviewer Agent -> AssistantAgent
interviewer_agent = AssistantAgent(
    name="Interviewer",
    model_client=model_client,
    description=f"an AI Agent that is interviewing for a {job_position} position",
    system_message=f'''
    You are a Professional Interviewer for a {job_position} position.
    Ask one clear question at a time and Wait for user to respond.
    Ask 3 questions in total covering technical skills and experience, problem-solving skills, and communication skills.
    After Asking 3 questions, say "TERMINATE" at the end of the interview.
    '''
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
    system_message=f'''
    You are a Career Coach Specializing in {job_position}.
    You will provide feedback and advice to a {job_position} candidate.
    After the interview, summarize the interviewee performance and provide feedback and advice to the interviewee.
    '''
)
