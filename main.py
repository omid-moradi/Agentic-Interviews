from AI_Interview import team_config, interview
import asyncio

async def main():
    job_position = "Data Scientist"
    team = await team_config(job_position)

    async for ReturnedMsg in interview(team, job_position):
        print("="*69)
        print(ReturnedMsg)

# run
if __name__ == "__main__":
    asyncio.run(main())