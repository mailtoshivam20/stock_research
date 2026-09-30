import os
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

stock = "ICICI Bank"



fundamental_agent = Agent(
    role="Fundamental Analyst",
    goal=f"Analyze the financial health of {stock}.",
    backstory="Expert in company fundamentals, financial statements and valuation.",
    llm=llm,
    verbose=True
)

technical_agent = Agent(
    role="Technical Analyst",
    goal=f"Analyze the price trend and other technical aspects of {stock}.",
    backstory="Expert in stock charts and technical indicators and technical analysis.",
    llm=llm,
    verbose=True
)

news_agent = Agent(
    role="News Analyst",
    goal=f"Analyze recent news about {stock}.",
    backstory="Expert in financial news and market sentiment.",
    llm=llm,
    verbose=True
)

advisor_agent = Agent(
    role="Investment Advisor",
    goal="Provide the final investment recommendation.",
    backstory="Senior investment advisor with years of market experience.",
    llm=llm,
    verbose=True
)



fundamental_task = Task(
    description=f"""
Analyze the fundamentals of {stock}.

Evaluate:
- Revenue
- Net Profit
- EPS
- ROE
- Debt
- Valuation

Give a score out of 10.
""",
    expected_output="Detailed fundamental analysis with score out of 10.",
    agent=fundamental_agent
)

technical_task = Task(
    description=f"""
Analyze the technical trend of {stock}.

Evaluate:
- Trend
- Support
- Resistance
- RSI
- MACD

Give a score out of 10.
""",
    expected_output="Detailed technical analysis with score out of 10.",
    agent=technical_agent
)

news_task = Task(
    description=f"""
Analyze the latest news and sentiment for {stock}.

Return:
- Positive news
- Negative news
- Overall sentiment

Give a score out of 10.
""",
    expected_output="News analysis with sentiment score out of 10.",
    agent=news_agent
)

advisor_task = Task(
    description=f"""
You are the final investment advisor.

Review:
1. Fundamental analysis
2. Technical analysis
3. News analysis

Rules:
- Recommend exactly ONE:
  BUY
  HOLD
  SELL

- Give ONE confidence score between 0 and 100.
- Do not use ranges.
- Output only an integer percentage.
- Explain why in 5-10 bullet points.

Format:

Recommendation: BUY/HOLD/SELL

Confidence: XX%

Risk: Low/Medium/High

Reasons:
- ...
- ...
- ...

""",
    expected_output="""
Recommendation: BUY/HOLD/SELL
Confidence: XX%
Risk: Low/Medium/High
Reasons:
- Bullet points explaining the decision.
""",
    context=[
        fundamental_task,
        technical_task,
        news_task
    ],
    agent=advisor_agent
)



crew = Crew(
    agents=[
        fundamental_agent,
        technical_agent,
        news_agent,
        advisor_agent
    ],
    tasks=[
        fundamental_task,
        technical_task,
        news_task,
        advisor_task
    ],
    process=Process.sequential,
    verbose=True
)



try:
    result = crew.kickoff()
    print("\n")
    print("=" * 60)
    print(result)
    print("=" * 60)

except Exception as e:
    print("ERROR:")
    print(e)