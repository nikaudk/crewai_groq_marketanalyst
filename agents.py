
from crewai import Agent


class CustomAgents:
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192",
        )
    def create_agent(self, role):
        descriptions = {
            "Market Analyst": "I specialize in market research and analysis, providing insights that are crucial for strategic planning.",
            "Marketing Strategist": "I develop marketing strategies that effectively target key demographics and maximize market penetration."
        }

        return Agent(
            role=role,
            backstory=descriptions[role],
            goal=f"Develop detailed, actionable insights for {role}.",
            verbose=True,
            llm=self.llm,
            max_rpm=3,
            max_Iter=3
        )

import os
from crewai import Agent
from langchain_groq import ChatGroq

from tools.browser_tools import BrowserTools
from tools.calculator_tools import CalculatorTools
from tools.search_tools import SearchTools
from tools.sec_tools import SECTools

from langchain.tools.yahoo_finance_news import YahooFinanceNewsTool

class StockAnalysisAgents():
    def __init__(self):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama3-70b-8192",
        )
        
    def financial_analyst(self):
        return Agent(
        role='The Best Financial Analyst',
        goal="""Impress all customers with your financial data 
        and market trends analysis""",
        backstory="""The most seasoned financial analyst with 
        lots of expertise in stock market analysis and investment
        strategies that is working for a super important customer.""",
        verbose=True,
        llm=self.llm,
        max_rpm=3,
        max_Iter=3,
        tools=[
            BrowserTools.scrape_and_summarize_website,
            SearchTools.search_internet,
            CalculatorTools.calculate,
            SECTools.search_10q,
            SECTools.search_10k
          ]
        )

    def research_analyst(self):
        return Agent(
        role='Staff Research Analyst',
        goal="""Being the best at gather, interpret data and amaze
        your customer with it""",
        backstory="""Known as the BEST research analyst, you're
        skilled in sifting through news, company announcements, 
        and market sentiments. Now you're working on a super 
        important customer""",
        verbose=True,
        llm=self.llm,
        max_rpm=3,
        max_Iter=3,
        tools=[
            BrowserTools.scrape_and_summarize_website,
            SearchTools.search_internet,
            SearchTools.search_news,
            YahooFinanceNewsTool(),
            SECTools.search_10q,
            SECTools.search_10k
          ]
        )

    def investment_advisor(self):
        return Agent(
        role='Private Investment Advisor',
        goal="""Impress your customers with full analyses over stocks
        and completer investment recommendations""",
        backstory="""You're the most experienced investment advisor
        and you combine various analytical insights to formulate
        strategic investment advice. You are now working for
        a super important customer you need to impress.""",
        verbose=True,
        llm=self.llm,
        max_rpm=3,
        max_Iter=3,
        tools=[
            BrowserTools.scrape_and_summarize_website,
            SearchTools.search_internet,
            SearchTools.search_news,
            CalculatorTools.calculate,
            YahooFinanceNewsTool()
            ]
        )