from ast import Nonlocal
from heapq import merge
from typing import List, Optional
from src.state.rag_state import RAGState
from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.wikidata.tool import  WikidataQueryRun


class RAGNodes:

    def __init__(self, retriever, llm) -> None:
        self.retriever = retriever
        self.llm = llm
        self.agent = None

    def retrieve_docs(self, state:RAGState)->RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs
        )
    
    def _build_tools(self)->List[Tool]:
        def retriever_tool_fun(query:str)->str:
            docs: List[Document] = self.retriever.invoke(query)

            if not docs:
                return "No doc present"

            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            return "\n\n".join(merged)

        retriever_tool = Tool(
            name = "retriever",
            description="Fetch passages form indexed vectore store",
            func = retriever_tool_fun
        )

        wiki =  WikidataQueryRun(
            api_wrapper=WikipediaAPIWrapper(top_k_results=3, lang = 'en')
        )

        wikiPediaTool = Tool(
            name= 'Wikipedia',
            description = 'Search Wikipedia for general knowledge'
            func = wiki.run
        )

        return [retriever_tool, wikiPediaTool]


    def _build_Agent(self):
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided docs; use 'wikipedia' for general knowledge. "
            "Return only the final useful answer."
        )
        self._agent = create_react_agent(self.llm, tools = tools, prompt=system_prompt)

