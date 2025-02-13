from langchain_core.prompts import ChatPromptTemplate
from typing import List
from pydantic import Field
from pydantic import BaseModel as PydanticBaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
import os
from settings import *


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class MovieRecReasoning(BaseModel):
    title: str = Field(description="Title of this particular recommended video")
    genres: list = Field(description="Genres of this particular recommended video")
    sources: list = Field(description="Sources of this particular recommended video")
    rating: float = Field(description="Rating of the recommended video")
    reasoning: str = Field(description="Explanation why this particular recommendation is a good match for user's query")


class RecOutput(BaseModel):
    recommendations: List[MovieRecReasoning] = Field(description="A list of recommendations.")
    reply: str = Field(description="A conversational output with a summarization of five recommendations and why each one is a good fit in one short paragraph.")

class RecRag():
    def __init__(self):
        self.llm = ChatOpenAI(model_name=RAG_LLM, temperature=0.2, frequency_penalty = 0.3, api_key=os.getenv('OPENAI_API_KEY'))
        self.system_prompt = (
        "You are given a context of a list of videos: {context}."
        "Choose five best recommendations based on the user's input query. If in doubt between equally relevant titles, prioritize those with higher rating."
        "Do not invent. Only use the videos provided in the context."
        "Return two things:"
        "1. A list of five recommendations as five jsons with title, genres, sources, rating and reasoning. Make sure that all elements come from the same document."
        "2. A conversational output where you give five recommendations and explain why each one is a good recommendation given the user's query in one short paragraph."
        "Keep the answer concise."
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{input}"),
            ]
        )
        self.rag_chain_from_docs = (
                    {
                        "input": lambda x: x["input"],
                        "context": lambda x: x["context"]
                    }
                    | self.prompt
                    | self.llm.with_structured_output(RecOutput)
                )

        self.chain = RunnablePassthrough.assign(answer=self.rag_chain_from_docs)

    def get_final_recommendations(self, query, retrieved_context):
        output = self.chain.invoke({"input": query, "context": retrieved_context})
        return output['answer']
