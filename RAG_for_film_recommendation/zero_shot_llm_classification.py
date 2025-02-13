from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Optional
from pydantic import BaseModel as PydanticBaseModel
from typing_extensions import Annotated, TypedDict
import pycountry
from pydantic import Field
from tenacity import(
    retry,
    stop_after_attempt,
    wait_random_exponential
)


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True

class MovieDetails(BaseModel):
    """Additional metadata for the TMDB entries."""
    genres: list = Field(description="List of movie genres")
    show_types: list = Field(description="List of show types")

class QueryMetadata(TypedDict):
    """Metadata of the audivisual product the user is looking for."""

    genres: Annotated[list, [], "List of genres the user is interested in"]
    show_types: Annotated[list, [], "List of show types the user is interested in"]
    production_countries: Annotated[Optional[list], None, "A list of full name of countries where it has to be produced"]
    years: Annotated[Optional[list], None, "Year or list of years it has to be filmed in"]


class ZeroShotOllamaClassifier():
    def __init__(self, model_name='llama3.2'):
        self.model_name = model_name
        self.llm = ChatOllama(model= self.model_name, temperature=0.0)     

        self.show_types_ref = ["Movie", "TV Series", "Short", "Music Video", "TV Mini Series"]
        self.country_ref = [country.name for country in list(pycountry.countries)]

        self.tmbd_structured_llm = self.llm.with_structured_output(MovieDetails)
        self.query_structured_llm = self.llm.with_structured_output(QueryMetadata)

        self.tmbd_classification_prompt = ChatPromptTemplate.from_template(
        """
        Analyse the following description.
        Infer the genre of the described audiovisual product choosing all that apply from the following list: {genres}.
        Infer the show type of the described audiovisual product choosing all that apply from the following list: {show_types}.
        Take into account the runtime if available. Videos shorter than 60 minutes tend to be of show type "Short". Videos with more than 300 minutes of runtime are not likely to be a "Movie".
        Input: {input}
        """
        )
        self.query_classification_prompt = ChatPromptTemplate.from_template(
        """
        Analyse user's query for a video recommendation. Extract properties the user has mentioned.
        For example:
        - genres the user might be most interested in. Use the following list as reference: {genres_list}.
        - show types the user might be most interested in. Use the following list as reference: {show_types_list}.
        - countries where it should be produced. Use country names from the following list: {countries_list}. Use only data mentioned directly in the input.
        - potential release year or years. Use only data mentioned directly in the input.
        Output only those properties that you are absolutely sure of based on the user's input.
        Input: {input}
        """
        )
        self.tmbd_classification_chain = self.tmbd_classification_prompt | self.tmbd_structured_llm
        self.query_classification_chain = self.query_classification_prompt | self.query_structured_llm



    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
    def infer_genres_and_types(self, text, genres_ref):
        result = self.tmbd_classification_chain.invoke({'genres_list': genres_ref, 'show_types_list': self.show_types_ref, 'input': text})
        genres = result.genres
        show_types = result.show_types
        return genres, show_types

    @retry(wait=wait_random_exponential(min=1, max=30), stop=stop_after_attempt(3))
    def infer_query_metadata(self, query, genres_ref):
        return self.query_classification_chain.invoke({'genres_list': genres_ref, 'show_types_list': self.show_types_ref, 'countries_list': self.country_ref, 'input': query})


