# 🎬 RAG-based Movie and Series Recommendation System

## 📌 Overview

This project is a **Retrieval-Augmented Generation (RAG)**-based movie recommendation system. It leverages the following models and tools:
- **Pinecone** as vector database used for storing and retrieval of candidates for recommendation
- **Hybrid search** using dense and sparse vectors calculated with
    - **E5-base-v2** with 12 layers and the embedding size of 768 for embedding dense vectors
    - **Sparse Lexical and Expansion model, or SPLADE** for getting BERT-enriched sparse vectors
- **Llama 3.2 3B model** used locally via **Ollama** for Zero-shot classification
- **OpenAI's GPT-3.5 Turbo model** used to generate final movie recommendations (a combination of structured and conversational response) based on user queries.

## 📂 Project Structure

```
movie_rag.ipynb                     # Main process for the system preparation and use
zero_shot_llm_classification.py     # Extracts categories from user queries and movie metadata using LLM
pinecone_helpers.py                 # Manages embedding, indexing, and retrieval using Pinecone
rag_helpers.py                      # Handles RAG-based reasoning and response generation
tmbd_eda_profile.html               # EDA profile of the TMDB movie dataset used in this project
settings.py                         # Configuration settings (models, etc.)
```


## ✨ Features
- **Zero-shot classification with LLM** for metadata filtering at retrieval stage

    Zero-shot classification is carried out utilizing **Llama 3.2 3B** model via **Ollama** with structured output via Pydantic BaseModel and TypedDict. It is applied to infer key metadata like genres, show types, years, and countries of production for user queries and genres and show types for movies from the database. This enables efficient filtering at the retrieval stage, improving search relevance and reducing unnecessary processing. However, in order to counteract potentially limiting effects if the filters, the filtering is done in three steps with gradually less strict filters to allow for broader spectrum of candidates.

- **Hybrid search** (dense and sparse embeddings) using **Pinecone** index

    The retrieval step is performed using hybrid vector search. The choice of **Pinecone** as vector database is due to both its low-latency retrieval and the option of using SPLADE for hybrid search. Combining dense vectors (in our case, **E5-base-v2** with 768 dimensions) and SPLADE (Sparse Lexical and Expansion) vectors enhances RAG by leveraging the strengths of both semantic and lexical retrieval. Dense vectors capture deep contextual meanings, improving recall for queries with complex or implicit intent, while SPLADE expands sparse keyword representations, ensuring precise keyword matching.

- **Conversational and structured AI output**

    The final AI output generated with **OpenAI's GPT-3.5 Turbo model** combines a conversational user-friendly reply and a structured part with detailed information about each recommendation ranging from metadata extacted from retrieved documents, like title, links to IMDB, etc., and rating to reasoning behind each recommendation in particular.



## ⚙️ Main Steps

1. **Dataset EDA, Cleanup and Feature Engineering**: EDA with pandas profiling, final dataset curation, including inferring categories for imputing missing values (Genres) or adding a new field (Show types).
2. **Indexing**: Movie data is embedded into dense and sparse vectors, and sent to Pinecone with metadata.
3. **User query processing**: The zero-shot classifier extracts from user's query metadata that can be used to construct filters at retrieval stage.
4. **Retrieval**: The system retrieves documents about the most relevant videos for the user's query.
5. **Generation**: The system outputs a conversational response and a curated list of the top 5 movie recommendations with individual reasoning for each recommendation.

## 🎥 Example Query

```
User: "Recommend me some crime, detective or thriller tv-series based on a book and filmed in UK."
```
### Sample Retrieval Logs:
```
INFO:root:Looking for show_types(s): ['TV Series']
INFO:root:Looking for production_countries(s): ['United Kingdom']
INFO:root:Looking for genres(s): ['Crime', 'Detective', 'Thriller']
```

### Sample Output:

```
Reply:
I recommend 'The Woman in White' and 'The Hound of the Baskervilles: Sherlock the Movie' for their adaptation from books and UK filming locations. Additionally, 'Reichenbach Falls' offers a psychological thriller set in Edinburgh, while 'A Very English Murder' presents a classic detective story during Christmas. Lastly, 'Jigsaw' follows local detectives along the coast from Brighton in solving a murder case.

Recommendation 1:
	Title: The Woman in White
	Genres: Crime, Mystery
	Links: ['https://www.themoviedb.org/movie/629696', 'https://www.imdb.com/title/tt0161117']
	Rating: 8.0
	Reasoning: Based on the novel by Wilkie Collins, this detective-mystery film is set in Victorian England, meeting your criteria for a TV series based on a book and filmed in the UK.

Recommendation 2:
	Title: The Hound of the Baskervilles: Sherlock the Movie
	Genres: Mystery, Crime
	Links: ['https://baskervilles-movie.jp/', 'https://www.themoviedb.org/movie/886024', 'https://www.imdb.com/title/tt22814834']
	Rating: 7.0
	Reasoning: This movie version of the TV drama 'Sherlock' is based on the feature-length novel by Arthur Conan Doyle and fits your criteria for a UK-based series adapted from a book.

Recommendation 3:
	Title: Reichenbach Falls
	Genres: Drama, Mystery, TV Movie
	Links: ['http://www.bbc.co.uk/programmes/b0074tgn', 'https://www.themoviedb.org/movie/54802', 'https://www.imdb.com/title/tt0862641']
	Rating: 7.5
	Reasoning: A psychological thriller set in Edinburgh, 'Reichenbach Falls' explores a murder case with dark undertones, meeting your criteria for a UK-based series with a detective theme.

Recommendation 4:
	Title: A Very English Murder
	Genres: Mystery, TV Movie
	Links: ['https://www.themoviedb.org/movie/121270', 'https://www.imdb.com/title/tt0072780']
	Rating: 6.55
	Reasoning: Based on a novel by Cyril Hare, this classic detective story unfolds during Christmas at a country house, aligning with your preference for crime or detective TV series based on books.
    
Recommendation 5:
	Title: Jigsaw
	Genres: Crime, Mystery
	Links: ['https://www.themoviedb.org/movie/93193', 'https://www.imdb.com/title/tt0056121']
	Rating: 7.1
	Reasoning: Set along the coast from Brighton, 'Jigsaw' follows local detectives investigating a murder case, fitting your criteria for a UK-based crime or detective TV series.
```

## 🔮 Future Improvements

- Enhance retrieval stage with Ragatouille reranker.


