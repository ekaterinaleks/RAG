import time
import re
import pandas as pd
from unidecode import unidecode
import torch
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import SpladeEncoder
from langchain_core.documents import Document
import itertools
from settings import *

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

class CustomDocumentLoader():
    def __init__(self, device, dense_vector_name=DENSE_VECTOR_MODEL):
        self.dense_vector_model = SentenceTransformer(dense_vector_name, device=device)
        self.sparse_vector_model = SpladeEncoder(max_seq_length=512, device=device)

    def to_unix_time(self, time_stamp):
        try:
            result = pd.Timestamp(time_stamp).timestamp()
        except:
            result = 0
        return result
    
    def normalize_overview(self, text):
        '''
            Remove elements like special characters, urls, emails, double spaces, newline escapes, extra spaces before punctuation, etc.
            Returns None if the value is too short.
        '''
        try:
            text = unidecode(text)
            text = re.sub(r'(www|http)\S+|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}|', '', text)
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'\s+([.,;:!?])', '\\1', text)
            text = re.sub(r'[.,;:!?]*([.,;:!?])', '\\1', text)
            special_characters = 'å¼«¥ª°©ð±§µæ¹¢³®$ł£¥฿₦€/'
            text = text.translate(str.maketrans('', '', special_characters))
            if len(text) < 10:
                text = None
        except:
            text = None
        return text

    def turn_to_records(self, df):
        data_records = df.to_dict('records')
        data = []
        for record in data_records:
            data.append({
                  "id": record["id"],
                  "context": record["page_content"],
                  "sources": record["sources"],
                  "title": record["title"],
                  "original_title": record["original_title"],
                  "overview": record["overview"],
                  "genres": record["genres"],
                  "show_type": record["llm_show_types"],
                  "production_countries": record["production_countries"],
                  "production_companies": record["production_companies"],
                  "release_date": self.to_unix_time(record["release_date"]),
                  "vote_average": record["vote_average"]
              })
        return data

    def async_document_loader(self, records, index, batch_size):
        async_results = []
        it = iter(records)
        chunk = tuple(itertools.islice(it, batch_size))
        while chunk:
            ids = [str(x["id"]) for x in chunk]
            contexts = [x["context"] for x in chunk]
            texts_for_e5 = ["passage: " + x["context"] for x in chunk]
            sources = [list(x['sources'].values()) for x in chunk]
            titles = [x["title"] for x in chunk]
            original_titles = [x["original_title"] for x in chunk]
            overviews = [x["overview"] for x in chunk]
            genres = [x["genres"] for x in chunk]
            show_types = [x["show_type"] for x in chunk]
            production_countries = [x["production_countries"] for x in chunk]
            production_companies  = [x["production_companies"] for x in chunk]
            release_dates = [x["release_date"] for x in chunk]
            ratings = [x["vote_average"] for x in chunk]
           
            dense_vectors = self.dense_vector_model.encode(texts_for_e5).tolist()
            sparse_vectors = self.sparse_vector_model.encode_documents(contexts)
            upsert_chunk = []

            for _id, dense_vector, sparse_vector, context, source, title, original_title, overview, genre, show_type, production_country, production_company, release_date, rating in zip(ids,
                                                                    dense_vectors,
                                                                    sparse_vectors,
                                                                    contexts,
                                                                    sources,
                                                                    titles,
                                                                    original_titles,
                                                                    overviews,
                                                                    genres,
                                                                    show_types,
                                                                    production_countries,
                                                                    production_companies,
                                                                    release_dates,
                                                                    ratings
                                                                    ):
                metadata = {
                    "context": context,
                    "sources": source,
                    "title": title,
                    "original_title": original_title,
                    "overview": overview,
                    "genres": genre,
                    "show_types": show_type,
                    "producion_countries": production_country,
                    "production_companies": production_company,
                    "release_date": release_date,
                    "rating": rating                    
                }
                upsert_chunk.append(
                    {
                        "id": _id,
                        "values": dense_vector,
                        "sparse_values": sparse_vector,
                        "metadata": metadata
                    }
            )
            async_result = index.upsert(vectors=upsert_chunk, async_req=True)
            async_results.append(async_result.get())
            chunk = tuple(itertools.islice(it, batch_size))
        return async_results
    

class CustomRetriever():

    def __init__(self, dense_model_name, device):
        self.dense_model = SentenceTransformer(dense_model_name, device=device)
        self.sparse_model = SpladeEncoder(max_seq_length=512, device=device)
    
        
    def encode_scale_vectors(self, query, alpha):
        """Encode sparse and dense vectors, and apply hybrid vector scaling using a convex combination alpha * dense + (1 - alpha) * sparse
        Args:
        alpha (float): 0 == sparse only, 1 == dense only
        """
        sparse_query = query
        dense_query = "query: " + query
        #create sparse query vector
        sparse = self.sparse_model.encode_queries(sparse_query)
        #create dense query vector
        dense = self.dense_model.encode(dense_query).tolist()
        # scale sparse and dense vectors
        if alpha < 0 or alpha > 1:
            raise ValueError("Alpha must be between 0 and 1")
        hyb_sparse = {
            'indices': sparse['indices'],
            'values':  [v * (1 - alpha) for v in sparse['values']]
        }
        hyb_dense = [v * alpha for v in dense]
        return hyb_dense, hyb_sparse

    def retrieve_from_pinecone(self, index, query, metadata_filter, top_k, alpha):
        hyb_dense, hyb_sparse = self.encode_scale_vectors(query=query, alpha=alpha)
        result = index.query(
            top_k=top_k,
            vector=hyb_dense,
            sparse_vector=hyb_sparse,
            include_metadata=True,
            include_values=False,
            filter=metadata_filter
        )

        docs = [Document(page_content=f"""Title: {item['metadata']['title']}\nDescription:  {item['metadata']['overview']}\nGenres: {', '.join(item['metadata']['genres'])}\nRating: {item['metadata']['rating']}""",
        metadata={"title": item['metadata']['title'],
        "overview": item['metadata']['overview'],
        "genres": item['metadata']["genres"],
        'sources': item['metadata']['sources'],
        "rating": item['metadata']['rating']}) for item in result["matches"]]
        return docs