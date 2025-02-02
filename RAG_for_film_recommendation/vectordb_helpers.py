import time
import re
from unidecode import unidecode
import torch
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import SpladeEncoder
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import itertools
from settings import *

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

def to_unix_time(time_stamp):
    return int(time.mktime(time_stamp.timetuple()))
    
def normalize_overview(text, max_len = DESC_LENGTH):
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
        text = text[:max_len]
        if len(text) < 10:
            text = None
    except:
        text = None
    return text

class CustomDocumentLoader():
    def __init__(self, device, dense_vector_name=DENSE_VECTOR_MODEL):
        self.dense_vector_model = SentenceTransformer(dense_vector_name, device=device)
        self.sparse_vector_model = SpladeEncoder(max_seq_length=512, device=device)

    def turn_to_records(self, df_name):
        data_records = df_name.to_dict('records')
        data = []
        for record in data_records:
            data.append(
              {
                  "id": record["id"],
                  "context": 
                  "genres": list(record["genres"])
        
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
           
            dense_vectors = self.dense_vector_model.encode(texts_for_e5).tolist()
            sparse_vectors = self.sparse_vector_model.encode_documents(contexts)
            upsert_chunk = []
            for _id, dense_vector, sparse_vector, context, in zip(ids,
                                   dense_vectors,
                                   sparse_vectors,
                                   contexts,
                                  ):
                metadata = {
                    "context": context,
                    
                    
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