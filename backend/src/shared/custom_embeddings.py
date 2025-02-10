from typing import Any, Dict, List, Optional
from langchain_core.embeddings import Embeddings
import requests
import logging
import os
import urllib3
import ssl
from requests.adapters import HTTPAdapter
from urllib3.poolmanager import PoolManager

# Disable SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TLSAdapter(HTTPAdapter):
    def init_poolmanager(self, connections, maxsize, block=False):
        """Create and initialize the urllib3 PoolManager with custom SSL context."""
        ctx = ssl.create_default_context()
        ctx.set_ciphers('DEFAULT@SECLEVEL=1')
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_version=ssl.PROTOCOL_TLSv1_2,
            ssl_context=ctx
        )

class CustomBGEEmbeddings(Embeddings):
    """Custom embeddings class for BGE-M3 API."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "bge-m3",
        batch_size: Optional[int] = None,
    ):
        """Initialize the embeddings class."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        # Enforce maximum batch size of 25 for API limit
        self.batch_size = min(batch_size or 25, 25)
        logging.info(f"Initialized embeddings with batch size: {self.batch_size}")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        # Create a session with our custom TLS adapter
        self.session = requests.Session()
        self.session.mount('https://', TLSAdapter())
        logging.warning("Using custom TLS configuration for API requests")

    def _call_api(self, texts: List[str]) -> List[List[float]]:
        """Call the embeddings API."""
        try:
            # Format request body according to BGE-M3 API expectations
            request_body = {
                "model": self.model,
                "input": texts,
                "encoding_format": "float",
                "normalize": True
            }
            logging.info(f"Making API request to {self.base_url}/embeddings")
            logging.debug(f"Request body: {request_body}")
            
            response = self.session.post(
                f"{self.base_url}/embeddings",
                headers=self.headers,
                json=request_body,
                timeout=30
            )
            
            # Log response details for debugging
            if not response.ok:
                error_msg = f"API Error Response (Status {response.status_code}): {response.text}"
                logging.error(error_msg)
                try:
                    error_json = response.json()
                    if 'error' in error_json:
                        error_msg = f"API Error Details: {error_json['error']}"
                        logging.error(error_msg)
                except:
                    pass
                response.raise_for_status()
                
            result = response.json()
            if "data" not in result:
                error_msg = f"Unexpected API response format: {result}"
                logging.error(error_msg)
                raise ValueError(error_msg)
                
            return result["data"]
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Request error: {str(e)}"
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_details = e.response.json()
                    error_msg += f"\nAPI Error Details: {error_details}"
                except:
                    error_msg += f"\nResponse Text: {e.response.text}"
            logging.error(error_msg)
            raise
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embeddings, one for each input text.
            
        Raises:
            ValueError: If the input is invalid or empty.
            Exception: If there's an error during the embedding process.
        """
        if not texts:
            return []
            
        total_texts = len(texts)
        logging.info(f"Embedding {total_texts} texts with batch size {self.batch_size}")
        
        # Process in batches to respect API limits
        embeddings = []
        total_batches = (total_texts + self.batch_size - 1) // self.batch_size
        
        for i in range(0, total_texts, self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            
            try:
                logging.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                batch_embeddings = self._call_api(batch)
                embeddings.extend([data.get("embedding", []) for data in batch_embeddings])
                logging.debug(f"Successfully embedded batch {batch_num}")
                
            except Exception as e:
                error_msg = f"Error embedding batch {batch_num}/{total_batches}: {str(e)}"
                if "超出批处理上限" in str(e):
                    error_msg = f"Batch size exceeds API limit (max: 25). Current batch size: {len(batch)}. Please reduce NUMBER_OF_CHUNKS_TO_COMBINE in .env"
                logging.error(error_msg)
                raise Exception(error_msg) from e
                
        logging.info(f"Successfully embedded all {total_texts} texts in {total_batches} batches")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        try:
            embeddings = self._call_api([text])
            if not embeddings or "embedding" not in embeddings[0]:
                raise ValueError("No embedding found in API response")
            return embeddings[0]["embedding"]
        except Exception as e:
            logging.error(f"Error embedding query: {str(e)}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async version of embed_documents."""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async version of embed_query."""
        return self.embed_query(text)