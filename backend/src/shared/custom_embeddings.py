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
        batch_size: int = 512,
    ):
        """Initialize the embeddings class."""
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch_size = batch_size
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
        """Embed a list of documents."""
        if not texts:
            return []
        
        # Process in batches to avoid hitting API limits
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_embeddings = self._call_api(batch)
                embeddings.extend([data.get("embedding", []) for data in batch_embeddings])
            except Exception as e:
                logging.error(f"Error embedding batch {i//self.batch_size + 1}: {str(e)}")
                raise
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