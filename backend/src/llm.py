import logging
from langchain.docstore.document import Document
import os
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_vertexai import ChatVertexAI
from langchain_groq import ChatGroq
from langchain_google_vertexai import HarmBlockThreshold, HarmCategory
from langchain_experimental.graph_transformers.diffbot import DiffbotGraphTransformer
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_anthropic import ChatAnthropic
from langchain_fireworks import ChatFireworks
from langchain_aws import ChatBedrock
from langchain_community.chat_models import ChatOllama
import boto3
import google.auth

DEFAULT_BASE_URLS = {
    # Standard models
    "openai": "https://apis.bioinforcode.com/v1",
    "azure": "https://apis.bioinforcode.com/v1",
    "anthropic": "https://apis.bioinforcode.com/v1",
    "gemini": "https://apis.bioinforcode.com/v1",
    "groq": "https://apis.bioinforcode.com/v1",
    
    # Open source models
    "ollama": "https://apis.bioinforcode.com/v1",
    "llama2": "https://apis.bioinforcode.com/v1",
    "mistral": "https://apis.bioinforcode.com/v1",
    "mixtral": "https://apis.bioinforcode.com/v1",
    "phi": "https://apis.bioinforcode.com/v1",
    
    # Cloud provider models
    "bedrock": "https://apis.bioinforcode.com/v1",
    "vertex": "https://apis.bioinforcode.com/v1",
    
    # Other services
    "diffbot": "https://apis.bioinforcode.com/v1",
    "cohere": "https://apis.bioinforcode.com/v1",
    "ai21": "https://apis.bioinforcode.com/v1"
}

# Model families that use ChatOpenAI handler
OPENAI_COMPATIBLE_MODELS = [
    "openai", "azure", "anthropic", "gemini", "groq",
    "llama2", "mistral", "mixtral", "phi", "cohere", "ai21"
]

def get_llm(model: str):
    """Retrieve the specified language model based on the model name."""
    model = model.lower().strip()
    env_key = f"LLM_MODEL_CONFIG_{model}"
    env_value = os.environ.get(env_key)

    if not env_value:
        err = f"Environment variable '{env_key}' is not defined as per format or missing"
        logging.error(err)
        raise Exception(err)
    
    logging.info("Model: {}".format(env_key))
    try:
        # Parse configuration
        parts = [p.strip() for p in env_value.split(",")]
        
        if len(parts) == 2:
            model_name, api_key = parts
            # Find the appropriate base URL by checking model name against known providers
            model_provider = next((k for k in DEFAULT_BASE_URLS.keys() if k in model.lower()), "openai")
            api_endpoint = DEFAULT_BASE_URLS[model_provider]
            logging.info(f"Using default base URL for {model_provider}: {api_endpoint}")
        elif len(parts) == 3:
            model_name, api_endpoint, api_key = parts
            logging.info(f"Using custom base URL: {api_endpoint}")
        else:
            err = f"Invalid configuration format for model '{model}'. Expected either 'model_name,api_key' or 'model_name,api_endpoint,api_key'"
            logging.error(err)
            raise ValueError(err)

        # Determine model type and create appropriate instance
        model_type = next((k for k in DEFAULT_BASE_URLS.keys() if k in model.lower()), None)
        
        if not model_type:
            err = f"Unknown model type for '{model}'. Please add it to DEFAULT_BASE_URLS first."
            logging.error(err)
            raise ValueError(err)
            
        logging.info(f"Creating LLM for model type: {model_type}")
        
        if model_type in OPENAI_COMPATIBLE_MODELS:
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
            )
            logging.info(f"Created ChatOpenAI compatible model: {model_name}")
            
        elif model_type == "fireworks":
            llm = ChatFireworks(api_key=api_key, model=model_name)
            logging.info(f"Created Fireworks model: {model_name}")
            
        elif model_type == "ollama":
            llm = ChatOllama(base_url=api_endpoint, model=model_name)
            logging.info(f"Created Ollama model: {model_name}")
            
        elif model_type == "diffbot":
            llm = ChatDiffbot(diffbot_api_token=api_key)
            logging.info(f"Created Diffbot model")
            
        elif model_type == "bedrock":
            llm = ChatBedrock(
                client=bedrock_client,
                model_id=model_name,
                model_kwargs=dict(temperature=0)
            )
            logging.info(f"Created Bedrock model: {model_name}")
            
        elif model_type == "vertex":
            llm = ChatVertexAI(model=model_name)
            logging.info(f"Created Vertex AI model: {model_name}")
            
        else:
            err = f"Model type '{model_type}' is defined but not implemented. Please add implementation to get_llm function."
            logging.error(err)
            raise NotImplementedError(err)
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name


def get_combined_chunks(chunkId_chunkDoc_list):
    # Maximum number of chunks that can be processed in one batch (API limit)
    MAX_BATCH_SIZE = 25
    # Get configured chunk combination size from env, but ensure it doesn't exceed MAX_BATCH_SIZE
    chunks_to_combine = min(int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE", "6")), MAX_BATCH_SIZE)
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM (max batch size: {MAX_BATCH_SIZE})")
    
    combined_chunk_document_list = []
    
    # Process chunks in batches that respect the MAX_BATCH_SIZE limit
    for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine):
        batch = chunkId_chunkDoc_list[i:i + chunks_to_combine]
        
        # Combine the page content for this batch
        combined_content = "".join(
            document["chunk_doc"].page_content
            for document in batch
        )
        
        # Collect chunk IDs for this batch
        combined_ids = [
            document["chunk_id"]
            for document in batch
        ]
        
        # Create a Document object for the combined batch
        combined_chunk_document_list.append(
            Document(
                page_content=combined_content,
                metadata={"combined_chunk_ids": combined_ids},
            )
        )
        
        # Log batch processing
        logging.debug(f"Processed batch of {len(batch)} chunks")
    
    logging.info(f"Created {len(combined_chunk_document_list)} combined documents")
    return combined_chunk_document_list

def get_chunk_id_as_doc_metadata(chunkId_chunkDoc_list):
    combined_chunk_document_list = [
       Document(
           page_content=document["chunk_doc"].page_content,
           metadata={"chunk_id": [document["chunk_id"]]},
       )
       for document in chunkId_chunkDoc_list
   ]
    return combined_chunk_document_list
      

async def get_graph_document_list(
    llm, combined_chunk_document_list, allowedNodes, allowedRelationship
):
    futures = []
    graph_document_list = []
    if "diffbot_api_key" in dir(llm):
        llm_transformer = llm
    else:
        if "get_name" in dir(llm) and llm.get_name() != "ChatOenAI" or llm.get_name() != "ChatVertexAI" or llm.get_name() != "AzureChatOpenAI":
            node_properties = False
            relationship_properties = False
        else:
            node_properties = ["description"]
            relationship_properties = ["description"]
        llm_transformer = LLMGraphTransformer(
            llm=llm,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            allowed_nodes=allowedNodes,
            allowed_relationships=allowedRelationship,
            ignore_tool_usage=True,
        )
    
    if isinstance(llm,DiffbotGraphTransformer):
        graph_document_list = llm_transformer.convert_to_graph_documents(combined_chunk_document_list)
    else:
        graph_document_list = await llm_transformer.aconvert_to_graph_documents(combined_chunk_document_list)
    return graph_document_list


async def get_graph_from_llm(model, chunkId_chunkDoc_list, allowedNodes, allowedRelationship):
    try:
        llm, model_name = get_llm(model)
        combined_chunk_document_list = get_combined_chunks(chunkId_chunkDoc_list)
        
        if  allowedNodes is None or allowedNodes=="":
            allowedNodes =[]
        else:
            allowedNodes = allowedNodes.split(',')    
        if  allowedRelationship is None or allowedRelationship=="":   
            allowedRelationship=[]
        else:
            allowedRelationship = allowedRelationship.split(',')
            
        graph_document_list = await get_graph_document_list(
            llm, combined_chunk_document_list, allowedNodes, allowedRelationship
        )
        return graph_document_list
    except Exception as e:
        err = f"Error during extracting graph with llm: {e}"
        logging.error(err)
        raise
