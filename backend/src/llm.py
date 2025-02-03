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
    "openai": "https://apis.bioinforcode.com/v1",
    "azure": "https://apis.bioinforcode.com/v1",
    "anthropic": "https://apis.bioinforcode.com/v1",
    "bedrock": "https://apis.bioinforcode.com/v1",
    "ollama": "https://apis.bioinforcode.com/v1",
    "diffbot": "https://apis.bioinforcode.com/v1",
    "groq": "https://apis.bioinforcode.com/v1",
    "bedrock": "https://apis.bioinforcode.com/v1",
    "gemini": "https://apis.bioinforcode.com/v1"
}

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
        # Split configuration and handle both 2-part and 3-part formats
        parts = [p.strip() for p in env_value.split(",")]
        
        if len(parts) == 2:
            model_name, api_key = parts
            api_endpoint = DEFAULT_BASE_URLS.get(
                next((k for k in DEFAULT_BASE_URLS.keys() if k in model.lower()), "openai")
            )
        elif len(parts) == 3:
            model_name, api_endpoint, api_key = parts
        else:
            err = f"Invalid configuration format for model '{model}'. Expected either 'model_name,api_key' or 'model_name,api_endpoint,api_key'"
            logging.error(err)
            raise ValueError(err)
            
        if "fireworks" in model:
            llm = ChatFireworks(api_key=api_key, model=model_name)
        elif "ollama" in model:
            llm = ChatOllama(base_url=api_endpoint or DEFAULT_BASE_URLS["ollama"], model=model_name)
        elif "diffbot" in model:
            llm = ChatDiffbot(diffbot_api_token=api_key)
        elif "bedrock" in model:
            llm = ChatBedrock(
                client=bedrock_client,
                model_id=model_name,
                model_kwargs=dict(temperature=0)
            )
        else:
            # Use ChatOpenAI for all other models (openai, gemini, anthropic, groq, etc.)
            llm = ChatOpenAI(
                api_key=api_key,
                base_url=api_endpoint,
                model=model_name,
                temperature=0,
            )
            logging.info(f"Created ChatOpenAI with model={model_name}, base_url={api_endpoint}")
    except Exception as e:
        err = f"Error while creating LLM '{model}': {str(e)}"
        logging.error(err)
        raise Exception(err)
 
    logging.info(f"Model created - Model Version: {model}")
    return llm, model_name


def get_combined_chunks(chunkId_chunkDoc_list):
    chunks_to_combine = int(os.environ.get("NUMBER_OF_CHUNKS_TO_COMBINE"))
    logging.info(f"Combining {chunks_to_combine} chunks before sending request to LLM")
    combined_chunk_document_list = []
    combined_chunks_page_content = [
        "".join(
            document["chunk_doc"].page_content
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        )
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]
    combined_chunks_ids = [
        [
            document["chunk_id"]
            for document in chunkId_chunkDoc_list[i : i + chunks_to_combine]
        ]
        for i in range(0, len(chunkId_chunkDoc_list), chunks_to_combine)
    ]

    for i in range(len(combined_chunks_page_content)):
        combined_chunk_document_list.append(
            Document(
                page_content=combined_chunks_page_content[i],
                metadata={"combined_chunk_ids": combined_chunks_ids[i]},
            )
        )
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
