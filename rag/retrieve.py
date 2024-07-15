from vectorStoreUtils import VertexAIVectorStore
from utils import load_config
from langchain_google_vertexai import VertexAIEmbeddings


if __name__ == "__main__":

    config = load_config()
    file_path = config["data"]["path"]


    embedding_model = VertexAIEmbeddings(config["ingest"]["embedding_model"])
    
    vector_store_ai = VertexAIVectorStore(
        project_id=config["vertexai"]["project_id"], 
        region=config["vertexai"]["region"], 
        index_name=config["vertexai"]["index_name"], 
        index_endpoint_name=config["vertexai"]["index_endpoint_name"], 
        dimensions=config["vertexai"]["dimensions"],
        )

    # Ingest text data into Vertex AI Vector Store
    retriever = vector_store_ai.retrieve(embedding_model, config["vertexai"]["data_store_kwargs"])

    print(retriever.invoke("What are my options in breathable fabric?"))




    