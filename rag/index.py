from vectorStoreUtils import VertexAIVectorStore
from utils import load_config

if __name__ == "__main__":

    config = load_config()
    file_path = config["data"]["path"]
    vector_store = VertexAIVectorStore(
        project_id=config["vertexai"]["project_id"], 
        region=config["vertexai"]["region"], 
        index_name=config["vertexai"]["index_name"], 
        index_endpoint_name=config["vertexai"]["index_endpoint_name"], 
        dimensions=config["vertexai"]["dimensions"],
        )



    vector_store.deploy()