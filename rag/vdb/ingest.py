"""Load html from files, clean up, split, ingest into Qdrant."""
from rag.vdb.vectorStoreUtils import VertexAIVectorStore
from utils import load_config
from langchain_google_vertexai import VertexAIEmbeddings

# Input text with metadata
record_data = [
    {
        "description": "A versatile pair of dark-wash denim jeans."
        "Made from durable cotton with a classic straight-leg cut, these jeans"
        " transition easily from casual days to dressier occasions.",
        "price": 65.00,
        "color": "blue",
        "season": ["fall", "winter", "spring"],
    },
    {
        "description": "A lightweight linen button-down shirt in a crisp white."
        " Perfect for keeping cool with breathable fabric and a relaxed fit.",
        "price": 34.99,
        "color": "white",
        "season": ["summer", "spring"],
    },
    {
        "description": "A soft, chunky knit sweater in a vibrant forest green. "
        "The oversized fit and cozy wool blend make this ideal for staying warm "
        "when the temperature drops.",
        "price": 89.99,
        "color": "green",
        "season": ["fall", "winter"],
    },
    {
        "description": "A classic crewneck t-shirt in a soft, heathered blue. "
        "Made from comfortable cotton jersey, this t-shirt is a wardrobe essential "
        "that works for every season.",
        "price": 19.99,
        "color": "blue",
        "season": ["fall", "winter", "summer", "spring"],
    },
    {
        "description": "A flowing midi-skirt in a delicate floral print. "
        "Lightweight and airy, this skirt adds a touch of feminine style "
        "to warmer days.",
        "price": 45.00,
        "color": "white",
        "season": ["spring", "summer"],
    },
]



# Parse and prepare input data
texts = []
metadatas = []
for record in record_data:
    record = record.copy()
    page_content = record.pop("description")
    texts.append(page_content)
    if isinstance(page_content, str):
        metadata = {**record}
        metadatas.append(metadata)


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
    vector_store_ai.upsert(texts, metadatas, embedding_model, config["vertexai"]["data_store_kwargs"])




    