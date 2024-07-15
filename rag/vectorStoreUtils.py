
from google.cloud import aiplatform
from langchain_google_vertexai import (
   VectorSearchVectorStoreDatastore
)

from typing import Optional
from logger import logger

from google.cloud import aiplatform_v1 as aipv1
from google.api_core.client_options import ClientOptions


class VertexAIVectorStore:
    def __init__(self, project_id: str,
                 region: str,
                 index_name: str,
                 index_endpoint_name: Optional[str] = None,
                 dimensions: int = 768 
                 ):
        """
        Initializes a new instance of the Raggcp class.

        Args:
            project_id (str): The ID of the Google Cloud project.
            region (str): The region where the index and index endpoint will be created.
            index_name (str): The name of the index.
            index_endpoint_name (Optional[str], optional): The name of the index endpoint. If not provided, it will be set to "{index_name}_endpoint". Defaults to None.
            dimensions (int, optional): The number of dimensions for the index. Defaults to 768.
        """
        
        self.project_id = project_id
        self.region = region
        self.dimensions = dimensions
        self.index_name = index_name
        self.index_endpoint_name = index_endpoint_name or f"{self.index_name}_endpoint"
        self.PARENT = f"projects/{self.project_id}/locations/{self.region}"

        ENDPOINT = f"{self.region}-aiplatform.googleapis.com"

        # set index client
        self.index_client = aipv1.IndexServiceClient(
            client_options=ClientOptions(api_endpoint=ENDPOINT)
        )
        # set index endpoint client
        self.index_endpoint_client = aipv1.IndexEndpointServiceClient(
            client_options=ClientOptions(api_endpoint=ENDPOINT)
        )

    def get_index(self):
        """
        Retrieves the index with the specified name.

        Returns:
            The index with the specified name, or None if it doesn't exist.
            :rtype: google.cloud.aiplatform_v1.types.index.Index or None
        """
        # Check if index exists
        page_result = self.index_client.list_indexes(
            request=aipv1.ListIndexesRequest(parent=self.PARENT)
        )
        indexes = [
            response.name
            for response in page_result
            if response.display_name == self.index_name
        ]

        if len(indexes) == 0:
            return None

        index_id = indexes[0]
        return self.index_client.get_index(request=aipv1.GetIndexRequest(name=index_id))
    
    def get_index_endpoint(self):
        """
        Retrieves the index endpoint with the specified name.

        Returns:
            The index endpoint object if found, None otherwise.
            :rtype: google.cloud.aiplatform_v1.types.index.IndexEndpoint or None
        """
        
        # Check if index endpoint exists
        page_result = self.index_endpoint_client.list_index_endpoints(
            request=aipv1.ListIndexEndpointsRequest(parent=self.PARENT)
        )
        index_endpoints = [
            response.name
            for response in page_result
            if response.display_name == self.index_endpoint_name
        ]

        if len(index_endpoints) == 0:
            return None

        index_endpoint_id = index_endpoints[0]
        return self.index_endpoint_client.get_index_endpoint(
            request=aipv1.GetIndexEndpointRequest(name=index_endpoint_id)
        )

    def delete_index(self):
        """
        Deletes the matching engine index.
        """

        # Check if index exists
        index = self.get_index()

        # create index if does not exists
        if index:
            # Delete index
            index_id = index.name
            logger.info(f"Deleting Index {self.index_name} with id {index_id}")
            self.index_client.delete_index(name=index_id)
        else:
            raise Exception("Index {index_name} does not exists.")

    def delete_index_endpoint(self):
        """
        Deletes the matching engine index endpoint.
        """

        # Check if index endpoint exists
        index_endpoint = self.get_index_endpoint()

        # Create Index Endpoint if does not exists
        if index_endpoint:
            logger.info(
                f"Index endpoint {self.index_endpoint_name}  exists with resource "
                + f"name as {index_endpoint.name} and endpoint domain name as "
                + f"{index_endpoint.public_endpoint_domain_name}"
            )

            index_endpoint_id = index_endpoint.name
            index_endpoint = self.index_endpoint_client.get_index_endpoint(
                name=index_endpoint_id
            )
            # Undeploy existing indexes
            for d_index in index_endpoint.deployed_indexes:
                logger.info(
                    f"Undeploying index with id {d_index.id} from Index endpoint {self.index_endpoint_name}"
                )
                request = aipv1.UndeployIndexRequest(
                    index_endpoint=index_endpoint_id, deployed_index_id=d_index.id
                )
                r = self.index_endpoint_client.undeploy_index(request=request)
                response = r.result()
                logger.info(response)

            # Delete index endpoint
            logger.info(
                f"Deleting Index endpoint {self.index_endpoint_name} with id {index_endpoint_id}"
            )
            self.index_endpoint_client.delete_index_endpoint(name=index_endpoint_id)
        else:
            raise Exception(
                f"Index endpoint {self.index_endpoint_name} does not exists."
            )
        
    def delete_all(self):
        """
        Deletes the matching engine index and endpoint.
        """
        self.delete_index_endpoint()
        self.delete_index()

    def create_index(self):
        """
        Creates a matching engine index with the specified parameters.

        Returns:
            The created matching engine index.
            :rtype: google.cloud.aiplatform.matching_engine.matching_engine_index.MatchingEngineIndex

        """

        # Get index
        index = self.get_index()
        # Create index if does not exists
        if index:
            logger.info(f"Index {self.index_name} already exists with id {index.name}")
            return aiplatform.MatchingEngineIndex(index_name=index.name)
        
        logger.info(f"Index {self.index_name} does not exists. Creating index ...")
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=self.index_name,
            location=self.region,
            dimensions=self.dimensions,
            approximate_neighbors_count=150,
            distance_measure_type="DOT_PRODUCT_DISTANCE",
            index_update_method="STREAM_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE
        )

        logger.info(f"Index created: {index.display_name}")
        return index

    def create_endpoint(self):
        """
        Creates an endpoint for the matching engine index.

        Returns:
            The created index endpoint.
            :rtype: google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint.MatchingEngineIndexEndpoint
        """

        # Get index endpoint if exists
        index_endpoint = self.get_index_endpoint()
        # Create Index Endpoint if does not exists
        if index_endpoint:
            logger.info(
                f"Index endpoint {self.index_endpoint_name} already exists with resource "
                + f"name as {index_endpoint.name} and endpoint domain name as "
                + f"{index_endpoint.public_endpoint_domain_name}"
            )
            return aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=index_endpoint.name)
        
        logger.info(
            f"Index endpoint {self.index_endpoint_name} does not exists. Creating index endpoint..."
        )
        # Create an endpoint
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=self.index_endpoint_name, 
            location=self.region,
            public_endpoint_enabled=True
        )

        logger.info(f"Endpoint created: {index_endpoint.display_name}")
        return index_endpoint

    def deploy(self):
        # NOTE : This operation can take upto 20 minutes
        """
        Deploys the index to the endpoint.

        Returns:
            None
        """
        # Deploy index to endpoint
        index = self.create_index()
        index_endpoint = self.create_endpoint()
        deployed_index_id = index_endpoint.display_name

        index_endpoint = index_endpoint.deploy_index(
            index=index, 
            deployed_index_id=deployed_index_id
        )
        logger.info(f"Index deployed to endpoint: {index_endpoint.display_name}")
        logger.info(f"Deployed indexes: {index_endpoint.deployed_indexes}")

    def get_vector_store(self, embedding_model, datastore_client_kwargs):
        # https://python.langchain.com/v0.2/docs/integrations/retrievers/google_vertex_ai_search/

        # add filters
        # https://github.com/langchain-ai/langchain/issues/5073

        """
            Parameters:
            - embedding_model: The embedding model used for vectorization.
            - datastore_client_kwargs: The datastore client kwargs.

            Returns:
            - vector_store: The vector store object for the matching engine index.
        """

        vector_store = VectorSearchVectorStoreDatastore.from_components(
            project_id=self.project_id,
            region=self.region,
            index_id=self.get_index().name,
            endpoint_id=self.get_index_endpoint().name,
            embedding=embedding_model,
            datastore_client_kwargs=datastore_client_kwargs,
            stream_update=True,
        )
        return vector_store

    def upsert(self, texts, metadatas, embedding_model, datastore_client_kwargs):
        # create datastore first 
        """
        Upserts the given texts and metadatas into the specified index.

        Parameters:
        - texts (list): A list of texts to be upserted.
        - metadatas (list): A list of metadata associated with each text.
        - index_endpoint (IndexEndpoint): The index endpoint to upsert the data into.
        - index (Index): The index to upsert the data into.
        - embedding_model: The embedding model used for vectorization.
        - datastore_client_kwargs: The datastore client kwargs.

        Returns:
        None
        """
        
        vector_store = self.get_vector_store(embedding_model, datastore_client_kwargs)
        vector_store.add_texts(texts=texts, metadatas=metadatas, is_complete_overwrite=True)
        logger.info("Upserted data to the index.")

    def retrieve(self, embedding_model, datastore_client_kwargs):
        """
        Retrieves the vector store using the specified embedding model and datastore client arguments.

        Parameters:
        - embedding_model: The embedding model to use for retrieval.
        - datastore_client_kwargs: The keyword arguments to pass to the datastore client.

        Returns:
        - retriever: The retriever object for the vector store.
        """
        vector_store = self.get_vector_store(embedding_model, datastore_client_kwargs)
        return vector_store.as_retriever()

if __name__ == "__main__":
    print("Hello World")