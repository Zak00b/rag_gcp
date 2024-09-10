import openparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document
from typing import Iterable
import os 
from utils import load_config
from generate import transcribe_table_to_text 
from logger import logger


current_path = os.getcwd()
doc_path = "senegal_test/tabulartest.pdf"


def save_docs_to_jsonl(array: Iterable[Document], file_path: str) -> None:
    """
    Save a list of documents to a JSONL file.

    Args:
        array (Iterable[Document]): The list of documents to be saved.
        file_path (str): The path to the JSONL file.

    Returns:
        None
    """
    with open(file_path, "w") as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + "\n")


def is_table(node: openparse.Node):
    return node.dict()["variant"] == {'table'}

# Define OpenParse parser
parser = openparse.DocumentParser(table_args={"parsing_algorithm": "pymupdf"})

if __name__ == "__main__":

    # Load the document
    parsed_doc = parser.parse(doc_path)

    config = load_config()

    # Define PyMuPDF loader
    loader = PyMuPDFLoader(doc_path)
    data = loader.load()

    chunks = []
    for i in range(len(parsed_doc.nodes)):
        page_number = parsed_doc.nodes[i].end_page
        title = data[page_number].page_content.split('\n')[0]

        logger.info(
            f"Processing node {i} with page number {page_number}"
        )

        if is_table(parsed_doc.nodes[i]):
            chunks.append(
                Document(page_content= transcribe_table_to_text("le titre est: " + title + "\nle contenu est: " + parsed_doc.nodes[i].text),
                metadata= data[page_number].metadata)
                )
        else:
            chunks.append(
                Document(page_content= "le titre est: " + title + "\nle contenu est: " + parsed_doc.nodes[i].text,
                metadata= data[page_number].metadata)
                )
        
    logger.info("Saving to jsonl...")
    save_path = os.path.join(current_path, config["data"]["path"])
    save_docs_to_jsonl(chunks, save_path)





