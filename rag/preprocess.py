import openparse
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.docstore.document import Document

doc_path = "../senegal_test/bf_splitted.pdf"

# Define OpenParse parser
parser = openparse.DocumentParser(table_args={"parsing_algorithm": "pymupdf"})

if __name__ == "__main__":

    # Load the document
    parsed_doc = parser.parse(doc_path)

    # Define PyMuPDF loader
    loader = PyMuPDFLoader(doc_path)
    data = loader.load()

    chunks = []
    for i in range(len(parsed_doc.nodes)):
        page_number = parsed_doc.nodes[i].end_page
        title = data[page_number].page_content.split('\n')[0]
        chunks.append(
            Document(page_content= "le titre est: " + title + " le contenu est: " + parsed_doc.nodes[i].text,
            metadata= data[page_number].metadata)
            )
        
    print(chunks)






