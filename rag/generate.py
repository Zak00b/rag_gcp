
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    HarmBlockThreshold,
    HarmCategory,
)

# Load environment variables from .env file
load_dotenv() 

safety_settings = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
}

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-001",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    safety_settings=safety_settings,
)

def transcribe_table_to_text(table: str) -> str:
    messages = [
    (
        "system",
        "Vous êtes un assistant serviable qui peut convertir des tableaux en texte complet en phrases correctes, en utilisant les en-têtes de colonne comme sujets et les valeurs de cellule comme attributs. Capturez toutes les informations et respectez l'alignement. Veuillez ne pas formater le tableau en Markdown ou en HTML et que le texte soit en français."
    ),
    (   "human", 
        table),
    ]

    ai_msg = llm.invoke(messages)
    return ai_msg.content