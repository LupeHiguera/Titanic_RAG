import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import openai

load_dotenv(dotenv_path='.env')
with open("Text_Data/titanic_wikipedia.txt", "r") as file:
    full_text = file.read()

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n### ", "\n#### ", "\n", " "],
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
)

titanic_wikipedia_txt = splitter.split_text(full_text)

with open("Text_Data/the_loss_of_the_s.s_titanic.txt", "r") as file:
    full_text = file.read()

the_loss_of_the_s_s_titanic_txt = splitter.split_text(full_text)

chunks = the_loss_of_the_s_s_titanic_txt + titanic_wikipedia_txt

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ['OPEN_API_KEY'])
# Create a FAISS vector store
vector_store = FAISS.from_texts(chunks, embeddings)
vector_store.save_local("faiss_titanic_store")

query = "How many people survived the Titanic disaster?"
results = vector_store.similarity_search_with_score(query, k=10)
# print([doc for doc, score in results if score > 0])
for res, score in results:
    print(f"Score: {score}\nContent: {res.page_content}\n")
