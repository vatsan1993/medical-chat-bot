from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embedding
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from src.prompt import *
from dotenv import load_dotenv
import os

app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings_model = download_hugging_face_embedding()

pinecone_client = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

doc_search = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings_model,

)

PROMPT = PromptTemplate.from_template(
    template = prompt_template,
)

chain_type_kwargs = {"prompt": PROMPT}


llm = CTransformers(
    model = "model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type = "llama",
    config = {
        "max_new_tokens": 512,
        "temperature": 0.8,
        "context_length": 4096
    }
)

# Create chain using LCEL

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

retriever = doc_search.as_retriever(search_kwargs={"k": 2})
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)


@app.route("/", methods=["GET"])
def index():
    return render_template("chat.html")


@app.route("/get", methods= ['GET', 'POST'])
def chat():
    msg = request.form["msg"]
    input  = msg
    print(input)
    result = rag_chain.invoke(input)
    print(result)
    return str(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
