import os
from langchain_huggingface import HuggingFaceEmbeddings 
from dotenv import load_dotenv
import pinecone
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from flask import Flask, request, jsonify
from flask_cors import CORS
# from langchain_community.vectorstores import Pinecone
# from langchain_pinecone.vectorstores import Pinecone

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
# Load environment variables
load_dotenv()

# Set API keys
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
os.environ['HUGGINGFACE_API_KEY'] = os.getenv('HUGGINGFACE_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

app = Flask(__name__)
CORS(app)

# Initialize Pinecone instance
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)

pinecone_index = pc.Index(host='https://pdf-vector-db-3e6e42d.svc.aped-4627-b74a.pinecone.io')
# Initialize the language model
llm = ChatGroq(model="mixtral-8x7b-32768")

# Define the prompt template
# Define the prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Please provide the answer based on the given prompt.
    Please provide the accurate answer based on the given prompt.
    <context>
    {context}
    </context>

    Question: {input}
    """
)


text_key="text"
# Initialize Pinecone retriever with text_key and embedding function
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = Pinecone(pinecone_index,embeddings.embed_query,text_key)

# Create a retrieval-based QA chain
retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever.as_retriever(),
    chain_type="stuff",  # Combines documents with the query
    return_source_documents=True
)

# User input and query
text_key = "text"

@app.route("/query",methods=['POST'])
def handle_request():
    data=request.json
    user_query = data.get("query","")
    document_chain = create_stuff_documents_chain(llm,prompt)
    
    # Set up retriever
    retrieval = retriever.as_retriever()

    # Create retrieval chain
    retrieval_chain = create_retrieval_chain(retrieval, document_chain)

    output = retrieval_chain.invoke({"input": user_query})
    # print("Answer:", output['answer'])
    return jsonify({"Answer:": output['answer']})


if __name__ == '__main__':
  port = int(os.environ.get("PORT", 3000))  # Default to 3000 if PORT is not set
  app.run(host="0.0.0.0", port=port, debug=False)