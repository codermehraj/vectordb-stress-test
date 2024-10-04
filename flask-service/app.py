# app.py

import http
import os
from weaviate.classes.config import Property, DataType
from flask import Flask, json, request, jsonify
import chromadb
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
import uuid
import psycopg2
import weaviate
from weaviate.classes.query import MetadataQuery
import time
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import pymupdf
import logging
# import json

app = Flask(__name__)

weaviate_client = None

weaviate_host = os.getenv("WEAVIATE_URL", "localhost")
pg_host = os.getenv("PG_URL", "localhost")
logging.basicConfig(level=logging.INFO)

try:
    weaviate_client = weaviate.connect_to_local(host=weaviate_host)
    print("Weaviate client connected successfully")
except Exception as e:
    print("[Retrying] Error connecting to Weaviate client:", e)
    time.sleep(5)
    weaviate_client = weaviate.connect_to_local(host=weaviate_host)
    print("Weaviate client connected successfully")

try:
    # Note that you can use `client.collections.create_from_dict()` to create a collection from a v3-client-style JSON object
    weviate_collection = weaviate_client.collections.create(
        name="DocumentSearch",
        properties=[
            Property(name="document", data_type=DataType.TEXT),
            Property(name="fileName", data_type=DataType.TEXT),
            Property(name="chunkNo", data_type=DataType.INT),
            # Property(name="vector", data_type=DataType.NUMBER_ARRAY),
        ]
    )
    print("Collection created successfully")
except Exception as e:
    if "Collection already exists" in str(e):
        weviate_collection = weaviate_client.collections.get(name="DocumentSearch")
        print("Collection already exists")


print("connectting to pg " , pg_host)

pgvector_conn = psycopg2.connect(
    user="myuser",
    password="mypassword",
    host=pg_host,
    port=5432,  # The port you exposed in docker-compose.yml
    database="mydb"
)

chroma_client = chromadb.PersistentClient(
    path="test",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

# Create or get a collection named 'embeddings'
collection = chroma_client.get_or_create_collection("embeddings")

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Define a projection matrix to transform the 384-dim embedding to 512-dim
projection_matrix = np.random.rand(384, 512)

try:
    path = os.path.dirname(os.path.abspath(__file__))
    upload_folder=os.path.join(
    path.replace("/file_folder",""),"tmp")
    os.makedirs(upload_folder, exist_ok=True)
    app.config['uploads'] = upload_folder
except Exception as e:
    app.logger.info('An error occurred while creating temp folder')
    app.logger.error('Exception occurred : {}'.format(e))

def chunk_text(text, chunk_size):
    words = text.split()  # Split the text into words
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        # Get the chunk of words from index i to i + chunk_size
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    
    return chunks

def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', max_length=512, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        embedding = embeddings.squeeze().numpy()
    
    # Project the 384-dim embedding to 512-dim using the projection matrix
    projected_embedding = np.dot(embedding, projection_matrix)
    
    return projected_embedding.tolist()

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    """
    API to generate an embedding from a given text.
    Expects a 'text' query parameter.
    """
    
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400
    
    print("Received payload for embedding :", data)  # Debug print statement
    
    text = data.get('text')

    if not text:
        return jsonify({"error": "Missing 'text' query parameter"}), 400

    print("Received text:", text)  # Debug print statement

    try:
        # Generate the embedding using the pre-trained model
        embedding = generate_embedding(text)
        # print("Generated embedding:", embedding)  # Debug print statement
        return jsonify(embedding), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

########################## CHROMA DB API ENDPOINTS ##########################

@app.route('/add_embedding_chroma', methods=['POST'])
def add_embedding_chroma():
    """
    API to add a vector embedding to Chroma DB.
    Expects JSON payload with 'embedding' and optional 'metadata'.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    document = data.get('document')
    embedding = data.get('embedding')
    metadata = data.get('metadata', {})

    print("Received embedding:", embedding)  # Debug print statement

    if not embedding or not isinstance(embedding, list):
        return jsonify({"error": "'embedding' must be a list of numbers"}), 400

    # Generate a unique ID for the embedding
    embedding_id = str(uuid.uuid4())

    try:
        collection.add(
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[embedding_id],
            documents=[document]
        )
        print("Embedding added successfully")
        print(embedding)
        return jsonify({"message": "Embedding added successfully", "id": embedding_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_embeddings_chroma', methods=['GET'])
def get_embeddings_chroma():
    """
    API to retrieve all vector embeddings from Chroma DB.
    """
    try:
        # Retrieve all embeddings with their IDs and metadata        
        results = collection.get(include=['embeddings', 'documents', 'metadatas'])

        # Debugging print statements
        print("Raw results from collection.get():", results)

        # Check if 'embeddings' is None
        if results['embeddings'] is None:
            return jsonify({"error": "No embeddings found"}), 404

        documents = results['documents']
        embeddings = results['embeddings']
        metadatas = results['metadatas']
        ids = results['ids']

        # Ensure the embeddings, metadatas, and ids are JSON serializable
        # documents = [list(map(str, doc)) for doc in documents]  # Convert documents to string
        embeddings = [list(map(float, emb)) for emb in embeddings]
        metadatas = [dict(meta) for meta in metadatas]
        ids = [str(eid) for eid in ids]

        # Combine into a list of dictionaries
        data = []
        for eid, emb, meta in zip(ids, embeddings, metadatas):
            tmp = {
                "id": eid,
                "embedding": emb,
                "metadata": meta,
                "document": documents  # Change this if documents are meant to be singular
            }
            data.append(tmp)

        return jsonify({"embeddings": data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_embeddings_chroma', methods=['POST'])
def search_embeddings_chroma():
    """
    API to search for the top N embeddings that match a given query embedding.
    Expects JSON payload with 'query_embedding' and optional 'top_k' parameter.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    query_embedding = data.get('query_embedding')
    top_k = data.get('top_k', 2)  # Default to top 2 matches

    if not query_embedding or not isinstance(query_embedding, list):
        return jsonify({"error": "'query_embedding' must be a list of numbers"}), 400

    try:
        # Perform a similarity search in Chroma DB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        
        # print("Raw results from collection.query():", results)  # Debug print statement

        # Extract the results                
        matched_metadatas = results['metadatas']
        matched_documents = results['documents']
        
        print("Matched metadatas:", matched_metadatas)  # Debug print statement

        # Prepare response
        data = []
        for meta, doc in zip(matched_metadatas, matched_documents):
            data.append({
                "metadata": meta,
                "document": doc  # Change this if documents are meant to be singular
            })

        return jsonify({"matches": data}), 200

    except Exception as e:        
        return jsonify({"error": str(e)}), 500


########################## PGVECTOR ENDPOINTS ##########################
@app.route('/add_embedding_pgvector', methods=['POST'])
def add_embedding_pgvector():    
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    document = data.get('document')
    embedding = data.get('embedding')
    metadata = data.get('metadata', {})

    # print("Received embedding:", embedding)  # Debug print statement

    if not embedding or not isinstance(embedding, list):
        return jsonify({"error": "'embedding' must be a list of numbers"}), 400

    # Generate a unique ID for the embedding
    embedding_id = str(uuid.uuid4())

    try:
        pg_cursor = pgvector_conn.cursor()
        
        # print embedding dimensions
        print("Embedding dimensions: ", len(embedding))
        
        pg_cursor.execute(
                "INSERT INTO items (document, embedding, FileName, chunkNo) VALUES (%s, %s, %s, %s)",
                (document, embedding, metadata['FileName'], metadata['ChunkNo'])
            )
        
        # pg_cursor.connection.commit()
        pg_cursor.close()
        
                        
        print("Embedding added successfully")
        print(embedding)
        return jsonify({"message": "Embedding added successfully"}), 201
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/get_embeddings_pgvector', methods=['GET'])
def get_embeddings_pgvector():
    """
    API to retrieve all vector embeddings from PGVector DB.
    """
    try:
        # Retrieve all embeddings with their IDs and metadata        
        pg_cursor = pgvector_conn.cursor()
        
        pg_cursor.execute(
            "SELECT id, document, embedding, FileName, chunkNo FROM items"
        )
        
        results = []
        
        print("ID, CONTENT, EMBEDDING, FILENAME, CHUNKNO")
        print("-------------------------------------------------")        
        
        for row in pg_cursor.fetchall():
            results.append(row)
            print(f"ID: {row[0]}, CONTENT: {row[1]}, embedding: {row[2]} FileName: {row[3]}, chunkNo: {row[4]}")
            
        pg_cursor.close()
        

        # documents = pg_cursor.fetchall(0)
        # embeddings = 
        # metadatas = 
        # ids = 

        # Ensure the embeddings, metadatas, and ids are JSON serializable
        # documents = [list(map(str, doc)) for doc in documents]  # Convert documents to string
        # embeddings = [list(map(float, emb)) for emb in embeddings]
        # metadatas = [dict(meta) for meta in metadatas]
        # ids = [str(eid) for eid in ids]

        # Combine into a list of dictionaries
        # data = []
        # for eid, emb, meta in zip(ids, embeddings, metadatas):
        #     tmp = {
        #         "id": eid,
        #         "embedding": emb,
        #         "metadata": meta,
        #         "document": documents  # Change this if documents are meant to be singular
        #     }
        #     data.append(tmp)

        return jsonify({"embeddings": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_embeddings_pgvector', methods=['POST'])
def search_embeddings_pgvector():
    """
    API to search for the top N embeddings that match a given query embedding.
    Expects JSON payload with 'query_embedding' and optional 'top_k' parameter.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    query_embedding = data.get('query_embedding')
    top_k = data.get('top_k', 2)  # Default to top 2 matches

    if not query_embedding or not isinstance(query_embedding, list):
        return jsonify({"error": "'query_embedding' must be a list of numbers"}), 400

    try:
        # Perform a similarity search in PGVECTOR DB
        pg_cursor = pgvector_conn.cursor()
        
        # Perform a cosine similarity search
        pg_cursor.execute(
            """SELECT id, document, FileName, chunkNo, 1 - (embedding <=> %s::vector) AS cosine_similarity
               FROM items
               ORDER BY cosine_similarity DESC LIMIT %s""",
            (query_embedding, top_k)                    
        )
        
        results = []
        
        for row in pg_cursor.fetchall():
            results.append({
                "id": row[0],
                "document": row[1],
                "FileName": row[2],
                "chunkNo": row[3],
                "cosine_similarity": row[4]
            })
            print(f"ID: {row[0]}, CONTENT: {row[1]}, FileName: {row[2]}, chunkNo: {row[3]} - Cosine Similarity: {row[4]}")
        
        # print("Raw results from collection.query():", results)  # Debug print statement

        # Extract the results                
        # matched_metadatas = results['metadatas']
        # matched_documents = results['documents']
        
        # print("Matched metadatas:", matched_metadatas)  # Debug print statement

        # # Prepare response
        # data = []
        # for meta, doc in zip(matched_metadatas, matched_documents):
        #     data.append({
        #         "metadata": meta,
        #         "document": doc  # Change this if documents are meant to be singular
        #     })

        return jsonify({"matches": results}), 200

    except Exception as e:        
        return jsonify({"error": str(e)}), 500

########################## WEVIATE API ENDPOINTS ##########################

@app.route('/add_embedding_weviate', methods=['POST'])
def add_embedding_weviate():
    """
    API to add a vector embedding to Weviate DB.
    Expects JSON payload with 'embedding' and optional 'metadata'.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    document = data.get('document')
    embedding = data.get('embedding')
    metadata = data.get('metadata', {})

    print("Received embedding:", embedding)  # Debug print statement

    if not embedding or not isinstance(embedding, list):
        return jsonify({"error": "'embedding' must be a list of numbers"}), 400

    # Generate a unique ID for the embedding
    embedding_id = str(uuid.uuid4())

    try:
        weviate_collection = weaviate_client.collections.get(name="DocumentSearch")
        weviate_collection.data.insert({"document": document, "fileName": metadata['FileName'], "chunkNo": metadata['ChunkNo']}, vector=embedding)
        
        # with weaviate_client.batch as batch:
        #     properties = {"document": document, "metadata": metadata}
        #     batch.add_data_object(properties, "DocumentSearch", vector=embedding)
        
        print("Embedding added successfully")
        print(embedding)
        return jsonify({"message": "Embedding added successfully", "id": embedding_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/get_embeddings_weviate', methods=['GET'])
def get_embeddings_weviate():
    """
    API to retrieve all vector embeddings from Weviate DB.
    """
    try:
        # Retrieve all embeddings with their IDs and metadata        
        # results = collection.get(include=['embeddings', 'documents', 'metadatas'])
        results = []
        
        print("Retrieving embeddings from Weviate DB")
        weviate_collection = weaviate_client.collections.get(name="DocumentSearch")
        for item in weviate_collection.iterator(include_vector=True):
            results.append(item)
            # print(item.uuid, item.properties)
        

        # Debugging print statements
        print("Raw results from collection.get():", results)

        # # Check if 'embeddings' is None
        # if results['embeddings'] is None:
        #     return jsonify({"error": "No embeddings found"}), 404

        # documents = results['documents']
        # embeddings = results['embeddings']
        # metadatas = results['metadatas']
        # ids = results['ids']

        # # Ensure the embeddings, metadatas, and ids are JSON serializable
        # # documents = [list(map(str, doc)) for doc in documents]  # Convert documents to string
        # embeddings = [list(map(float, emb)) for emb in embeddings]
        # metadatas = [dict(meta) for meta in metadatas]
        # ids = [str(eid) for eid in ids]

        # # Combine into a list of dictionaries
        # data = []
        # for eid, emb, meta in zip(ids, embeddings, metadatas):
        #     tmp = {
        #         "id": eid,
        #         "embedding": emb,
        #         "metadata": meta,
        #         "document": documents  # Change this if documents are meant to be singular
        #     }
        #     data.append(tmp)

        return jsonify({"embeddings": str(results)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/search_embeddings_weviate', methods=['POST'])
def search_embeddings_weviate():
    """
    API to search for the top N embeddings that match a given query embedding.
    Expects JSON payload with 'query_embedding' and optional 'top_k' parameter.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "Invalid JSON payload"}), 400

    query_embedding = data.get('query_embedding')
    top_k = data.get('top_k', 2)  # Default to top 2 matches

    if not query_embedding or not isinstance(query_embedding, list):
        return jsonify({"error": "'query_embedding' must be a list of numbers"}), 400

    try:
        # Perform a similarity search in Weviate DB
        weviate_collection = weaviate_client.collections.get(name="DocumentSearch")
        
        result = weviate_collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        
        results = []
        
        print("QUERY RESULT -------------------------------")
        for o in result.objects:
            results.append({
                "properties": o.properties,
                "distance": o.metadata.distance
            })
            print(o.properties)
            print(o.metadata.distance)
        
        # result = weaviate_client.query.get("DocumentSearch", ["document", "metadata"]).with_near_vector({
        #     "vector": query_embedding        
        # }).with_limit(top_k).with_additional(['distance']).do()
        
        # print("Raw results from collection.query():", result)  # Debug print statement
        
        # results = json.dumps(result, indent=2)
        
        # print("Raw results from collection.query():", results)  # Debug print statement

        # Extract the results                
        # matched_metadatas = results['metadatas']
        # matched_documents = results['documents']
        
        # print("Matched metadatas:", matched_metadatas)  # Debug print statement

        # # Prepare response
        # data = []
        # for meta, doc in zip(matched_metadatas, matched_documents):
        #     data.append({
        #         "metadata": meta,
        #         "document": doc  # Change this if documents are meant to be singular
        #     })

        return jsonify({"matches": results}), 200

    except Exception as e:        
        return jsonify({"error": str(e)}), 500

########################## PDF PROCESS ENDPOINTS CHROMA ##########################
@app.route('/upload/chroma', methods=['POST'])
def upload_pdf_chroma():
    """
    API to upload a PDF document and extract embeddings using Chroma DB.
    Expects a PDF file in the 'pdf' form-data field.
    """
    if 'pdf' not in request.files:
        logging.error("No pdf given")
        return jsonify({"error": "No pdf given"}), 400

    file = request.files['pdf']

    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the PDF file to disk
        logging.info("Uploading PDF file:" + file.filename)
        
        file_path = os.path.join(app.config.get('uploads') , file.filename)
        file.save(file_path)
        
        # file_path = os.path.join("uploads", file.filename)
        # file.save(file_path)
        
        logging.info("PDF uploaded successfully:" + file.filename)
         
        doc = pymupdf.open(file_path)

        # Extract text from the PDF
        full_text = ""
        for page in doc: # iterate the document pages
            text = page.get_text()
            full_text += text
        
        # Log the full text extracted from the PDF
        short_text = " ".join(full_text.split()[:20]) + "..."
        logging.info("Full text extracted from PDF:")
        logging.info(short_text)
        
        # Chunk the text into smaller parts
        chunk_size = 50

        chunks = chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            short_text = str(i + 1) + ") ".join(chunk.split()[:10]) + "..."
            logging.info("Embedding Chunk: ")
            logging.info(short_text)

            # Generate an embedding for the text
            embedding = generate_embedding(chunk)
            
            embedding_id = str(uuid.uuid4())

            # Add the embedding to Chroma DB
            try:
                collection.add(
                    embeddings=[embedding],
                    metadatas=[{
                        "fileName": file.filename,
                        "chunkNo": i + 1
                    }],
                    ids=[embedding_id],
                    documents=[chunk]
                )
                logging.info("Embedding added successfully for the chunk")
                # print(embedding)                
            except Exception as e:
                logging.error("ERROR WHILE STORING IN CHROMA => ", str(e))
                return jsonify({"error": str(e)}), 500        
        
        return jsonify({"message": "PDF uploaded and embeddings extracted successfully"}), 201
    
    except Exception as e:
        logging.error("ERROR WHILE STORING EMBEDDING FROM PDF => ", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/search/chroma', methods=['POST'])
def search_query_chroma():
    """
    API to search for the top N embeddings that match a given query embedding.
    Expects a text in the 'query' form-data field.
    """
    
    data = request.get_json()

    if not data:
        logging.error("Invalid JSON payload")
        return jsonify({"error": "Invalid JSON payload"}), 400

    query = data.get('query')
    top_k = data.get('top_k', 2)  # Default to top 2 matches
    
    logging.info("Received query:" + query)  # Debug print statement
    
    if not query:
        logging.error("Missing 'query' parameter")
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        # Generate an embedding for the query text
        query_embedding = generate_embedding(query)
        
        logging.info("Generated embedding for query")  # Debug print statement
        logging.info("Searching for similar embeddings in DB")  # Debug print statement
        
        # Perform a similarity search in Chroma DB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=['documents', 'metadatas']
        )
        
        # print("Raw results from collection.query():", results)  # Debug print statement
        logging.info("DB QUEERY SUCCESSFUL")  # Debug print statement

        # Extract the results                
        matched_metadatas = results['metadatas']
        matched_documents = results['documents']
        
        # print("Matched metadatas:", matched_metadatas)  # Debug print statement

        # Prepare response
        data = []
        for meta, doc in zip(matched_metadatas, matched_documents):
            data.append({
                "metadata": meta,
                "document": doc  # Change this if documents are meant to be singular
            })

        return jsonify({"matches": data}), 200
    
    except Exception as e:
        logging.error("ERROR WHILE SEARCHING IN CHROMA => ", str(e))
        return jsonify({"error": str(e)}), 500
    

########################## PDF PROCESS ENDPOINTS PGVECTOR ##########################
@app.route('/upload/pgvector', methods=['POST'])
def upload_pdf_pgvector():
    """
    API to upload a PDF document and extract embeddings using pgvector DB.
    Expects a PDF file in the 'pdf' form-data field.
    """
    if 'pdf' not in request.files:
        logging.error("No pdf given")
        return jsonify({"error": "No pdf given"}), 400

    file = request.files['pdf']

    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the PDF file to disk
        logging.info("Uploading PDF file:")
        logging.info(file.filename)
        
        file_path = os.path.join(app.config.get('uploads') , file.filename)
        file.save(file_path)
        
        # file_path = os.path.join("uploads", file.filename)
        # file.save(file_path)
        
        logging.info("PDF uploaded successfully:")
        
        doc = pymupdf.open(file_path)

        # Extract text from the PDF
        full_text = ""
        for page in doc: # iterate the document pages
            text = page.get_text()
            full_text += text
        
        short_text = " ".join(full_text.split()[:20]) + "..."
        logging.info("Full text extracted from PDF:")
        logging.info(short_text)
        
        # Chunk the text into smaller parts
        chunk_size = 50

        chunks = chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            
            short_text = str(i + 1) + ") ".join(chunk.split()[:10]) + "..."
            logging.info("Embedding Chunk: ")
            logging.info(short_text)

            # Generate an embedding for the text
            embedding = generate_embedding(chunk)
            
            # embedding_id = str(uuid.uuid4())

            # Add the embedding to pgvector DB
            try:
                pg_cursor = pgvector_conn.cursor()
        
                # print embedding dimensions
                print("Embedding dimensions: ", len(embedding))
                
                pg_cursor.execute(
                        "INSERT INTO items (document, embedding, FileName, chunkNo) VALUES (%s, %s, %s, %s)",
                        (chunk, embedding, file.filename, i + 1)
                    )
                
                # pg_cursor.connection.commit()
                pg_cursor.close()
                
                logging.info("Embedding added successfully for chunk")
                                                
                print(embedding)
                return jsonify({"message": "Embedding added successfully"}), 201
            except Exception as e:
                logging.error("ERROR WHILE STORING IN PGVECTOR => ", str(e))
                return jsonify({"error": str(e)}), 500    
        
        return jsonify({"message": "PDF uploaded and embeddings extracted successfully"}), 201
    
    except Exception as e:
        logging.error("ERROR WHILE STORING EMBEDDING FROM PDF => ", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/search/pgvector', methods=['POST'])
def search_query_pgvector():
    """
    API to search for the top N embeddings that match a given query embedding.
    Expects a text in the 'query' form-data field.
    """
    
    data = request.get_json()

    if not data:
        logging.error("Invalid JSON payload")
        return jsonify({"error": "Invalid JSON payload"}), 400

    query = data.get('query')
    top_k = data.get('top_k', 2)  # Default to top 2 matches
    
    logging.info("Received query:", query)  # Debug print statement
    
    if not query:
        logging.error("Missing 'query' parameter")
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        # Generate an embedding for the query text
        query_embedding = generate_embedding(query)
        
        logging.info("Generated embedding for query")  # Debug print statement
        logging.info("Searching for similar embeddings in DB")  # Debug print statement
        
        pg_cursor = pgvector_conn.cursor()
        
        # Perform a cosine similarity search
        pg_cursor.execute(
            """SELECT id, document, FileName, chunkNo, 1 - (embedding <=> %s::vector) AS cosine_similarity
               FROM items
               ORDER BY cosine_similarity DESC LIMIT %s""",
            (query_embedding, top_k)                    
        )        
        
        results = []
        
        for row in pg_cursor.fetchall():
            results.append({
                "id": row[0],
                "document": row[1],
                "FileName": row[2],
                "chunkNo": row[3],
                "cosine_similarity": row[4]
            })
            print(f"ID: {row[0]}, CONTENT: {row[1]}, FileName: {row[2]}, chunkNo: {row[3]} - Cosine Similarity: {row[4]}")
        
        # print("Raw results from collection.query():", results)  # Debug print statement

        # Extract the results                
        # matched_metadatas = results['metadatas']
        # matched_documents = results['documents']
        
        # print("Matched metadatas:", matched_metadatas)  # Debug print statement

        # # Prepare response
        # data = []
        # for meta, doc in zip(matched_metadatas, matched_documents):
        #     data.append({
        #         "metadata": meta,
        #         "document": doc  # Change this if documents are meant to be singular
        #     })
        
        logging.info("DB QUEERY SUCCESSFUL")  # Debug print statement

        return jsonify({"matches": results}), 200

    except Exception as e:        
        return jsonify({"error": str(e)}), 500
    

########################## PDF PROCESS ENDPOINTS WEVIATE ##########################
@app.route('/upload/weviate', methods=['POST'])
def upload_pdf_weviate():
    """
    API to upload a PDF document and extract embeddings using Weviate DB.
    Expects a PDF file in the 'pdf' form-data field.
    """
    if 'pdf' not in request.files:
        logging.error("No pdf given")
        return jsonify({"error": "No pdf given"}), 400

    file = request.files['pdf']

    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the PDF file to disk
        logging.info("Uploading PDF file:")
        logging.info(file.filename)
        
        file_path = os.path.join(app.config.get('uploads') , file.filename)
        file.save(file_path)
        
        # file_path = os.path.join("uploads", file.filename)
        # file.save(file_path)
        
        logging.info("PDF uploaded successfully")
        
        doc = pymupdf.open(file_path)

        # Extract text from the PDF
        full_text = ""
        for page in doc: # iterate the document pages
            text = page.get_text()
            full_text += text
            
        short_text = " ".join(full_text.split()[:20]) + "..."
        logging.info("Full text extracted from PDF:")
        logging.info(short_text)
        
        # Chunk the text into smaller parts
        chunk_size = 50

        chunks = chunk_text(text, chunk_size)
        for i, chunk in enumerate(chunks):
            short_text = str(i + 1) + ") ".join(chunk.split()[:10]) + "..."
            logging.info("Embedding Chunk: ")
            logging.info(short_text)

            # Generate an embedding for the text
            embedding = generate_embedding(chunk)
            
            # embedding_id = str(uuid.uuid4())

            # Add the embedding to Weviate DB
            try:
                weviate_collection = weaviate_client.collections.get(name="DocumentSearch")
                weviate_collection.data.insert({"document": chunk, "fileName": file.filename, "chunkNo": i + 1}, vector=embedding)
                
                # with weaviate_client.batch as batch:
                #     properties = {"document": document, "metadata": metadata}
                #     batch.add_data_object(properties, "DocumentSearch", vector=embedding)
                
                logging.info("Embedding added successfully for chunk")
                # print(embedding)                
            except Exception as e:
                logging.error("ERROR WHILE STORING IN WEVIATE => ", str(e))
                return jsonify({"error": str(e)}), 500        
        
        return jsonify({"message": "PDF uploaded and embeddings extracted successfully"}), 201
    
    except Exception as e:
        logging.error("ERROR WHILE STORING EMBEDDING FROM PDF => ", str(e))
        return jsonify({"error": str(e)}), 500
    
@app.route('/search/weviate', methods=['POST'])
def search_query_weviate():
    """
    API to search for the top N embeddings that match a given query embedding.
    Expects a text in the 'query' form-data field.
    """
    
    data = request.get_json()

    if not data:
        logging.error("Invalid JSON payload")
        return jsonify({"error": "Invalid JSON payload"}), 400

    query = data.get('query')
    top_k = data.get('top_k', 2)
    
    logging.info("Received query:", query)  # Debug print statement
    
    if not query:
        logging.error("Missing 'query' parameter")
        return jsonify({"error": "Missing 'query' parameter"}), 400
    
    try:
        # Generate an embedding for the query text
        query_embedding = generate_embedding(query)
        
        logging.info("Generated embedding for query")  # Debug print statement
        logging.info("Searching for similar embeddings in DB")  # Debug print statement
        
        # Perform a similarity search in Weviate DB
        weviate_collection = weaviate_client.collections.get(name="DocumentSearch")
        
        result = weviate_collection.query.near_vector(
            near_vector=query_embedding,
            limit=top_k,
            return_metadata=MetadataQuery(distance=True)
        )
        
        results = []
        
        # print("QUERY RESULT -------------------------------")
        for o in result.objects:
            results.append({
                "properties": o.properties,
                "distance": o.metadata.distance
            })
            # print(o.properties)
            # print(o.metadata.distance)
        
        logging.info("DB QUEERY SUCCESSFUL")  # Debug print statement
        
        return jsonify({"matches": results}), 200
    
    except Exception as e:
        logging.error("ERROR WHILE SEARCHING IN WEVIATE => ", str(e))
        return jsonify({"error": str(e)}), 500
    


@app.route('/')
def index():
    return "Flask DB Service is Running!", 200


if __name__ == '__main__':
    # Run the Flask app
    # print('weviate_is_ready:', weaviate_client.is_ready())
    app.run(host='0.0.0.0', port=5001, debug=True)
