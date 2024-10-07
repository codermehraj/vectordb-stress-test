# VectorDB Stress Test

This repository contains a Flask-based application that connects to ChromaDB, Weaviate, and PgVector for managing and querying vector embeddings. The service provides APIs for uploading PDF documents and searching for the nearest matches to a given query vector across the three vector databases. I have written a detailed blog on this project, which you can find [here](https://dev.to/codermehraj/stress-testing-vector-databases-dockerizing-a-flask-app-with-chroma-db-pgvector-and-weaviate-running-locally-part-1-34m5/edit).

![Summary of the service made in step 1 of this blog](https://dev-to-uploads.s3.amazonaws.com/uploads/articles/fime4utrpr1sadiu81tg.png)

## Features

- **Upload PDF documents** to store vector embeddings in ChromaDB, Weaviate, and PgVector.
- **Search API** to query the stored embeddings and retrieve the top-k matches.
- Built with **Docker** for easy deployment using `docker-compose`.

## API Endpoints

### Base URL: `localhost:5001`

#### 1. **GET - `/`**
   - Health check endpoint for the API.

#### 2. **POST - `/upload/chroma`**
   - Upload a PDF and store embeddings in ChromaDB.
   - **Body:** Multipart form-data where `pdf` is the form field for the PDF file.

#### 3. **POST - `/search/chroma`**
   - Search for embeddings in ChromaDB.
   - **Body:**
     ```json
     {
         "query": "...",
         "top_k": 2
     }
     ```

#### 4. **POST - `/upload/pgvector`**
   - Upload a PDF and store embeddings in PgVector.
   - **Body:** Multipart form-data where `pdf` is the form field for the PDF file.

#### 5. **POST - `/search/pgvector`**
   - Search for embeddings in PgVector.
   - **Body:**
     ```json
     {
         "query": "...",
         "top_k": 2
     }
     ```

#### 6. **POST - `/upload/weviate`**
   - Upload a PDF and store embeddings in Weaviate.
   - **Body:** Multipart form-data where `pdf` is the form field for the PDF file.

#### 7. **POST - `/search/weviate`**
   - Search for embeddings in Weaviate.
   - **Body:**
     ```json
     {
         "query": "...",
         "top_k": 2
     }
     ```

## Directory Structure

- `/flask-service`: Contains the Flask application and Dockerfile.
- `/postgres`: Contains configurations and Dockerfile for PostgreSQL with PgVector.

## Prerequisites

Make sure you have Docker and Docker Compose installed on your machine.

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/codermehraj/vectordb-stress-test
   ```

2. Navigate to the project directory:

   ```bash
   cd vectordb-stress-test
   ```

3. Build and start the services using Docker Compose:

   ```bash
   docker-compose up --build
   ```

4. The application will be running at `http://localhost:5001`.

## Notes

- Ensure that you have your ChromaDB, Weaviate, and PgVector instances configured correctly.
- API expects a PDF file for upload and JSON with `query` and `top_k` for search.
