package main

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/go-resty/resty/v2"
	"github.com/gorilla/mux"
	"github.com/oliverpool/unipdf/v3/extractor"
	"github.com/oliverpool/unipdf/v3/model"
)

// Extract text from the PDF file
func extractTextFromPDF(pdfFile *os.File) (string, error) {

	fmt.Println(pdfFile.Name())

	// open pdf file and extract text
	// f, err := os.Open(pdfFile.Name())
	f, err := os.Open(pdfFile.Name())
	if err != nil {
		log.Fatalf("Failed to open PDF: %v\n", err)
	}
	defer f.Close()
	pdfReader, err := model.NewPdfReader(f)
	if err != nil {
		log.Fatalf("Failed to read PDF: %v\n", err)
	}
	numPages, err := pdfReader.GetNumPages()
	if err != nil {
		log.Fatalf("Failed to read PDF: %v\n", err)
	}

	fullText := ""

	// fmt.Printf("--------------------\n")
	// fmt.Printf("PDF to text extraction:\n")
	// fmt.Printf("--------------------\n")
	for i := 0; i < numPages; i++ {
		pageNum := i + 1

		page, err := pdfReader.GetPage(pageNum)
		if err != nil {
			log.Fatalf("Failed to extract page: %v\n", err)
		}

		ex, err := extractor.New(page)
		if err != nil {
			log.Fatalf("Failed to extract: %v\n", err)
		}

		text, err := ex.ExtractText()
		if err != nil {
			log.Fatalf("Failed to extract text: %v\n", err)
		}

		// fmt.Println("------------------------------")
		// fmt.Printf("Page %d:\n", pageNum)
		// fmt.Printf("\"%s\"\n", text)
		// fmt.Println("------------------------------")
		fullText += text
	}

	// fmt.Printf("------------------------------\n")
	// fmt.Printf("Full text:\n")
	// fmt.Printf("\"%s\"\n", fullText)
	// fmt.Printf("------------------------------\n")

	return fullText, nil
}

// Chunk the extracted text into smaller pieces
func chunkText(text string, chunkSize int) []string {
	words := strings.Split(text, " ")
	var chunks []string
	for i := 0; i < len(words); i += chunkSize {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunks = append(chunks, strings.Join(words[i:end], " "))
	}
	return chunks
}

// Call the Python script to generate embeddings
func generateEmbedding(giventext string) ([]float64, error) {
	// fmt.Println("Generating embedding for text: ", text)
	client := resty.New()
	fmt.Println("Calling Python service for embedding...")

	type EmbeddingRequest struct {
		Text string `json:"text"`
	}

	data := EmbeddingRequest{
		Text: giventext,
	}

	jsonData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
	}

	fmt.Println("Genereting embedding from python...")
	fmt.Println("Text: ", giventext)
	fmt.Println("Data: ", data)
	fmt.Println("jsonData: ", jsonData)

	resp, err := client.R().
		SetHeader("Content-Type", "application/json").
		SetBody(data).
		Post("http://flask-service:5001/get_embedding")

	fmt.Println("Embedding Response: ", resp)

	if err != nil {
		fmt.Println("Error calling Python service: ", err)
	}

	if resp.IsError() {
		fmt.Printf("error response from while generating embedding: %s error ms: %s", resp.Status(), resp.String())
	}
	// cmd := exec.Command("python", "embedding.py", text)
	// output, err := cmd.Output()
	fmt.Println("Output: ")
	fmt.Println(resp)
	if err != nil {
		fmt.Println("Error calling Python script: ", err)
		return nil, err
	}

	var embedding []float64
	fmt.Println("Unmarshalling output...")
	err = json.Unmarshal(resp.Body(), &embedding)
	if err != nil {
		return nil, err
	}
	fmt.Println("Embedding: ", embedding)
	return embedding, nil
}

// Store the embeddings in Chroma DB
func storeEmbeddingInChroma(id string, embedding []float64, document string, fileName string, chunkNo int) error {
	client := resty.New()

	// Struct to hold metadata
	type Metadata struct {
		FileName string `json:"FileName"`
		ChunkNo  int    `json:"ChunkNo"`
	}

	// Struct to hold the JSON structure
	type EmbeddingRequest struct {
		ID        string    `json:"id"`
		Document  string    `json:"document"`
		Embedding []float64 `json:"embedding"`
		Metadata  Metadata  `json:"metadata"`
	}

	data := EmbeddingRequest{
		ID:        id,
		Embedding: embedding,
		Document:  document,
		Metadata: Metadata{
			FileName: fileName,
			ChunkNo:  chunkNo,
		},
	}

	// data := map[string]interface{}{
	// 	"document":   document,
	// 	"embeddings": []interface{}{embedding},
	// 	"metadata": []interface{}{map[string]interface{}{
	// 		"filename": fileName,
	// 		"chunk":    chunkNo,
	// 	}},
	// }

	jsonData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
		return err
	}

	fmt.Println("Storing embedding in Chroma DB...")
	fmt.Println("Data: ", string(jsonData))

	resp, err := client.R().
		SetHeader("Content-Type", "application/json").
		SetBody(data).
		Post("http://flask-service:5001/add_embedding_chroma")

	if err != nil {
		return err
	}

	if resp.IsError() {
		return fmt.Errorf("error response from Chroma DB: %s error ms: %s", resp.Status(), resp.String())
	}
	return nil
}

// Store the embeddings in PGVECTOR DB
func storeEmbeddingInPgVector(id string, embedding []float64, document string, fileName string, chunkNo int) error {
	client := resty.New()

	// Struct to hold metadata
	type Metadata struct {
		FileName string `json:"FileName"`
		ChunkNo  int    `json:"ChunkNo"`
	}

	// Struct to hold the JSON structure
	type EmbeddingRequest struct {
		ID        string    `json:"id"`
		Document  string    `json:"document"`
		Embedding []float64 `json:"embedding"`
		Metadata  Metadata  `json:"metadata"`
	}

	data := EmbeddingRequest{
		ID:        id,
		Embedding: embedding,
		Document:  document,
		Metadata: Metadata{
			FileName: fileName,
			ChunkNo:  chunkNo,
		},
	}

	// data := map[string]interface{}{
	// 	"document":   document,
	// 	"embeddings": []interface{}{embedding},
	// 	"metadata": []interface{}{map[string]interface{}{
	// 		"filename": fileName,
	// 		"chunk":    chunkNo,
	// 	}},
	// }

	jsonData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
		return err
	}

	fmt.Println("Storing embedding in PG Vector...")
	fmt.Println("Data: ", string(jsonData))

	resp, err := client.R().
		SetHeader("Content-Type", "application/json").
		SetBody(data).
		Post("http://flask-service:5001/add_embedding_pgvector")

	if err != nil {
		return err
	}

	if resp.IsError() {
		return fmt.Errorf("error response from PGVECTOR DB: %s error ms: %s", resp.Status(), resp.String())
	}
	return nil
}

// Store the embeddings in Weviate
func storeEmbeddingInWeviate(id string, embedding []float64, document string, fileName string, chunkNo int) error {
	client := resty.New()

	// Struct to hold metadata
	type Metadata struct {
		FileName string `json:"FileName"`
		ChunkNo  int    `json:"ChunkNo"`
	}

	// Struct to hold the JSON structure
	type EmbeddingRequest struct {
		ID        string    `json:"id"`
		Document  string    `json:"document"`
		Embedding []float64 `json:"embedding"`
		Metadata  Metadata  `json:"metadata"`
	}

	data := EmbeddingRequest{
		ID:        id,
		Embedding: embedding,
		Document:  document,
		Metadata: Metadata{
			FileName: fileName,
			ChunkNo:  chunkNo,
		},
	}

	// data := map[string]interface{}{
	// 	"document":   document,
	// 	"embeddings": []interface{}{embedding},
	// 	"metadata": []interface{}{map[string]interface{}{
	// 		"filename": fileName,
	// 		"chunk":    chunkNo,
	// 	}},
	// }

	jsonData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
		return err
	}

	fmt.Println("Storing embedding in Weviate DB...")
	fmt.Println("Data: ", string(jsonData))

	resp, err := client.R().
		SetHeader("Content-Type", "application/json").
		SetBody(data).
		Post("http://flask-service:5001/add_embedding_weviate")

	if err != nil {
		return err
	}

	if resp.IsError() {
		return fmt.Errorf("error response from Weviate DB: %s error ms: %s", resp.Status(), resp.String())
	}
	return nil
}

// Process the PDF: extract text, chunk it, generate embeddings, and store in Chroma DB
func processPDF(filePath *os.File, dbName string) error {
	text, err := extractTextFromPDF(filePath)
	fmt.Print("Extracted text: --------------------------------------")
	fmt.Print(text)
	if err != nil {
		return fmt.Errorf("failed to extract text: %v", err)
	}

	chunks := chunkText(text, 500)

	for i, chunk := range chunks {
		embedding, err := generateEmbedding(chunk)
		if err != nil {
			return fmt.Errorf("failed to generate embedding: %v", err)
		}

		if dbName == "chroma" {
			err = storeEmbeddingInChroma(fmt.Sprintf("%s_chunk_%d", filePath.Name(), i), embedding, chunk, filePath.Name(), i)
			if err != nil {
				return fmt.Errorf("failed to store embedding: %v", err)
			}
		} else if dbName == "pgvector" {
			err = storeEmbeddingInPgVector(fmt.Sprintf("%s_chunk_%d", filePath.Name(), i), embedding, chunk, filePath.Name(), i)
			if err != nil {
				return fmt.Errorf("failed to store embedding: %v", err)
			}
		} else if dbName == "weviate" {
			err = storeEmbeddingInWeviate(fmt.Sprintf("%s_chunk_%d", filePath.Name(), i), embedding, chunk, filePath.Name(), i)
			if err != nil {
				return fmt.Errorf("failed to store embedding: %v", err)
			}
		} else {
			return fmt.Errorf("invalid DB name: %s", dbName)
		}
	}

	return nil
}

// Handler for the POST endpoint to upload PDF
func uploadPDFHandler(w http.ResponseWriter, r *http.Request) {

	// get db name from the request
	dbName := r.FormValue("db")
	fmt.Println("DB Name: ", dbName)

	// Parse the uploaded file
	file, _, err := r.FormFile("pdf")
	if err != nil {
		http.Error(w, "Invalid file upload", http.StatusBadRequest)
		return
	}
	defer file.Close()

	// Create a temporary file
	// Ensure the "uploads" directory exists
	if _, err := os.Stat("uploads"); os.IsNotExist(err) {
		err = os.Mkdir("uploads", 0755)
		if err != nil {
			http.Error(w, "Unable to create uploads directory", http.StatusInternalServerError)
			return
		}
	}

	tempFile, err := os.CreateTemp("uploads", "*.pdf")
	if err != nil {
		http.Error(w, "Unable to create temp file", http.StatusInternalServerError)
		return
	}
	defer os.Remove(tempFile.Name()) // Clean up the temp file later

	// Write the uploaded content to the temp file
	_, err = io.Copy(tempFile, file) // Use io.Copy to write file content
	if err != nil {
		http.Error(w, "Unable to write uploaded file to temp file", http.StatusInternalServerError)
		return
	}

	// Close the tempFile to ensure all data is flushed to disk
	err = tempFile.Close()
	if err != nil {
		http.Error(w, "Unable to close temp file", http.StatusInternalServerError)
		return
	}

	// Process the PDF
	// err = processPDF(tempFile.Name())
	err = processPDF(tempFile, dbName)
	if err != nil {
		http.Error(w, fmt.Sprintf("Error processing PDF: %v", err), http.StatusInternalServerError)
		return
	}

	// Send a success response
	w.WriteHeader(http.StatusOK)
	w.Write([]byte("PDF processed successfully, embeddings stored in DB"))
}

func queryPDFHandler(w http.ResponseWriter, r *http.Request) {
	client := resty.New()

	// Struct to hold the JSON structure
	type QueryRequest struct {
		QueryEmbedding []float64 `json:"query_embedding"`
	}

	// getting query text & db name from the request
	queryText := r.FormValue("query")
	dbName := r.FormValue("db")

	embedding_value, err := generateEmbedding(queryText)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("failed to generate embedding:" + err.Error()))
	}

	data := QueryRequest{
		QueryEmbedding: embedding_value,
	}

	// data := map[string]interface{}{
	// 	"document":   document,
	// 	"embeddings": []interface{}{embedding},
	// 	"metadata": []interface{}{map[string]interface{}{
	// 		"filename": fileName,
	// 		"chunk":    chunkNo,
	// 	}},
	// }

	jsonData, err := json.Marshal(data)
	if err != nil {
		fmt.Println("Error marshalling JSON:", err)
	}

	fmt.Println("Querying embedding in Chroma DB...")
	fmt.Println("Data: ", string(jsonData))

	if dbName == "chroma" {
		resp, err := client.R().
			SetHeader("Content-Type", "application/json").
			SetBody(data).
			Post("http://flask-service:5001/search_embeddings_chroma")

		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("error response from Chroma DB: " + err.Error()))
		}

		if resp.IsError() {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("error response from Chroma DB: " + resp.Status() + " error ms: " + resp.String()))
		}

		// Struct to hold the JSON structure
		/*
			{
				"matches": [
				"document": ["document1", "document2", "document3"],
				"metadata": ["{"ChunkNo": 0, "FileName": "asda"}", "{"ChunkNo": 1, "FileName": "asda"}", "{"ChunkNo": 2, "FileName": "asda"}"],
				]
			}
		*/

		// Marshal the struct to JSON
		// jsonData, err := json.Marshal(resp.String())
		// if err != nil {
		// 	http.Error(w, "Error marshaling JSON", http.StatusInternalServerError)
		// 	return
		// }

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(resp.String()))
	} else if dbName == "pgvector" {
		resp, err := client.R().
			SetHeader("Content-Type", "application/json").
			SetBody(data).
			Post("http://flask-service:5001/search_embeddings_pgvector")

		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("error response from PgVector DB: " + err.Error()))
		}

		if resp.IsError() {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("error response from PgVector DB: " + resp.Status() + " error ms: " + resp.String()))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(resp.String()))
	} else if dbName == "weviate" {
		resp, err := client.R().
			SetHeader("Content-Type", "application/json").
			SetBody(data).
			Post("http://flask-service:5001/search_embeddings_weviate")

		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("error response from PgVector DB: " + err.Error()))
		}

		if resp.IsError() {
			w.WriteHeader(http.StatusBadRequest)
			w.Write([]byte("error response from PgVector DB: " + resp.Status() + " error ms: " + resp.String()))
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(resp.String()))
	} else {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte("invalid db name"))
	}

}

// Main function to set up the server
func main() {
	r := mux.NewRouter()
	r.HandleFunc("/upload-pdf", uploadPDFHandler).Methods("POST")
	// r.HandleFunc("/store-chroma", storeEmbeddingInChromaHandler).Methods("POST")
	r.HandleFunc("/query-pdf", queryPDFHandler).Methods("POST")

	fmt.Println("Server running on port 8081")
	log.Fatal(http.ListenAndServe(":8081", r))
}

// Handler for the POST endpoint to store embeddings in Chroma DB
// func storeEmbeddingInChromaHandler(w http.ResponseWriter, r *http.Request) {
// 	// Parse the request body
// 	var data map[string]interface{}
// 	err := json.NewDecoder(r.Body).Decode(&data)
// 	if err != nil {
// 		http.Error(w, "Invalid request body", http.StatusBadRequest)
// 		return
// 	}

// 	// Store the embeddings in Chroma DB
// 	fmt.Println("Calling Python script to store embeddings in Chroma DB...")
// 	fmt.Println("Data: ", fmt.Sprintf("%v", data))
// 	cmd := exec.Command("python", "postChroma.py", fmt.Sprintf("%v", data))
// 	output, err := cmd.Output()
// 	fmt.Println("Output: ")
// 	fmt.Println(output)
// 	if err != nil {
// 		fmt.Println("Error calling Python script: ", err)
// 		return
// 	}

// 	// For now, just print the data
// 	fmt.Println("Storing embeddings in Chroma DB...")
// 	fmt.Println(data)

// 	// Send a success response
// 	w.WriteHeader(http.StatusOK)
// 	w.Write([]byte("Embeddings stored in Chroma DB"))
// }
