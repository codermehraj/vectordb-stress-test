package main

import (
	"context"
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"math"
)

// Embedder fetches embeddings.
type Embedder[T any] interface {
	// Embeddings fetches embeddings and returns them.
	Embed(context.Context, T) ([]*Embedding, error)
}

// Embedding is vector embedding.
type Embedding struct {
	Vector []float64 `json:"vector"`
}

// ToFloat32 returns Embedding verctor as a slice of float32.
func (e Embedding) ToFloat32() []float32 {
	floats := make([]float32, len(e.Vector))
	for i, f := range e.Vector {
		floats[i] = float32(f)
	}
	return floats
}

// Base64 is base64 encoded embedding string.
type Base64 string

// Decode decodes base64 encoded string into a slice of floats.
func (s Base64) Decode() (*Embedding, error) {
	decoded, err := base64.StdEncoding.DecodeString(string(s))
	if err != nil {
		return nil, err
	}

	if len(decoded)%8 != 0 {
		return nil, fmt.Errorf("invalid base64 encoded string length")
	}

	floats := make([]float64, len(decoded)/8)

	for i := 0; i < len(floats); i++ {
		bits := binary.LittleEndian.Uint64(decoded[i*8 : (i+1)*8])
		floats[i] = math.Float64frombits(bits)
	}

	return &Embedding{
		Vector: floats,
	}, nil
}

// Helper function to encode a slice of float64 values into a Base64 string
func encodeToBase64(floats []float64) string {
	bytes := make([]byte, len(floats)*8)
	for i, f := range floats {
		bits := math.Float64bits(f)
		binary.LittleEndian.PutUint64(bytes[i*8:], bits)
	}
	return base64.StdEncoding.EncodeToString(bytes)
}

// func main() {
// 	// Example float64 slice to encode
// 	floats := []float64{3.14159, 2.71828, 1.61803}

// 	// Encode the float64 slice into a Base64 string
// 	encoded := encodeToBase64(floats)
// 	fmt.Println("Encoded Base64 string:", encoded)

// 	// Create a Base64 type from the encoded string
// 	base64Str := Base64(encoded)

// 	// Call the Decode method to decode the Base64 string into an Embedding
// 	embedding, err := base64Str.Decode()
// 	if err != nil {
// 		fmt.Println("Error decoding base64 string:", err)
// 		return
// 	}

// 	// Print the decoded embedding vector
// 	fmt.Println("Decoded embedding vector:", embedding.Vector)
// }
