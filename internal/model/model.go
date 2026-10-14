// Package model описывает контракт предиктора и метаданные модели.
package model

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Prediction — результат для одного объекта.
type Prediction struct {
	Label         int64     `json:"label"`
	Class         string    `json:"class"`
	Probabilities []float32 `json:"probabilities"`
}

// Predictor выполняет инференс над батчем. Реализации обязаны быть безопасны
// для конкурентного использования.
type Predictor interface {
	Predict(ctx context.Context, batch [][]float32) ([]Prediction, error)
	Metadata() Metadata
	Close() error
}

type Feature struct {
	Name string  `json:"name"`
	Min  float32 `json:"min"`
	Max  float32 `json:"max"`
}

type Metadata struct {
	Name     string             `json:"name"`
	Version  string             `json:"version"`
	SHA256   string             `json:"sha256"`
	Features []Feature          `json:"features"`
	Classes  []string           `json:"classes"`
	Metrics  map[string]float64 `json:"metrics"`
}

// LoadMetadata читает metadata.json из каталога модели.
func LoadMetadata(dir string) (Metadata, error) {
	var md Metadata
	b, err := os.ReadFile(filepath.Join(dir, "metadata.json"))
	if err != nil {
		return md, fmt.Errorf("read metadata: %w", err)
	}
	if err := json.Unmarshal(b, &md); err != nil {
		return md, fmt.Errorf("parse metadata: %w", err)
	}
	if len(md.Features) == 0 || len(md.Classes) == 0 {
		return md, fmt.Errorf("metadata: features and classes must be non-empty")
	}
	return md, nil
}

// ClassName возвращает имя класса по метке или "unknown".
func (m Metadata) ClassName(label int64) string {
	if label < 0 || int(label) >= len(m.Classes) {
		return "unknown"
	}
	return m.Classes[label]
}
