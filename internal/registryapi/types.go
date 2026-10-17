// Package registryapi — типы HTTP API реестра моделей. Общие для сервера и клиента,
// поэтому не тянут за собой драйверы БД и S3.
package registryapi

import (
	"regexp"
	"time"

	"mlops/internal/model"
)

// Stage — стадия жизненного цикла версии модели.
type Stage string

const (
	StageNone       Stage = "none"
	StageStaging    Stage = "staging"
	StageProduction Stage = "production"
	StageArchived   Stage = "archived"
)

func (s Stage) Valid() bool {
	switch s {
	case StageNone, StageStaging, StageProduction, StageArchived:
		return true
	}
	return false
}

// Файлы, из которых состоит версия модели.
const (
	FileModel    = "model.onnx"
	FileMetadata = "metadata.json"
)

var nameRe = regexp.MustCompile(`^[a-z0-9][a-z0-9_-]{0,62}$`)

// ValidName проверяет имя модели: строчные латинские буквы, цифры, '-', '_'.
func ValidName(name string) bool { return nameRe.MatchString(name) }

// Version — одна зарегистрированная версия модели.
type Version struct {
	Name           string         `json:"name"`
	Version        int            `json:"version"`
	Stage          Stage          `json:"stage"`
	SHA256         string         `json:"sha256"`
	SizeBytes      int64          `json:"size_bytes"`
	Metadata       model.Metadata `json:"metadata"`
	CreatedBy      string         `json:"created_by"`
	CreatedAt      time.Time      `json:"created_at"`
	StageUpdatedAt time.Time      `json:"stage_updated_at"`
}

// ModelSummary — строка в списке моделей.
type ModelSummary struct {
	Name              string    `json:"name"`
	LatestVersion     int       `json:"latest_version"`
	ProductionVersion *int      `json:"production_version"`
	CreatedAt         time.Time `json:"created_at"`
}

type SetStageRequest struct {
	Stage Stage `json:"stage"`
}

type ErrorResponse struct {
	Error string `json:"error"`
}
