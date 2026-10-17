// Package registry — реестр моделей: версии в PostgreSQL, файлы в S3.
package registry

import (
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"path"
	"strconv"

	"mlops/internal/model"
	api "mlops/internal/registryapi"
)

var ErrChecksum = errors.New("checksum mismatch")

// Service — бизнес-логика реестра поверх Store (метаданные) и Blob (файлы).
type Service struct {
	store *Store
	blob  *Blob
	log   *slog.Logger
}

func NewService(store *Store, blob *Blob, log *slog.Logger) *Service {
	return &Service{store: store, blob: blob, log: log}
}

// ObjectKey — путь файла версии в бакете: <name>/<version>/<file>.
func ObjectKey(name string, version int, file string) string {
	return path.Join(name, strconv.Itoa(version), file)
}

// Upload регистрирует новую версию модели.
//
// Порядок важен для целостности:
//  1. файл модели заливается во временный ключ, по пути считается SHA256;
//  2. хэш сверяется с тем, что записал в metadata.json скрипт обучения;
//  3. в транзакции выделяется номер версии, файлы копируются на постоянное место,
//     и только после этого в БД появляется запись о версии.
//
// Если любой шаг падает, версия в реестре не появляется.
func (s *Service) Upload(ctx context.Context, name, createdBy string, modelFile io.Reader, size int64, metaJSON []byte) (api.Version, error) {
	if !api.ValidName(name) {
		return api.Version{}, fmt.Errorf("%w: invalid model name %q", ErrInvalid, name)
	}
	if size <= 0 {
		return api.Version{}, fmt.Errorf("%w: empty model file", ErrInvalid)
	}
	meta, err := model.ParseMetadata(metaJSON)
	if err != nil {
		return api.Version{}, fmt.Errorf("%w: %v", ErrInvalid, err)
	}

	tmpKey := "uploads/" + randomHex(16)
	h := sha256.New()
	if err := s.blob.Put(ctx, tmpKey, io.TeeReader(modelFile, h), size, "application/octet-stream"); err != nil {
		return api.Version{}, err
	}
	defer func() {
		// Временный файл удаляем в любом случае, даже если клиент уже отключился.
		if err := s.blob.Delete(context.WithoutCancel(ctx), tmpKey); err != nil {
			s.log.Warn("delete temp upload", "key", tmpKey, "err", err)
		}
	}()

	sum := hex.EncodeToString(h.Sum(nil))
	if meta.SHA256 != "" && meta.SHA256 != sum {
		return api.Version{}, fmt.Errorf("%w: metadata.json says %s, uploaded file is %s", ErrChecksum, meta.SHA256, sum)
	}

	v := api.Version{Name: name, SHA256: sum, SizeBytes: size, Metadata: meta, CreatedBy: createdBy}
	return s.store.CreateVersion(ctx, v, func(v api.Version) error {
		if err := s.blob.Copy(ctx, tmpKey, ObjectKey(name, v.Version, api.FileModel)); err != nil {
			return err
		}
		// metadata.json сохраняем уже с присвоенными реестром версией и хэшем.
		b, err := json.MarshalIndent(v.Metadata, "", "  ")
		if err != nil {
			return err
		}
		return s.blob.Put(ctx, ObjectKey(name, v.Version, api.FileMetadata), bytes.NewReader(b), int64(len(b)), "application/json")
	})
}

// Resolve находит версию по ссылке: номеру ("3") или стадии ("production", "staging").
func (s *Service) Resolve(ctx context.Context, name, ref string) (api.Version, error) {
	if n, err := strconv.Atoi(ref); err == nil {
		return s.store.GetVersion(ctx, name, n)
	}
	st := api.Stage(ref)
	if !st.Valid() || st == api.StageNone {
		return api.Version{}, fmt.Errorf("%w: version must be a number or a stage, got %q", ErrInvalid, ref)
	}
	return s.store.GetByStage(ctx, name, st)
}

func (s *Service) ListModels(ctx context.Context) ([]api.ModelSummary, error) {
	return s.store.ListModels(ctx)
}

func (s *Service) ListVersions(ctx context.Context, name string) ([]api.Version, error) {
	return s.store.ListVersions(ctx, name)
}

func (s *Service) SetStage(ctx context.Context, name string, version int, stage api.Stage) (api.Version, error) {
	v, err := s.store.SetStage(ctx, name, version, stage)
	if err == nil {
		s.log.InfoContext(ctx, "stage changed", "model", name, "version", version, "stage", stage)
	}
	return v, err
}

// OpenFile открывает файл версии на чтение. Версия должна существовать в БД.
func (s *Service) OpenFile(ctx context.Context, name string, version int, file string) (io.ReadCloser, int64, api.Version, error) {
	if file != api.FileModel && file != api.FileMetadata {
		return nil, 0, api.Version{}, fmt.Errorf("%w: unknown file %q", ErrNotFound, file)
	}
	v, err := s.store.GetVersion(ctx, name, version)
	if err != nil {
		return nil, 0, v, err
	}
	rc, size, err := s.blob.Get(ctx, ObjectKey(name, version, file))
	return rc, size, v, err
}

func randomHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}
