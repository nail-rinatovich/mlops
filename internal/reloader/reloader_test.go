package reloader

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"mlops/internal/model"
	api "mlops/internal/registryapi"
	"mlops/internal/registryclient"
)

// fakeSource имитирует реестр: production-версия и содержимое файлов задаются в тесте.
type fakeSource struct {
	prod      *api.Version
	files     map[int]string // version -> содержимое model.onnx
	downloads int
	getErr    error
}

func sha(s string) string { h := sha256.Sum256([]byte(s)); return hex.EncodeToString(h[:]) }

func (f *fakeSource) setProd(v int) {
	f.prod = &api.Version{Name: "iris", Version: v, SHA256: sha(f.files[v])}
}

func (f *fakeSource) GetVersion(context.Context, string, string) (api.Version, error) {
	if f.getErr != nil {
		return api.Version{}, f.getErr
	}
	if f.prod == nil {
		return api.Version{}, registryclient.ErrNotFound
	}
	return *f.prod, nil
}

func (f *fakeSource) Download(_ context.Context, v api.Version, dest string) error {
	f.downloads++
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dest, api.FileModel), []byte(f.files[v.Version]), 0o644); err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dest, api.FileMetadata), []byte(`{}`), 0o644)
}

type stubPredictor struct{ version string }

func (s stubPredictor) Predict(context.Context, [][]float32) ([]model.Prediction, error) {
	return nil, nil
}
func (s stubPredictor) Metadata() model.Metadata { return model.Metadata{Version: s.version} }
func (s stubPredictor) Close() error             { return nil }

type recTarget struct{ set []string }

func (t *recTarget) SetPredictor(p model.Predictor) { t.set = append(t.set, p.Metadata().Version) }

func setup(t *testing.T) (*fakeSource, *recTarget, *Reloader, *error) {
	t.Helper()
	src := &fakeSource{files: map[int]string{1: "model-v1", 2: "model-v2", 3: "model-v3"}}
	tgt := &recTarget{}
	var loadErr error
	load := func(dir string) (model.Predictor, error) {
		if loadErr != nil {
			return nil, loadErr
		}
		return stubPredictor{version: filepath.Base(dir)}, nil
	}
	r := New(src, tgt, load, "iris", t.TempDir(), 0, slog.New(slog.NewTextHandler(io.Discard, nil)))
	return src, tgt, r, &loadErr
}

func TestSyncLifecycle(t *testing.T) {
	src, tgt, r, loadErr := setup(t)
	ctx := context.Background()

	if err := r.Sync(ctx); !errors.Is(err, ErrNoProduction) {
		t.Fatalf("no production: err = %v", err)
	}

	src.setProd(1)
	mustSync(t, r)
	mustSync(t, r) // повторная синхронизация без изменений ничего не делает
	if len(tgt.set) != 1 || src.downloads != 1 {
		t.Fatalf("after v1: set=%v downloads=%d", tgt.set, src.downloads)
	}

	src.setProd(2)
	mustSync(t, r)

	// Новая модель не грузится → продолжаем работать на старой, ошибка наружу.
	src.setProd(3)
	*loadErr = errors.New("broken onnx")
	if err := r.Sync(ctx); err == nil {
		t.Fatal("expected load error")
	}
	if r.Current().Version != 2 {
		t.Fatalf("current = v%d, want v2 kept after failed load", r.Current().Version)
	}
	*loadErr = nil

	// Откат на v1. В кэше хранятся только текущая и предыдущая версии, v1 уже вытеснена, поэтому скачается заново.
	src.setProd(1)
	mustSync(t, r)

	want := []string{"1", "2", "1"}
	if !slices.Equal(tgt.set, want) {
		t.Errorf("activations = %v, want %v", tgt.set, want)
	}
}

func TestSyncUsesCache(t *testing.T) {
	src, _, r, _ := setup(t)
	src.setProd(1)
	mustSync(t, r)
	src.setProd(2)
	mustSync(t, r)
	src.setProd(1) // откат на предыдущую: она ещё в кэше
	mustSync(t, r)
	if src.downloads != 2 {
		t.Errorf("downloads = %d, want 2 (rollback served from cache)", src.downloads)
	}
}

func TestSyncCorruptedCacheRedownloads(t *testing.T) {
	src, _, r, _ := setup(t)
	src.setProd(1)
	mustSync(t, r)
	// Портим файл в кэше и заставляем reloader считать, что модель другая.
	path := filepath.Join(r.cacheDir, "iris", "1", api.FileModel)
	if err := os.WriteFile(path, []byte("corrupted"), 0o644); err != nil {
		t.Fatal(err)
	}
	r.current = api.Version{}
	mustSync(t, r)
	if src.downloads != 2 {
		t.Errorf("downloads = %d, want 2 (corrupted cache must be re-downloaded)", src.downloads)
	}
}

func TestSyncRegistryDown(t *testing.T) {
	src, _, r, _ := setup(t)
	src.getErr = errors.New("connection refused")
	if err := r.Sync(context.Background()); err == nil || errors.Is(err, ErrNoProduction) {
		t.Errorf("err = %v, want connection error", err)
	}
}

func mustSync(t *testing.T, r *Reloader) {
	t.Helper()
	if err := r.Sync(context.Background()); err != nil {
		t.Fatal(err)
	}
}
