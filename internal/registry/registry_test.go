package registry_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/testcontainers/testcontainers-go"
	"github.com/testcontainers/testcontainers-go/modules/postgres"
	"github.com/testcontainers/testcontainers-go/wait"

	"mlops/internal/model"
	"mlops/internal/registry"
	api "mlops/internal/registryapi"
	"mlops/internal/registryclient"
)

// Интеграционные тесты на настоящих PostgreSQL и RustFS (S3) в контейнерах.
// Без Docker тесты пропускаются.

const token = "test-token"

var (
	envOnce sync.Once
	envErr  error
	envURL  string
	cleanup []func()
)

func TestMain(m *testing.M) {
	code := m.Run()
	for _, f := range cleanup {
		f()
	}
	os.Exit(code)
}

func startEnv() (string, error) {
	ctx := context.Background()

	pg, err := postgres.Run(ctx, "postgres:17-alpine",
		postgres.WithDatabase("registry"), postgres.WithUsername("test"), postgres.WithPassword("test"),
		postgres.BasicWaitStrategies())
	if err != nil {
		return "", fmt.Errorf("start postgres: %w", err)
	}
	cleanup = append(cleanup, func() { _ = pg.Terminate(context.Background()) })
	dsn, err := pg.ConnectionString(ctx, "sslmode=disable")
	if err != nil {
		return "", err
	}

	s3, err := testcontainers.Run(ctx, "rustfs/rustfs:1.0.2",
		testcontainers.WithExposedPorts("9000/tcp"),
		testcontainers.WithEnv(map[string]string{"RUSTFS_ACCESS_KEY": "testkey", "RUSTFS_SECRET_KEY": "testsecret"}),
		testcontainers.WithWaitStrategy(wait.ForListeningPort("9000/tcp").WithStartupTimeout(time.Minute)))
	if err != nil {
		return "", fmt.Errorf("start rustfs: %w", err)
	}
	cleanup = append(cleanup, func() { _ = s3.Terminate(context.Background()) })
	endpoint, err := s3.PortEndpoint(ctx, "9000/tcp", "")
	if err != nil {
		return "", err
	}

	pool, err := pgxpool.New(ctx, dsn)
	if err != nil {
		return "", err
	}
	cleanup = append(cleanup, pool.Close)
	if _, err := registry.Migrate(ctx, pool); err != nil {
		return "", err
	}
	blob, err := registry.NewBlob(registry.S3Config{Endpoint: endpoint, AccessKey: "testkey", SecretKey: "testsecret", Bucket: "models"})
	if err != nil {
		return "", err
	}
	// Порт открывается чуть раньше, чем S3 готов отвечать: повторяем.
	deadline := time.Now().Add(30 * time.Second)
	for err = blob.EnsureBucket(ctx); err != nil && time.Now().Before(deadline); err = blob.EnsureBucket(ctx) {
		time.Sleep(500 * time.Millisecond)
	}
	if err != nil {
		return "", err
	}

	log := slog.New(slog.NewTextHandler(io.Discard, nil))
	svc := registry.NewService(registry.NewStore(pool), blob, log)
	srv := httptest.NewServer(registry.NewAPI(svc, log, token, 64<<20, pool.Ping).Handler())
	cleanup = append(cleanup, srv.Close)
	return srv.URL, nil
}

// newClient поднимает окружение (один раз на прогон) и возвращает клиент с токеном.
func newClient(t *testing.T) *registryclient.Client {
	t.Helper()
	testcontainers.SkipIfProviderIsNotHealthy(t)
	envOnce.Do(func() { envURL, envErr = startEnv() })
	if envErr != nil {
		t.Fatal(envErr)
	}
	return registryclient.New(envURL, token, nil)
}

// modelDir — каталог с обученной моделью из проекта 1.
func modelDir() string { return filepath.Join("..", "..", "models", "iris") }

// uniqueName даёт каждому тесту свою модель, чтобы тесты не мешали друг другу.
func uniqueName(t *testing.T) string {
	return strings.ToLower(strings.NewReplacer("/", "-", "_", "-").Replace(t.Name()))
}

func TestPushAndList(t *testing.T) {
	c := newClient(t)
	ctx := context.Background()
	name := uniqueName(t)

	v1, err := c.Push(ctx, name, modelDir(), "alice")
	if err != nil {
		t.Fatal(err)
	}
	md, _ := model.LoadMetadata(modelDir())
	if v1.Version != 1 || v1.Stage != api.StageNone || v1.SHA256 != md.SHA256 || v1.CreatedBy != "alice" {
		t.Fatalf("unexpected v1: %+v", v1)
	}
	if v1.Metadata.Version != "1" || v1.Metadata.Name != name {
		t.Errorf("metadata must carry registry-assigned name/version, got %q v%q", v1.Metadata.Name, v1.Metadata.Version)
	}

	v2, err := c.Push(ctx, name, modelDir(), "bob")
	if err != nil {
		t.Fatal(err)
	}
	if v2.Version != 2 {
		t.Fatalf("v2.Version = %d", v2.Version)
	}

	vs, err := c.ListVersions(ctx, name)
	if err != nil {
		t.Fatal(err)
	}
	if len(vs) != 2 || vs[0].Version != 2 {
		t.Errorf("versions = %+v, want [v2 v1]", vs)
	}
}

func TestPromoteArchivesPreviousProduction(t *testing.T) {
	c := newClient(t)
	ctx := context.Background()
	name := uniqueName(t)
	for range 2 {
		if _, err := c.Push(ctx, name, modelDir(), "ci"); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := c.GetVersion(ctx, name, "production"); !errors.Is(err, registryclient.ErrNotFound) {
		t.Fatalf("before promote: err = %v, want not found", err)
	}
	if _, err := c.SetStage(ctx, name, 1, api.StageProduction); err != nil {
		t.Fatal(err)
	}
	if _, err := c.SetStage(ctx, name, 2, api.StageProduction); err != nil {
		t.Fatal(err)
	}

	prod, err := c.GetVersion(ctx, name, "production")
	if err != nil || prod.Version != 2 {
		t.Fatalf("production = v%d, err %v; want v2", prod.Version, err)
	}
	v1, _ := c.GetVersion(ctx, name, "1")
	if v1.Stage != api.StageArchived {
		t.Errorf("v1 stage = %s, want archived", v1.Stage)
	}

	// Откат: v1 снова production, v2 уходит в archived.
	if _, err := c.SetStage(ctx, name, 1, api.StageProduction); err != nil {
		t.Fatal(err)
	}
	if prod, _ := c.GetVersion(ctx, name, "production"); prod.Version != 1 {
		t.Errorf("after rollback production = v%d, want v1", prod.Version)
	}

	ms, err := c.ListModels(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, m := range ms {
		if m.Name == name && (m.LatestVersion != 2 || m.ProductionVersion == nil || *m.ProductionVersion != 1) {
			t.Errorf("summary = %+v", m)
		}
	}
}

func TestDownloadVerifiesChecksum(t *testing.T) {
	c := newClient(t)
	ctx := context.Background()
	name := uniqueName(t)
	v, err := c.Push(ctx, name, modelDir(), "ci")
	if err != nil {
		t.Fatal(err)
	}

	dest := filepath.Join(t.TempDir(), "pulled")
	if err := c.Download(ctx, v, dest); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(filepath.Join(dest, api.FileModel))
	if err != nil {
		t.Fatal(err)
	}
	want, _ := os.ReadFile(filepath.Join(modelDir(), api.FileModel))
	if string(got) != string(want) {
		t.Error("downloaded model differs from uploaded")
	}
	md, err := model.LoadMetadata(dest)
	if err != nil || md.Version != "1" {
		t.Errorf("downloaded metadata version = %q, err %v", md.Version, err)
	}

	// Клиент обязан отвергнуть файл, если хэш не совпал с заявленным реестром.
	v.SHA256 = strings.Repeat("0", 64)
	if err := c.Download(ctx, v, filepath.Join(t.TempDir(), "bad")); err == nil || !strings.Contains(err.Error(), "checksum") {
		t.Errorf("expected checksum error, got %v", err)
	}
}

func TestUploadRejectsBadInput(t *testing.T) {
	c := newClient(t)
	ctx := context.Background()
	name := uniqueName(t)

	// metadata.json заявляет один хэш, а файл модели другой.
	bad := t.TempDir()
	onnx, _ := os.ReadFile(filepath.Join(modelDir(), api.FileModel))
	_ = os.WriteFile(filepath.Join(bad, api.FileModel), onnx, 0o644)
	md, _ := model.LoadMetadata(modelDir())
	md.SHA256 = strings.Repeat("a", 64)
	b, _ := json.Marshal(md)
	_ = os.WriteFile(filepath.Join(bad, api.FileMetadata), b, 0o644)

	_, err := c.Push(ctx, name, bad, "ci")
	if err == nil || !strings.Contains(err.Error(), "422") {
		t.Fatalf("checksum mismatch: err = %v, want 422", err)
	}
	if _, err := c.ListVersions(ctx, name); !errors.Is(err, registryclient.ErrNotFound) {
		t.Errorf("rejected upload must not create a version, err = %v", err)
	}

	if _, err := c.Push(ctx, "Bad Name!", modelDir(), "ci"); err == nil || !strings.Contains(err.Error(), "400") {
		t.Errorf("invalid name: err = %v, want 400", err)
	}
	if _, err := c.SetStage(ctx, name, 1, "deployed"); err == nil {
		t.Error("unknown stage must be rejected")
	}
}

func TestWriteRequiresToken(t *testing.T) {
	newClient(t)
	anon := registryclient.New(envURL, "", nil)
	_, err := anon.Push(context.Background(), uniqueName(t), modelDir(), "eve")
	if err == nil || !strings.Contains(err.Error(), "401") {
		t.Fatalf("push without token: err = %v, want 401", err)
	}
	if _, err := anon.ListModels(context.Background()); err != nil {
		t.Errorf("reads must stay public, got %v", err)
	}
}

// TestConcurrentPushes: параллельные загрузки одной модели получают разные номера без пропусков.
func TestConcurrentPushes(t *testing.T) {
	c := newClient(t)
	ctx := context.Background()
	name := uniqueName(t)

	const n = 8
	var wg sync.WaitGroup
	errs := make(chan error, n)
	for range n {
		wg.Go(func() {
			if _, err := c.Push(ctx, name, modelDir(), "ci"); err != nil {
				errs <- err
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Fatal(err)
	}

	vs, err := c.ListVersions(ctx, name)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[int]bool{}
	for _, v := range vs {
		seen[v.Version] = true
	}
	for i := 1; i <= n; i++ {
		if !seen[i] {
			t.Errorf("version %d missing; got %d versions", i, len(vs))
		}
	}
}
