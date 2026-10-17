// Package reloader следит за production-версией модели в реестре и подменяет
// модель в inference-сервисе без остановки.
package reloader

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"mlops/internal/model"
	api "mlops/internal/registryapi"
	"mlops/internal/registryclient"
)

// Source — откуда брать модели (реализуется registryclient.Client).
type Source interface {
	GetVersion(ctx context.Context, name, ref string) (api.Version, error)
	Download(ctx context.Context, v api.Version, dest string) error
}

// Target — куда ставить загруженную модель (реализуется server.Server).
type Target interface {
	SetPredictor(model.Predictor)
}

// Loader загружает модель из каталога (в проде — model.LoadONNX).
type Loader func(dir string) (model.Predictor, error)

var ErrNoProduction = errors.New("model has no production version")

type Reloader struct {
	src      Source
	target   Target
	load     Loader
	name     string
	cacheDir string
	interval time.Duration
	log      *slog.Logger
	current  api.Version
}

func New(src Source, target Target, load Loader, name, cacheDir string, interval time.Duration, log *slog.Logger) *Reloader {
	return &Reloader{src: src, target: target, load: load, name: name, cacheDir: cacheDir, interval: interval, log: log}
}

// Current возвращает версию, которая сейчас обслуживает запросы (Version == 0, если ещё никакой).
func (r *Reloader) Current() api.Version { return r.current }

// Run синхронизируется сразу, затем каждые interval, пока не отменят ctx.
func (r *Reloader) Run(ctx context.Context) {
	t := time.NewTicker(r.interval)
	defer t.Stop()
	for {
		if err := r.Sync(ctx); err != nil && ctx.Err() == nil {
			// Ошибка не фатальна: продолжаем обслуживать запросы текущей моделью.
			r.log.Warn("model sync failed", "model", r.name, "serving_version", r.current.Version, "err", err)
		}
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// Sync сверяет production-версию в реестре с текущей и при расхождении подменяет модель.
// Сравниваем по SHA256, а не по номеру: так откат на старую версию тоже сработает.
func (r *Reloader) Sync(ctx context.Context) error {
	v, err := r.src.GetVersion(ctx, r.name, string(api.StageProduction))
	if errors.Is(err, registryclient.ErrNotFound) {
		return ErrNoProduction
	}
	if err != nil {
		return fmt.Errorf("get production version: %w", err)
	}
	if v.SHA256 == r.current.SHA256 {
		return nil
	}

	dir := filepath.Join(r.cacheDir, r.name, strconv.Itoa(v.Version))
	if !cached(dir, v.SHA256) {
		start := time.Now()
		if err := r.src.Download(ctx, v, dir); err != nil {
			return fmt.Errorf("download v%d: %w", v.Version, err)
		}
		r.log.Info("model downloaded", "model", r.name, "version", v.Version, "duration_ms", time.Since(start).Milliseconds())
	}

	p, err := r.load(dir)
	if err != nil {
		return fmt.Errorf("load v%d: %w", v.Version, err)
	}
	r.target.SetPredictor(p)
	prev := r.current
	r.current = v
	r.log.Info("model activated", "model", r.name, "version", v.Version, "previous_version", prev.Version, "sha256", v.SHA256)
	r.prune(prev.Version, v.Version)
	return nil
}

// cached проверяет, что в каталоге уже лежит нужная модель с правильным хэшем.
func cached(dir, sha string) bool {
	if _, err := os.Stat(filepath.Join(dir, api.FileMetadata)); err != nil {
		return false
	}
	f, err := os.Open(filepath.Join(dir, api.FileModel))
	if err != nil {
		return false
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false
	}
	return hex.EncodeToString(h.Sum(nil)) == sha
}

// prune удаляет из кэша все версии, кроме текущей и предыдущей (для быстрого отката).
func (r *Reloader) prune(keep ...int) {
	entries, err := os.ReadDir(filepath.Join(r.cacheDir, r.name))
	if err != nil {
		return
	}
	for _, e := range entries {
		n, err := strconv.Atoi(e.Name())
		if err != nil || !e.IsDir() {
			continue
		}
		if n != keep[0] && n != keep[1] {
			_ = os.RemoveAll(filepath.Join(r.cacheDir, r.name, e.Name()))
		}
	}
}
