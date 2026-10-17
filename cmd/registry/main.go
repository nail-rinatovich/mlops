// Команда registry — HTTP-сервис реестра моделей.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"mlops/internal/registry"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := registry.LoadConfig(os.Getenv)
	if err != nil {
		return err
	}
	var level slog.Level
	if err := level.UnmarshalText([]byte(cfg.LogLevel)); err != nil {
		return err
	}
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(log)
	if cfg.Token == "" {
		log.Warn("REGISTRY_TOKEN is empty: write endpoints are not protected")
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	pool, err := pgxpool.New(ctx, cfg.DatabaseURL)
	if err != nil {
		return err
	}
	defer pool.Close()
	blob, err := registry.NewBlob(cfg.S3)
	if err != nil {
		return err
	}

	// В docker compose БД и S3 стартуют параллельно с нами: ждём их, а не падаем.
	if err := retry(ctx, log, "postgres", func(ctx context.Context) error { return pool.Ping(ctx) }); err != nil {
		return err
	}
	applied, err := registry.Migrate(ctx, pool)
	if err != nil {
		return err
	}
	log.Info("migrations applied", "new", applied)
	if err := retry(ctx, log, "s3", blob.EnsureBucket); err != nil {
		return err
	}

	svc := registry.NewService(registry.NewStore(pool), blob, log)
	api := registry.NewAPI(svc, log, cfg.Token, cfg.MaxUploadBytes, pool.Ping)
	srv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           api.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		IdleTimeout:       60 * time.Second,
		// Read/WriteTimeout не ставим: загрузка и скачивание больших моделей могут идти долго.
	}

	errCh := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", cfg.Addr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()
	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}
	log.Info("shutting down")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}

// retry повторяет fn с растущей паузой, пока не получится или не истечёт минута.
func retry(ctx context.Context, log *slog.Logger, what string, fn func(context.Context) error) error {
	ctx, cancel := context.WithTimeout(ctx, time.Minute)
	defer cancel()
	delay := 500 * time.Millisecond
	for {
		err := fn(ctx)
		if err == nil {
			return nil
		}
		log.Info("waiting for dependency", "dependency", what, "err", err)
		select {
		case <-ctx.Done():
			return errors.Join(ctx.Err(), err)
		case <-time.After(delay):
		}
		delay = min(delay*2, 5*time.Second)
	}
}
