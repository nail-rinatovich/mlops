// Команда inference — HTTP-сервис предсказаний поверх ONNX-модели.
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

	"mlops/internal/config"
	"mlops/internal/model"
	"mlops/internal/server"
)

func main() {
	if err := run(); err != nil {
		slog.Error("fatal", "err", err)
		os.Exit(1)
	}
}

func run() error {
	cfg, err := config.Load(os.Getenv)
	if err != nil {
		return err
	}

	var level slog.Level
	if err := level.UnmarshalText([]byte(cfg.LogLevel)); err != nil {
		return err
	}
	log := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: level}))
	slog.SetDefault(log)

	if err := model.InitRuntime(cfg.ORTLibPath); err != nil {
		return err
	}
	defer model.DestroyRuntime()

	start := time.Now()
	pred, err := model.LoadONNX(cfg.ModelDir)
	if err != nil {
		return err
	}
	md := pred.Metadata()
	log.Info("model loaded", "name", md.Name, "version", md.Version, "sha256", md.SHA256,
		"load_ms", time.Since(start).Milliseconds())

	srv := server.New(server.Options{MaxBatch: cfg.MaxBatch, MaxBodyBytes: cfg.MaxBodyBytes}, log)
	srv.SetPredictor(pred)
	defer pred.Close()

	httpSrv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	errCh := make(chan error, 1)
	go func() {
		log.Info("listening", "addr", cfg.Addr)
		if err := httpSrv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()

	select {
	case err := <-errCh:
		return err
	case <-ctx.Done():
	}

	log.Info("shutting down", "timeout", cfg.ShutdownTimeout.String())
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	// Shutdown дожидается завершения активных запросов, новые не принимает.
	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		return err
	}
	log.Info("stopped")
	return nil
}
