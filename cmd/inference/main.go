// Команда inference — HTTP-сервис предсказаний поверх ONNX-модели.
//
// Два режима:
//   - локальный (REGISTRY_URL не задан): модель читается из MODEL_DIR при старте;
//   - реестр (REGISTRY_URL задан): модель скачивается из реестра и подменяется на лету,
//     когда в реестре меняется production-версия.
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
	"mlops/internal/registryclient"
	"mlops/internal/reloader"
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

	srv := server.New(server.Options{MaxBatch: cfg.MaxBatch, MaxBodyBytes: cfg.MaxBodyBytes}, log)
	defer srv.Close() // выгружаем модель до DestroyRuntime (defer выполняются в обратном порядке)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	loadONNX := func(dir string) (model.Predictor, error) {
		start := time.Now()
		p, err := model.LoadONNX(dir)
		if err != nil {
			return nil, err
		}
		md := p.Metadata()
		log.Info("model loaded", "name", md.Name, "version", md.Version, "sha256", md.SHA256,
			"load_ms", time.Since(start).Milliseconds())
		return p, nil
	}

	// rlDone закрывается, когда reloader завершился: до этого нельзя выгружать модель,
	// иначе он может успеть поставить новую уже после srv.Close().
	rlDone := make(chan struct{})
	defer func() { stop(); <-rlDone }() // выполнится раньше srv.Close: defer идут в обратном порядке
	if cfg.RegistryURL == "" {
		close(rlDone)
		p, err := loadONNX(cfg.ModelDir)
		if err != nil {
			return err
		}
		srv.SetPredictor(p)
	} else {
		// Сервер стартует сразу, а /readyz отвечает 503, пока не загрузится первая модель.
		client := registryclient.New(cfg.RegistryURL, "", &http.Client{Timeout: 5 * time.Minute})
		rl := reloader.New(client, srv, loadONNX, cfg.ModelName, cfg.CacheDir, cfg.PollInterval, log)
		log.Info("registry mode", "registry", cfg.RegistryURL, "model", cfg.ModelName, "poll", cfg.PollInterval.String())
		go func() {
			defer close(rlDone)
			rl.Run(ctx)
		}()
	}

	httpSrv := &http.Server{
		Addr:              cfg.Addr,
		Handler:           srv.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

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
