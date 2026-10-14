// Package config загружает настройки сервиса из переменных окружения (12-factor).
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"
)

type Config struct {
	Addr            string        // INFERENCE_ADDR
	ModelDir        string        // MODEL_DIR: каталог с model.onnx и metadata.json
	ORTLibPath      string        // ORT_LIB_PATH: путь к libonnxruntime.so / onnxruntime.dll
	MaxBatch        int           // MAX_BATCH: максимум объектов в одном запросе
	MaxBodyBytes    int64         // MAX_BODY_BYTES
	ShutdownTimeout time.Duration // SHUTDOWN_TIMEOUT
	LogLevel        string        // LOG_LEVEL: debug|info|warn|error
}

// Load читает конфигурацию через getenv (в проде — os.Getenv, в тестах — подмена).
func Load(getenv func(string) string) (Config, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	c := Config{
		Addr:       withDefault(getenv("INFERENCE_ADDR"), ":8080"),
		ModelDir:   withDefault(getenv("MODEL_DIR"), "models/iris"),
		ORTLibPath: getenv("ORT_LIB_PATH"),
		LogLevel:   withDefault(getenv("LOG_LEVEL"), "info"),
	}

	var err error
	if c.MaxBatch, err = intEnv(getenv, "MAX_BATCH", 256); err != nil {
		return Config{}, err
	}
	if c.MaxBatch <= 0 {
		return Config{}, fmt.Errorf("MAX_BATCH must be positive, got %d", c.MaxBatch)
	}
	maxBody, err := intEnv(getenv, "MAX_BODY_BYTES", 1<<20)
	if err != nil {
		return Config{}, err
	}
	c.MaxBodyBytes = int64(maxBody)

	c.ShutdownTimeout = 15 * time.Second
	if v := getenv("SHUTDOWN_TIMEOUT"); v != "" {
		if c.ShutdownTimeout, err = time.ParseDuration(v); err != nil {
			return Config{}, fmt.Errorf("SHUTDOWN_TIMEOUT: %w", err)
		}
	}
	return c, nil
}

func withDefault(v, def string) string {
	if v == "" {
		return def
	}
	return v
}

func intEnv(getenv func(string) string, key string, def int) (int, error) {
	v := getenv(key)
	if v == "" {
		return def, nil
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", key, err)
	}
	return n, nil
}
