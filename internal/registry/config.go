package registry

import (
	"errors"
	"fmt"
	"os"
	"strconv"
)

type Config struct {
	Addr           string // REGISTRY_ADDR
	DatabaseURL    string // DATABASE_URL
	S3             S3Config
	Token          string // REGISTRY_TOKEN: токен для записи; пусто — запись без авторизации
	MaxUploadBytes int64  // MAX_UPLOAD_BYTES
	LogLevel       string // LOG_LEVEL
}

func LoadConfig(getenv func(string) string) (Config, error) {
	if getenv == nil {
		getenv = os.Getenv
	}
	def := func(k, d string) string {
		if v := getenv(k); v != "" {
			return v
		}
		return d
	}
	c := Config{
		Addr:        def("REGISTRY_ADDR", ":8090"),
		DatabaseURL: getenv("DATABASE_URL"),
		S3: S3Config{
			Endpoint:  def("S3_ENDPOINT", "localhost:9000"),
			AccessKey: getenv("S3_ACCESS_KEY"),
			SecretKey: getenv("S3_SECRET_KEY"),
			Bucket:    def("S3_BUCKET", "models"),
		},
		Token:    getenv("REGISTRY_TOKEN"),
		LogLevel: def("LOG_LEVEL", "info"),
	}
	var err error
	if c.S3.UseSSL, err = strconv.ParseBool(def("S3_USE_SSL", "false")); err != nil {
		return Config{}, fmt.Errorf("S3_USE_SSL: %w", err)
	}
	if c.MaxUploadBytes, err = strconv.ParseInt(def("MAX_UPLOAD_BYTES", "268435456"), 10, 64); err != nil {
		return Config{}, fmt.Errorf("MAX_UPLOAD_BYTES: %w", err)
	}
	if c.DatabaseURL == "" {
		return Config{}, errors.New("DATABASE_URL is required")
	}
	if c.S3.AccessKey == "" || c.S3.SecretKey == "" {
		return Config{}, errors.New("S3_ACCESS_KEY and S3_SECRET_KEY are required")
	}
	return c, nil
}
