package config

import (
	"testing"
	"time"
)

func envMap(m map[string]string) func(string) string {
	return func(k string) string { return m[k] }
}

func TestLoad(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		want    Config
		wantErr bool
	}{
		{
			name: "defaults",
			env:  map[string]string{},
			want: Config{Addr: ":8080", ModelDir: "models/iris", MaxBatch: 256, MaxBodyBytes: 1 << 20,
				ShutdownTimeout: 15 * time.Second, LogLevel: "info"},
		},
		{
			name: "overrides",
			env: map[string]string{"INFERENCE_ADDR": ":9000", "MODEL_DIR": "/m", "ORT_LIB_PATH": "/lib/ort.so",
				"MAX_BATCH": "8", "MAX_BODY_BYTES": "1024", "SHUTDOWN_TIMEOUT": "3s", "LOG_LEVEL": "debug"},
			want: Config{Addr: ":9000", ModelDir: "/m", ORTLibPath: "/lib/ort.so", MaxBatch: 8, MaxBodyBytes: 1024,
				ShutdownTimeout: 3 * time.Second, LogLevel: "debug"},
		},
		{name: "bad batch", env: map[string]string{"MAX_BATCH": "abc"}, wantErr: true},
		{name: "zero batch", env: map[string]string{"MAX_BATCH": "0"}, wantErr: true},
		{name: "bad body", env: map[string]string{"MAX_BODY_BYTES": "x"}, wantErr: true},
		{name: "bad timeout", env: map[string]string{"SHUTDOWN_TIMEOUT": "soon"}, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Load(envMap(tt.env))
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}
