package model

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadMetadata(t *testing.T) {
	tests := []struct {
		name    string
		content string // "" — файла нет
		wantErr bool
	}{
		{name: "ok", content: `{"name":"iris","version":"1","features":[{"name":"a","min":0,"max":1}],"classes":["x"]}`},
		{name: "missing file", wantErr: true},
		{name: "bad json", content: `{`, wantErr: true},
		{name: "no features", content: `{"classes":["x"]}`, wantErr: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			if tt.content != "" {
				if err := os.WriteFile(filepath.Join(dir, "metadata.json"), []byte(tt.content), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			md, err := LoadMetadata(dir)
			if (err != nil) != tt.wantErr {
				t.Fatalf("err = %v, wantErr %v", err, tt.wantErr)
			}
			if !tt.wantErr && md.Name != "iris" {
				t.Errorf("name = %q", md.Name)
			}
		})
	}
}

func TestClassName(t *testing.T) {
	md := Metadata{Classes: []string{"a", "b"}}
	for label, want := range map[int64]string{0: "a", 1: "b", 2: "unknown", -1: "unknown"} {
		if got := md.ClassName(label); got != want {
			t.Errorf("ClassName(%d) = %q, want %q", label, got, want)
		}
	}
}
