//go:build cgo

package model

import (
	"context"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

// Тесты паритета: Go + ONNX Runtime должны давать те же ответы, что sklearn при обучении.
// Требуют ORT_LIB_PATH и обученную модель (MODEL_DIR, по умолчанию ../../models/iris).

type samples struct {
	Inputs        [][]float32 `json:"inputs"`
	Labels        []int64     `json:"labels"`
	Probabilities [][]float32 `json:"probabilities"`
}

func modelDir() string {
	if d := os.Getenv("MODEL_DIR"); d != "" {
		return d
	}
	return filepath.Join("..", "..", "models", "iris")
}

func TestMain(m *testing.M) {
	lib := os.Getenv("ORT_LIB_PATH")
	if lib == "" {
		os.Stderr.WriteString("ORT_LIB_PATH not set, skipping ONNX tests\n")
		os.Exit(0)
	}
	if err := InitRuntime(lib); err != nil {
		panic(err)
	}
	code := m.Run()
	_ = DestroyRuntime()
	os.Exit(code)
}

func loadSamples(t testing.TB) samples {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(modelDir(), "samples.json"))
	if err != nil {
		t.Fatal(err)
	}
	var s samples
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatal(err)
	}
	return s
}

func TestONNXParity(t *testing.T) {
	p, err := LoadONNX(modelDir())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	s := loadSamples(t)

	got, err := p.Predict(context.Background(), s.Inputs)
	if err != nil {
		t.Fatal(err)
	}
	for i, pr := range got {
		if pr.Label != s.Labels[i] {
			t.Errorf("row %d: label %d, want %d", i, pr.Label, s.Labels[i])
		}
		for j, v := range pr.Probabilities {
			if math.Abs(float64(v-s.Probabilities[i][j])) > 1e-4 {
				t.Errorf("row %d class %d: prob %f, want %f", i, j, v, s.Probabilities[i][j])
			}
		}
		if pr.Class != p.Metadata().ClassName(pr.Label) {
			t.Errorf("row %d: class %q mismatch", i, pr.Class)
		}
	}
}

func TestONNXConcurrent(t *testing.T) {
	p, err := LoadONNX(modelDir())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()
	s := loadSamples(t)

	var wg sync.WaitGroup
	for range 32 {
		wg.Go(func() {
			for range 50 {
				got, err := p.Predict(context.Background(), s.Inputs[:3])
				if err != nil {
					t.Error(err)
					return
				}
				if got[0].Label != s.Labels[0] {
					t.Errorf("label %d, want %d", got[0].Label, s.Labels[0])
					return
				}
			}
		})
	}
	wg.Wait()
}

func TestONNXErrors(t *testing.T) {
	p, err := LoadONNX(modelDir())
	if err != nil {
		t.Fatal(err)
	}
	defer p.Close()

	if _, err := p.Predict(context.Background(), [][]float32{{1, 2}}); err == nil {
		t.Error("expected error for wrong feature count")
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := p.Predict(ctx, [][]float32{{1, 2, 3, 4}}); err == nil {
		t.Error("expected error for cancelled context")
	}
	if _, err := LoadONNX(t.TempDir()); err == nil {
		t.Error("expected error for empty model dir")
	}
}

func BenchmarkONNXPredict(b *testing.B) {
	p, err := LoadONNX(modelDir())
	if err != nil {
		b.Fatal(err)
	}
	defer p.Close()
	in := loadSamples(b).Inputs[:1]
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		if _, err := p.Predict(ctx, in); err != nil {
			b.Fatal(err)
		}
	}
}
