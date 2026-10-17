package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"mlops/internal/model"
)

type fakePredictor struct {
	md     model.Metadata
	err    error
	panics bool
}

func (f *fakePredictor) Predict(_ context.Context, batch [][]float32) ([]model.Prediction, error) {
	if f.panics {
		panic("boom")
	}
	if f.err != nil {
		return nil, f.err
	}
	out := make([]model.Prediction, len(batch))
	for i := range batch {
		out[i] = model.Prediction{Label: 1, Class: f.md.ClassName(1), Probabilities: []float32{0.1, 0.8, 0.1}}
	}
	return out, nil
}
func (f *fakePredictor) Metadata() model.Metadata { return f.md }
func (f *fakePredictor) Close() error             { return nil }

var testMeta = model.Metadata{
	Name: "iris", Version: "1",
	Features: []model.Feature{
		{Name: "sepal_length", Min: 0, Max: 16}, {Name: "sepal_width", Min: 0, Max: 9},
		{Name: "petal_length", Min: 0, Max: 14}, {Name: "petal_width", Min: 0, Max: 5},
	},
	Classes: []string{"setosa", "versicolor", "virginica"},
}

func newTestServer(p model.Predictor) *Server {
	s := New(Options{MaxBatch: 3, MaxBodyBytes: 512}, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if p != nil {
		s.SetPredictor(p)
	}
	return s
}

func do(t testing.TB, h http.Handler, method, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	return rec
}

func TestPredict(t *testing.T) {
	tests := []struct {
		name       string
		pred       model.Predictor
		body       string
		wantStatus int
		wantInBody string
	}{
		{"ok single", &fakePredictor{md: testMeta}, `{"instances":[[5.1,3.5,1.4,0.2]]}`, 200, `"class":"versicolor"`},
		{"ok batch", &fakePredictor{md: testMeta}, `{"instances":[[5,3,1,0.2],[6,3,4,1.3],[7,3,6,2]]}`, 200, `"version":"1"`},
		{"not loaded", nil, `{"instances":[[5,3,1,0.2]]}`, 503, "not loaded"},
		{"bad json", &fakePredictor{md: testMeta}, `{"instances":`, 400, "invalid JSON"},
		{"unknown field", &fakePredictor{md: testMeta}, `{"inputs":[[1,2,3,4]]}`, 400, "unknown field"},
		{"wrong type", &fakePredictor{md: testMeta}, `{"instances":[["a",2,3,4]]}`, 400, "invalid JSON"},
		{"empty", &fakePredictor{md: testMeta}, `{"instances":[]}`, 400, "non-empty"},
		{"too many", &fakePredictor{md: testMeta}, `{"instances":[[1,1,1,1],[1,1,1,1],[1,1,1,1],[1,1,1,1]]}`, 400, "exceeds limit"},
		{"wrong dim", &fakePredictor{md: testMeta}, `{"instances":[[1,2,3]]}`, 400, "expected 4 features"},
		{"out of range", &fakePredictor{md: testMeta}, `{"instances":[[1,2,3,-1]]}`, 400, "out of range"},
		{"too large", &fakePredictor{md: testMeta}, `{"instances":[[` + strings.Repeat("1,", 400) + `1]]}`, 413, "too large"},
		{"predict error", &fakePredictor{md: testMeta, err: errors.New("ort failed")}, `{"instances":[[5,3,1,0.2]]}`, 500, "prediction failed"},
		{"panic", &fakePredictor{md: testMeta, panics: true}, `{"instances":[[5,3,1,0.2]]}`, 500, "internal error"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := do(t, newTestServer(tt.pred).Handler(), http.MethodPost, "/v1/predict", tt.body)
			if rec.Code != tt.wantStatus {
				t.Fatalf("status = %d, want %d; body: %s", rec.Code, tt.wantStatus, rec.Body)
			}
			if !strings.Contains(rec.Body.String(), tt.wantInBody) {
				t.Errorf("body %q does not contain %q", rec.Body, tt.wantInBody)
			}
			if ct := rec.Header().Get("Content-Type"); ct != "application/json" {
				t.Errorf("Content-Type = %q", ct)
			}
		})
	}
}

func TestPredictResponseShape(t *testing.T) {
	rec := do(t, newTestServer(&fakePredictor{md: testMeta}).Handler(), http.MethodPost, "/v1/predict",
		`{"instances":[[5,3,1,0.2],[6,3,4,1.3]]}`)
	var resp predictResponse
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatal(err)
	}
	if len(resp.Predictions) != 2 || resp.Model.Name != "iris" {
		t.Errorf("unexpected response %+v", resp)
	}
}

func TestProbes(t *testing.T) {
	tests := []struct {
		name       string
		pred       model.Predictor
		path       string
		wantStatus int
	}{
		{"healthz without model", nil, "/healthz", 200},
		{"readyz without model", nil, "/readyz", 503},
		{"readyz with model", &fakePredictor{md: testMeta}, "/readyz", 200},
		{"model without model", nil, "/v1/model", 503},
		{"model with model", &fakePredictor{md: testMeta}, "/v1/model", 200},
		{"index page", nil, "/", 200},
		{"wrong method", &fakePredictor{md: testMeta}, "/v1/predict", 405},
		{"not found", nil, "/nope", 404},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rec := do(t, newTestServer(tt.pred).Handler(), http.MethodGet, tt.path, "")
			if rec.Code != tt.wantStatus {
				t.Errorf("status = %d, want %d", rec.Code, tt.wantStatus)
			}
		})
	}
}

// trackingPredictor ведёт себя как ONNX: после Close обращаться к нему нельзя.
type trackingPredictor struct {
	fakePredictor
	closed   atomic.Bool
	useAfter *atomic.Int64
}

func (p *trackingPredictor) Predict(ctx context.Context, b [][]float32) ([]model.Prediction, error) {
	time.Sleep(200 * time.Microsecond) // имитация инференса: окно для гонки
	if p.closed.Load() {
		p.useAfter.Add(1)
		return nil, errors.New("use after close")
	}
	return p.fakePredictor.Predict(ctx, b)
}

func (p *trackingPredictor) Close() error { p.closed.Store(true); return nil }

func TestSetPredictorSwap(t *testing.T) {
	s := newTestServer(nil)
	var useAfter atomic.Int64
	first := &trackingPredictor{fakePredictor: fakePredictor{md: testMeta}, useAfter: &useAfter}
	s.SetPredictor(first)

	v2 := testMeta
	v2.Version = "2"
	s.SetPredictor(&trackingPredictor{fakePredictor: fakePredictor{md: v2}, useAfter: &useAfter})

	if !first.closed.Load() {
		t.Error("previous predictor must be closed after swap")
	}
	rec := do(t, s.Handler(), http.MethodGet, "/readyz", "")
	if !strings.Contains(rec.Body.String(), `"version":"2"`) {
		t.Errorf("expected version 2, got %s", rec.Body)
	}

	s.Close()
	if rec := do(t, s.Handler(), http.MethodGet, "/readyz", ""); rec.Code != http.StatusServiceUnavailable {
		t.Errorf("after Close readyz = %d, want 503", rec.Code)
	}
}

// TestHotSwapUnderLoad: модель многократно подменяется во время параллельных запросов.
// Ни один запрос не должен упасть и ни один не должен попасть в закрытую модель.
func TestHotSwapUnderLoad(t *testing.T) {
	s := newTestServer(nil)
	var useAfter atomic.Int64
	newPred := func(v int) *trackingPredictor {
		md := testMeta
		md.Version = strconv.Itoa(v)
		return &trackingPredictor{fakePredictor: fakePredictor{md: md}, useAfter: &useAfter}
	}
	s.SetPredictor(newPred(1))
	h := s.Handler()

	stop := make(chan struct{})
	var wg sync.WaitGroup
	var total, failed atomic.Int64
	for range 16 {
		wg.Go(func() {
			for {
				select {
				case <-stop:
					return
				default:
				}
				rec := do(t, h, http.MethodPost, "/v1/predict", `{"instances":[[5.1,3.5,1.4,0.2]]}`)
				total.Add(1)
				if rec.Code != http.StatusOK {
					failed.Add(1)
				}
			}
		})
	}
	for v := 2; v <= 50; v++ {
		time.Sleep(2 * time.Millisecond)
		s.SetPredictor(newPred(v))
	}
	close(stop)
	wg.Wait()

	if failed.Load() != 0 || useAfter.Load() != 0 {
		t.Fatalf("requests: %d, failed: %d, use-after-close: %d", total.Load(), failed.Load(), useAfter.Load())
	}
	t.Logf("requests: %d, swaps: 49, failed: 0", total.Load())
}

func BenchmarkPredictHandler(b *testing.B) {
	h := newTestServer(&fakePredictor{md: testMeta}).Handler()
	body := []byte(`{"instances":[[5.1,3.5,1.4,0.2]]}`)
	b.ReportAllocs()
	for b.Loop() {
		req := httptest.NewRequest(http.MethodPost, "/v1/predict", bytes.NewReader(body))
		rec := httptest.NewRecorder()
		h.ServeHTTP(rec, req)
		if rec.Code != 200 {
			b.Fatal(rec.Code)
		}
	}
}
