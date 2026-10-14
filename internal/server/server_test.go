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
	"strings"
	"testing"

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

func TestSetPredictorSwap(t *testing.T) {
	s := newTestServer(nil)
	first := &fakePredictor{md: testMeta}
	if old := s.SetPredictor(first); old != nil {
		t.Errorf("expected nil old predictor")
	}
	v2 := testMeta
	v2.Version = "2"
	if old := s.SetPredictor(&fakePredictor{md: v2}); old != first {
		t.Errorf("expected first predictor to be returned")
	}
	rec := do(t, s.Handler(), http.MethodGet, "/readyz", "")
	if !strings.Contains(rec.Body.String(), `"version":"2"`) {
		t.Errorf("expected version 2, got %s", rec.Body)
	}
}

func TestRequestID(t *testing.T) {
	h := newTestServer(nil).Handler()

	rec := do(t, h, http.MethodGet, "/healthz", "")
	if id := rec.Header().Get(requestIDHeader); len(id) != 16 {
		t.Errorf("generated id %q, want 16 hex chars", id)
	}

	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set(requestIDHeader, "abc-123")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if id := rec.Header().Get(requestIDHeader); id != "abc-123" {
		t.Errorf("propagated id = %q", id)
	}
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
