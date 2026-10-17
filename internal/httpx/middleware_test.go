package httpx

import (
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"
)

var discard = slog.New(slog.NewTextHandler(io.Discard, nil))

func TestRequestID(t *testing.T) {
	var seen string
	h := Wrap(discard, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = RequestID(r.Context())
	}))

	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if id := rec.Header().Get(RequestIDHeader); len(id) != 16 || id != seen {
		t.Errorf("generated id %q (in ctx %q), want 16 hex chars", id, seen)
	}

	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set(RequestIDHeader, "abc-123")
	rec = httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	if id := rec.Header().Get(RequestIDHeader); id != "abc-123" {
		t.Errorf("propagated id = %q", id)
	}
}

func TestRecoverer(t *testing.T) {
	h := Wrap(discard, http.HandlerFunc(func(http.ResponseWriter, *http.Request) { panic("boom") }))
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	if rec.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", rec.Code)
	}
}
