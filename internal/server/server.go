// Package server — HTTP API inference-сервиса.
package server

import (
	_ "embed"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"sync/atomic"

	"mlops/internal/model"
)

// indexHTML — страница для ручной проверки модели из браузера.
//
//go:embed ui/index.html
var indexHTML []byte

type Options struct {
	MaxBatch     int
	MaxBodyBytes int64
}

// Server держит текущую модель в atomic-указателе: её можно заменить на лету
// (пригодится для горячей перезагрузки из реестра в проекте 2).
type Server struct {
	opts   Options
	log    *slog.Logger
	active atomic.Pointer[loaded]
}

type loaded struct{ p model.Predictor }

func New(opts Options, log *slog.Logger) *Server {
	return &Server{opts: opts, log: log}
}

// SetPredictor атомарно подменяет модель и возвращает предыдущую (или nil).
func (s *Server) SetPredictor(p model.Predictor) model.Predictor {
	old := s.active.Swap(&loaded{p: p})
	if old == nil {
		return nil
	}
	return old.p
}

func (s *Server) predictor() model.Predictor {
	if l := s.active.Load(); l != nil {
		return l.p
	}
	return nil
}

// Handler возвращает роутер со всеми middleware.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/predict", s.handlePredict)
	mux.HandleFunc("GET /v1/model", s.handleModel)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(indexHTML)
	})
	return requestID(recoverer(s.log, logging(s.log, mux)))
}

type predictRequest struct {
	Instances [][]float32 `json:"instances"`
}

type modelRef struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type predictResponse struct {
	Model       modelRef           `json:"model"`
	Predictions []model.Prediction `json:"predictions"`
}

func (s *Server) handlePredict(w http.ResponseWriter, r *http.Request) {
	p := s.predictor()
	if p == nil {
		writeError(w, http.StatusServiceUnavailable, "model is not loaded")
		return
	}

	r.Body = http.MaxBytesReader(w, r.Body, s.opts.MaxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	var req predictRequest
	if err := dec.Decode(&req); err != nil {
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			writeError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		writeError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	md := p.Metadata()
	if err := validate(req.Instances, md.Features, s.opts.MaxBatch); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	preds, err := p.Predict(r.Context(), req.Instances)
	if err != nil {
		s.log.ErrorContext(r.Context(), "predict failed", "err", err, "request_id", requestIDFrom(r.Context()))
		writeError(w, http.StatusInternalServerError, "prediction failed")
		return
	}
	writeJSON(w, http.StatusOK, predictResponse{
		Model:       modelRef{Name: md.Name, Version: md.Version},
		Predictions: preds,
	})
}

// validate проверяет размер батча, размерность и диапазоны фичей.
func validate(batch [][]float32, features []model.Feature, maxBatch int) error {
	if len(batch) == 0 {
		return errors.New("instances must be non-empty")
	}
	if len(batch) > maxBatch {
		return fmt.Errorf("batch size %d exceeds limit %d", len(batch), maxBatch)
	}
	for i, row := range batch {
		if len(row) != len(features) {
			return fmt.Errorf("instance %d: expected %d features, got %d", i, len(features), len(row))
		}
		for j, v := range row {
			f := features[j]
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return fmt.Errorf("instance %d: feature %q is not finite", i, f.Name)
			}
			if v < f.Min || v > f.Max {
				return fmt.Errorf("instance %d: feature %q=%g out of range [%g, %g]", i, f.Name, v, f.Min, f.Max)
			}
		}
	}
	return nil
}

func (s *Server) handleModel(w http.ResponseWriter, _ *http.Request) {
	p := s.predictor()
	if p == nil {
		writeError(w, http.StatusServiceUnavailable, "model is not loaded")
		return
	}
	writeJSON(w, http.StatusOK, p.Metadata())
}

func (s *Server) handleReady(w http.ResponseWriter, _ *http.Request) {
	p := s.predictor()
	if p == nil {
		writeJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready"})
		return
	}
	md := p.Metadata()
	writeJSON(w, http.StatusOK, map[string]string{"status": "ready", "model": md.Name, "version": md.Version})
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}
