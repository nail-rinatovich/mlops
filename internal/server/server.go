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
	"sync"
	"sync/atomic"

	"mlops/internal/httpx"
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

// Server держит текущую модель в atomic-указателе, поэтому её можно заменить на лету.
type Server struct {
	opts   Options
	log    *slog.Logger
	active atomic.Pointer[loaded]
	swapMu sync.Mutex // сериализует SetPredictor/Close между собой
}

// loaded — загруженная модель со счётчиком использования. Запросы держат RLock
// на время инференса; при замене модели Lock дожидается их завершения, и только
// после этого старая модель закрывается. Так ни один запрос не обратится
// к уже освобождённой памяти ONNX Runtime.
type loaded struct {
	p      model.Predictor
	mu     sync.RWMutex
	closed bool
}

func New(opts Options, log *slog.Logger) *Server {
	return &Server{opts: opts, log: log}
}

// SetPredictor атомарно подменяет модель. Предыдущая модель закрывается после того,
// как завершатся все запросы, которые успели её взять.
func (s *Server) SetPredictor(p model.Predictor) {
	s.swapMu.Lock()
	defer s.swapMu.Unlock()
	s.retire(s.active.Swap(&loaded{p: p}))
}

// Close выгружает текущую модель (при остановке сервиса).
func (s *Server) Close() {
	s.swapMu.Lock()
	defer s.swapMu.Unlock()
	s.retire(s.active.Swap(nil))
}

func (s *Server) retire(old *loaded) {
	if old == nil {
		return
	}
	old.mu.Lock() // ждём, пока доработают запросы на старой модели
	old.closed = true
	old.mu.Unlock()
	if err := old.p.Close(); err != nil {
		s.log.Error("close predictor", "err", err)
	}
}

// acquire берёт текущую модель для одного запроса. Вызывающий обязан вызвать release.
// Если модель успели закрыть между Load и RLock, берём уже новую.
func (s *Server) acquire() (*loaded, bool) {
	for {
		l := s.active.Load()
		if l == nil {
			return nil, false
		}
		l.mu.RLock()
		if !l.closed {
			return l, true
		}
		l.mu.RUnlock()
	}
}

func (l *loaded) release() { l.mu.RUnlock() }

// Handler возвращает роутер со всеми middleware.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/predict", s.handlePredict)
	mux.HandleFunc("GET /v1/model", s.handleModel)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /readyz", s.handleReady)
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write(indexHTML)
	})
	return httpx.Wrap(s.log, mux)
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
	// Тело читаем до захвата модели: медленный клиент не должен задерживать её замену.
	r.Body = http.MaxBytesReader(w, r.Body, s.opts.MaxBodyBytes)
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	var req predictRequest
	if err := dec.Decode(&req); err != nil {
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			httpx.WriteError(w, http.StatusRequestEntityTooLarge, "request body too large")
			return
		}
		httpx.WriteError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}

	l, ok := s.acquire()
	if !ok {
		httpx.WriteError(w, http.StatusServiceUnavailable, "model is not loaded")
		return
	}
	md := l.p.Metadata()
	if err := validate(req.Instances, md.Features, s.opts.MaxBatch); err != nil {
		l.release()
		httpx.WriteError(w, http.StatusBadRequest, err.Error())
		return
	}
	preds, err := l.p.Predict(r.Context(), req.Instances)
	l.release()
	if err != nil {
		s.log.ErrorContext(r.Context(), "predict failed", "err", err, "request_id", httpx.RequestID(r.Context()))
		httpx.WriteError(w, http.StatusInternalServerError, "prediction failed")
		return
	}
	httpx.WriteJSON(w, http.StatusOK, predictResponse{
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

func (s *Server) metadata() (model.Metadata, bool) {
	l, ok := s.acquire()
	if !ok {
		return model.Metadata{}, false
	}
	defer l.release()
	return l.p.Metadata(), true
}

func (s *Server) handleModel(w http.ResponseWriter, _ *http.Request) {
	md, ok := s.metadata()
	if !ok {
		httpx.WriteError(w, http.StatusServiceUnavailable, "model is not loaded")
		return
	}
	httpx.WriteJSON(w, http.StatusOK, md)
}

func (s *Server) handleReady(w http.ResponseWriter, _ *http.Request) {
	md, ok := s.metadata()
	if !ok {
		httpx.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready"})
		return
	}
	httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ready", "model": md.Name, "version": md.Version})
}
