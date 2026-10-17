package registry

import (
	"context"
	"crypto/subtle"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"strconv"
	"strings"

	"mlops/internal/httpx"
	api "mlops/internal/registryapi"
)

// API — HTTP-интерфейс реестра.
type API struct {
	svc            *Service
	log            *slog.Logger
	token          string // если пусто, запись открыта всем (только для локальной разработки)
	maxUploadBytes int64
	ready          func(ctx context.Context) error
}

func NewAPI(svc *Service, log *slog.Logger, token string, maxUploadBytes int64, ready func(context.Context) error) *API {
	return &API{svc: svc, log: log, token: token, maxUploadBytes: maxUploadBytes, ready: ready}
}

func (a *API) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models", a.listModels)
	mux.HandleFunc("GET /v1/models/{name}/versions", a.listVersions)
	mux.HandleFunc("POST /v1/models/{name}/versions", a.auth(a.upload))
	mux.HandleFunc("GET /v1/models/{name}/versions/{ref}", a.getVersion)
	mux.HandleFunc("PUT /v1/models/{name}/versions/{version}/stage", a.auth(a.setStage))
	mux.HandleFunc("GET /v1/models/{name}/versions/{version}/files/{file}", a.getFile)
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ok"})
	})
	mux.HandleFunc("GET /readyz", func(w http.ResponseWriter, r *http.Request) {
		if err := a.ready(r.Context()); err != nil {
			httpx.WriteJSON(w, http.StatusServiceUnavailable, map[string]string{"status": "not ready", "error": err.Error()})
			return
		}
		httpx.WriteJSON(w, http.StatusOK, map[string]string{"status": "ready"})
	})
	return httpx.Wrap(a.log, mux)
}

// auth защищает изменяющие запросы Bearer-токеном. Сравнение за постоянное время,
// чтобы токен нельзя было подобрать по времени ответа.
func (a *API) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if a.token == "" {
			next(w, r)
			return
		}
		got, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || subtle.ConstantTimeCompare([]byte(got), []byte(a.token)) != 1 {
			httpx.WriteError(w, http.StatusUnauthorized, "missing or invalid token")
			return
		}
		next(w, r)
	}
}

// writeErr переводит доменные ошибки в HTTP-статусы.
func (a *API) writeErr(w http.ResponseWriter, r *http.Request, err error) {
	switch {
	case errors.Is(err, ErrNotFound):
		httpx.WriteError(w, http.StatusNotFound, "not found")
	case errors.Is(err, ErrInvalid):
		httpx.WriteError(w, http.StatusBadRequest, err.Error())
	case errors.Is(err, ErrChecksum):
		httpx.WriteError(w, http.StatusUnprocessableEntity, err.Error())
	default:
		a.log.ErrorContext(r.Context(), "request failed", "err", err, "request_id", httpx.RequestID(r.Context()))
		httpx.WriteError(w, http.StatusInternalServerError, "internal error")
	}
}

func (a *API) listModels(w http.ResponseWriter, r *http.Request) {
	ms, err := a.svc.ListModels(r.Context())
	if err != nil {
		a.writeErr(w, r, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, ms)
}

func (a *API) listVersions(w http.ResponseWriter, r *http.Request) {
	vs, err := a.svc.ListVersions(r.Context(), r.PathValue("name"))
	if err != nil {
		a.writeErr(w, r, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, vs)
}

func (a *API) getVersion(w http.ResponseWriter, r *http.Request) {
	v, err := a.svc.Resolve(r.Context(), r.PathValue("name"), r.PathValue("ref"))
	if err != nil {
		a.writeErr(w, r, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, v)
}

// upload принимает multipart/form-data с полями:
//
//	model      — файл model.onnx
//	metadata   — файл metadata.json
//	created_by — кто загрузил (необязательно)
func (a *API) upload(w http.ResponseWriter, r *http.Request) {
	r.Body = http.MaxBytesReader(w, r.Body, a.maxUploadBytes)
	// До 32 МБ держим в памяти, остальное Go сам сбросит во временные файлы.
	if err := r.ParseMultipartForm(32 << 20); err != nil {
		if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
			httpx.WriteError(w, http.StatusRequestEntityTooLarge, "upload too large")
			return
		}
		httpx.WriteError(w, http.StatusBadRequest, "invalid multipart form: "+err.Error())
		return
	}
	defer r.MultipartForm.RemoveAll()

	mf, mh, err := r.FormFile("model")
	if err != nil {
		httpx.WriteError(w, http.StatusBadRequest, "missing file field 'model'")
		return
	}
	defer mf.Close()
	metaFile, _, err := r.FormFile("metadata")
	if err != nil {
		httpx.WriteError(w, http.StatusBadRequest, "missing file field 'metadata'")
		return
	}
	defer metaFile.Close()
	metaJSON, err := io.ReadAll(io.LimitReader(metaFile, 1<<20))
	if err != nil {
		httpx.WriteError(w, http.StatusBadRequest, "read metadata: "+err.Error())
		return
	}

	v, err := a.svc.Upload(r.Context(), r.PathValue("name"), r.FormValue("created_by"), mf, mh.Size, metaJSON)
	if err != nil {
		a.writeErr(w, r, err)
		return
	}
	a.log.InfoContext(r.Context(), "version registered", "model", v.Name, "version", v.Version,
		"sha256", v.SHA256, "size_bytes", v.SizeBytes, "created_by", v.CreatedBy)
	httpx.WriteJSON(w, http.StatusCreated, v)
}

func (a *API) setStage(w http.ResponseWriter, r *http.Request) {
	version, err := strconv.Atoi(r.PathValue("version"))
	if err != nil {
		httpx.WriteError(w, http.StatusBadRequest, "version must be a number")
		return
	}
	var req api.SetStageRequest
	if err := json.NewDecoder(io.LimitReader(r.Body, 4096)).Decode(&req); err != nil {
		httpx.WriteError(w, http.StatusBadRequest, "invalid JSON: "+err.Error())
		return
	}
	v, err := a.svc.SetStage(r.Context(), r.PathValue("name"), version, req.Stage)
	if err != nil {
		a.writeErr(w, r, err)
		return
	}
	httpx.WriteJSON(w, http.StatusOK, v)
}

// getFile отдаёт файл версии потоком из S3. Для model.onnx в заголовке
// X-Checksum-SHA256 передаётся ожидаемый хэш, чтобы клиент сверил скачанное.
func (a *API) getFile(w http.ResponseWriter, r *http.Request) {
	version, err := strconv.Atoi(r.PathValue("version"))
	if err != nil {
		httpx.WriteError(w, http.StatusBadRequest, "version must be a number")
		return
	}
	file := r.PathValue("file")
	rc, size, v, err := a.svc.OpenFile(r.Context(), r.PathValue("name"), version, file)
	if err != nil {
		a.writeErr(w, r, err)
		return
	}
	defer rc.Close()

	ct := "application/octet-stream"
	if file == api.FileMetadata {
		ct = "application/json"
	} else {
		w.Header().Set("X-Checksum-SHA256", v.SHA256)
	}
	w.Header().Set("Content-Type", ct)
	w.Header().Set("Content-Length", strconv.FormatInt(size, 10))
	if _, err := io.Copy(w, rc); err != nil {
		a.log.WarnContext(r.Context(), "stream file", "err", err, "request_id", httpx.RequestID(r.Context()))
	}
}
