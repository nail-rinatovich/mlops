// Package registryclient — HTTP-клиент реестра моделей (для CLI mreg и inference).
package registryclient

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	api "mlops/internal/registryapi"
)

// ErrNotFound — модель, версия или стадия не найдены.
var ErrNotFound = errors.New("not found")

type Client struct {
	base  string
	token string
	http  *http.Client
}

func New(baseURL, token string, hc *http.Client) *Client {
	if hc == nil {
		hc = http.DefaultClient
	}
	return &Client{base: strings.TrimRight(baseURL, "/"), token: token, http: hc}
}

func (c *Client) url(parts ...string) string {
	esc := make([]string, len(parts))
	for i, p := range parts {
		esc[i] = url.PathEscape(p)
	}
	return c.base + "/v1/models/" + strings.Join(esc, "/")
}

func (c *Client) do(req *http.Request, out any) error {
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return responseError(resp)
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

func responseError(resp *http.Response) error {
	var e api.ErrorResponse
	_ = json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&e)
	if resp.StatusCode == http.StatusNotFound {
		return ErrNotFound
	}
	if e.Error == "" {
		e.Error = http.StatusText(resp.StatusCode)
	}
	return fmt.Errorf("registry: %d %s", resp.StatusCode, e.Error)
}

func (c *Client) get(ctx context.Context, u string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u, nil)
	if err != nil {
		return err
	}
	return c.do(req, out)
}

func (c *Client) ListModels(ctx context.Context) ([]api.ModelSummary, error) {
	var out []api.ModelSummary
	return out, c.get(ctx, c.base+"/v1/models", &out)
}

func (c *Client) ListVersions(ctx context.Context, name string) ([]api.Version, error) {
	var out []api.Version
	return out, c.get(ctx, c.url(name, "versions"), &out)
}

// GetVersion получает версию по номеру ("3") или стадии ("production").
func (c *Client) GetVersion(ctx context.Context, name, ref string) (api.Version, error) {
	var out api.Version
	return out, c.get(ctx, c.url(name, "versions", ref), &out)
}

func (c *Client) SetStage(ctx context.Context, name string, version int, stage api.Stage) (api.Version, error) {
	body, _ := json.Marshal(api.SetStageRequest{Stage: stage})
	req, err := http.NewRequestWithContext(ctx, http.MethodPut,
		c.url(name, "versions", strconv.Itoa(version), "stage"), bytes.NewReader(body))
	if err != nil {
		return api.Version{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	var out api.Version
	return out, c.do(req, &out)
}

// Push загружает model.onnx и metadata.json из dir как новую версию модели.
// Файл модели передаётся потоком через io.Pipe, без чтения целиком в память.
func (c *Client) Push(ctx context.Context, name, dir, createdBy string) (api.Version, error) {
	files := map[string]string{
		"metadata": filepath.Join(dir, api.FileMetadata),
		"model":    filepath.Join(dir, api.FileModel),
	}
	for _, p := range files {
		if _, err := os.Stat(p); err != nil {
			return api.Version{}, err
		}
	}

	pr, pw := io.Pipe()
	mw := multipart.NewWriter(pw)
	go func() {
		err := func() error {
			if err := mw.WriteField("created_by", createdBy); err != nil {
				return err
			}
			for _, field := range []string{"metadata", "model"} {
				f, err := os.Open(files[field])
				if err != nil {
					return err
				}
				part, err := mw.CreateFormFile(field, filepath.Base(files[field]))
				if err == nil {
					_, err = io.Copy(part, f)
				}
				f.Close()
				if err != nil {
					return err
				}
			}
			return mw.Close()
		}()
		pw.CloseWithError(err)
	}()

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.url(name, "versions"), pr)
	if err != nil {
		pr.Close()
		return api.Version{}, err
	}
	req.Header.Set("Content-Type", mw.FormDataContentType())
	var out api.Version
	return out, c.do(req, &out)
}

// Download скачивает файлы версии в каталог dest и проверяет SHA256 модели.
// Файлы сначала пишутся во временный каталог рядом и переименовываются в dest
// одним действием, поэтому в dest никогда не окажется недокачанная модель.
func (c *Client) Download(ctx context.Context, v api.Version, dest string) error {
	parent := filepath.Dir(dest)
	if err := os.MkdirAll(parent, 0o755); err != nil {
		return err
	}
	tmp, err := os.MkdirTemp(parent, ".download-*")
	if err != nil {
		return err
	}
	defer os.RemoveAll(tmp) // после успешного Rename каталога уже нет, и это no-op

	for _, file := range []string{api.FileMetadata, api.FileModel} {
		sum, err := c.downloadFile(ctx, v, file, filepath.Join(tmp, file))
		if err != nil {
			return err
		}
		if file == api.FileModel && sum != v.SHA256 {
			return fmt.Errorf("checksum mismatch for %s v%d: expected %s, got %s", v.Name, v.Version, v.SHA256, sum)
		}
	}
	if err := os.RemoveAll(dest); err != nil {
		return err
	}
	return os.Rename(tmp, dest)
}

func (c *Client) downloadFile(ctx context.Context, v api.Version, file, path string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet,
		c.url(v.Name, "versions", strconv.Itoa(v.Version), "files", file), nil)
	if err != nil {
		return "", err
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", responseError(resp)
	}

	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	h := sha256.New()
	_, err = io.Copy(io.MultiWriter(f, h), resp.Body)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return "", fmt.Errorf("download %s: %w", file, err)
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}
