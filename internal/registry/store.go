package registry

import (
	"context"
	"embed"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"strconv"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/jackc/pgx/v5/stdlib"
	"github.com/pressly/goose/v3"

	api "mlops/internal/registryapi"
)

//go:embed migrations/*.sql
var migrationsFS embed.FS

var (
	ErrNotFound = errors.New("not found")
	ErrInvalid  = errors.New("invalid argument")
)

// Store хранит метаданные моделей и версий в PostgreSQL.
type Store struct {
	pool *pgxpool.Pool
}

func NewStore(pool *pgxpool.Pool) *Store { return &Store{pool: pool} }

// Migrate применяет встроенные в бинарник миграции goose.
func Migrate(ctx context.Context, pool *pgxpool.Pool) ([]string, error) {
	db := stdlib.OpenDBFromPool(pool)
	defer db.Close()
	sub, err := fs.Sub(migrationsFS, "migrations")
	if err != nil {
		return nil, err
	}
	p, err := goose.NewProvider(goose.DialectPostgres, db, sub)
	if err != nil {
		return nil, fmt.Errorf("goose provider: %w", err)
	}
	res, err := p.Up(ctx)
	if err != nil {
		return nil, fmt.Errorf("migrate: %w", err)
	}
	applied := make([]string, 0, len(res))
	for _, r := range res {
		applied = append(applied, r.Source.Path)
	}
	return applied, nil
}

const versionColumns = `m.name, v.version, v.stage, v.sha256, v.size_bytes, v.metadata,
	v.created_by, v.created_at, v.stage_updated_at`

func scanVersion(row pgx.Row) (api.Version, error) {
	var v api.Version
	var meta []byte
	err := row.Scan(&v.Name, &v.Version, &v.Stage, &v.SHA256, &v.SizeBytes, &meta,
		&v.CreatedBy, &v.CreatedAt, &v.StageUpdatedAt)
	if errors.Is(err, pgx.ErrNoRows) {
		return v, ErrNotFound
	}
	if err != nil {
		return v, err
	}
	if err := json.Unmarshal(meta, &v.Metadata); err != nil {
		return v, fmt.Errorf("decode metadata: %w", err)
	}
	return v, nil
}

// CreateVersion атомарно выделяет следующий номер версии и регистрирует её.
// publish вызывается внутри транзакции, когда номер версии уже присвоен: туда выносится
// публикация артефактов в S3. Если publish вернёт ошибку, версия не появится.
func (s *Store) CreateVersion(ctx context.Context, v api.Version, publish func(api.Version) error) (api.Version, error) {
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return v, err
	}
	defer tx.Rollback(ctx)

	// Upsert берёт блокировку строки модели до конца транзакции: параллельные
	// загрузки одной модели выстраиваются в очередь и не получат одинаковый номер.
	var modelID int64
	err = tx.QueryRow(ctx, `INSERT INTO models (name) VALUES ($1)
		ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name RETURNING id`, v.Name).Scan(&modelID)
	if err != nil {
		return v, fmt.Errorf("upsert model: %w", err)
	}
	if err := tx.QueryRow(ctx, `SELECT COALESCE(MAX(version), 0) + 1 FROM model_versions WHERE model_id = $1`,
		modelID).Scan(&v.Version); err != nil {
		return v, fmt.Errorf("next version: %w", err)
	}

	v.Metadata.Name = v.Name
	v.Metadata.Version = strconv.Itoa(v.Version)
	v.Metadata.SHA256 = v.SHA256
	if err := publish(v); err != nil {
		return v, err
	}

	meta, err := json.Marshal(v.Metadata)
	if err != nil {
		return v, err
	}
	err = tx.QueryRow(ctx, `INSERT INTO model_versions (model_id, version, sha256, size_bytes, metadata, created_by)
		VALUES ($1, $2, $3, $4, $5, $6) RETURNING stage, created_at, stage_updated_at`,
		modelID, v.Version, v.SHA256, v.SizeBytes, meta, v.CreatedBy).Scan(&v.Stage, &v.CreatedAt, &v.StageUpdatedAt)
	if err != nil {
		return v, fmt.Errorf("insert version: %w", err)
	}
	return v, tx.Commit(ctx)
}

func (s *Store) GetVersion(ctx context.Context, name string, version int) (api.Version, error) {
	return scanVersion(s.pool.QueryRow(ctx, `SELECT `+versionColumns+`
		FROM model_versions v JOIN models m ON m.id = v.model_id
		WHERE m.name = $1 AND v.version = $2`, name, version))
}

// GetByStage возвращает самую свежую версию в указанной стадии.
func (s *Store) GetByStage(ctx context.Context, name string, stage api.Stage) (api.Version, error) {
	return scanVersion(s.pool.QueryRow(ctx, `SELECT `+versionColumns+`
		FROM model_versions v JOIN models m ON m.id = v.model_id
		WHERE m.name = $1 AND v.stage = $2 ORDER BY v.version DESC LIMIT 1`, name, stage))
}

func (s *Store) ListVersions(ctx context.Context, name string) ([]api.Version, error) {
	rows, err := s.pool.Query(ctx, `SELECT `+versionColumns+`
		FROM model_versions v JOIN models m ON m.id = v.model_id
		WHERE m.name = $1 ORDER BY v.version DESC`, name)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []api.Version{}
	for rows.Next() {
		v, err := scanVersion(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, v)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}
	if len(out) == 0 {
		return nil, ErrNotFound
	}
	return out, nil
}

func (s *Store) ListModels(ctx context.Context) ([]api.ModelSummary, error) {
	rows, err := s.pool.Query(ctx, `SELECT m.name, m.created_at,
			COALESCE(MAX(v.version), 0),
			MAX(v.version) FILTER (WHERE v.stage = 'production')
		FROM models m LEFT JOIN model_versions v ON v.model_id = m.id
		GROUP BY m.id ORDER BY m.name`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	out := []api.ModelSummary{}
	for rows.Next() {
		var ms api.ModelSummary
		if err := rows.Scan(&ms.Name, &ms.CreatedAt, &ms.LatestVersion, &ms.ProductionVersion); err != nil {
			return nil, err
		}
		out = append(out, ms)
	}
	return out, rows.Err()
}

// SetStage переводит версию в новую стадию. При переводе в production
// предыдущая production-версия в той же транзакции уходит в archived.
func (s *Store) SetStage(ctx context.Context, name string, version int, stage api.Stage) (api.Version, error) {
	if !stage.Valid() {
		return api.Version{}, fmt.Errorf("%w: unknown stage %q", ErrInvalid, stage)
	}
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return api.Version{}, err
	}
	defer tx.Rollback(ctx)

	var modelID int64
	err = tx.QueryRow(ctx, `SELECT id FROM models WHERE name = $1 FOR UPDATE`, name).Scan(&modelID)
	if errors.Is(err, pgx.ErrNoRows) {
		return api.Version{}, ErrNotFound
	}
	if err != nil {
		return api.Version{}, err
	}

	if stage == api.StageProduction {
		if _, err := tx.Exec(ctx, `UPDATE model_versions SET stage = 'archived', stage_updated_at = now()
			WHERE model_id = $1 AND stage = 'production' AND version <> $2`, modelID, version); err != nil {
			return api.Version{}, fmt.Errorf("archive previous production: %w", err)
		}
	}
	tag, err := tx.Exec(ctx, `UPDATE model_versions SET stage = $3, stage_updated_at = now()
		WHERE model_id = $1 AND version = $2`, modelID, version, stage)
	if err != nil {
		return api.Version{}, err
	}
	if tag.RowsAffected() == 0 {
		return api.Version{}, ErrNotFound
	}
	if err := tx.Commit(ctx); err != nil {
		return api.Version{}, err
	}
	return s.GetVersion(ctx, name, version)
}
