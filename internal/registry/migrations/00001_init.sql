-- +goose Up
CREATE TABLE models (
    id         BIGSERIAL PRIMARY KEY,
    name       TEXT        NOT NULL UNIQUE CHECK (name ~ '^[a-z0-9][a-z0-9_-]{0,62}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE model_versions (
    id               BIGSERIAL PRIMARY KEY,
    model_id         BIGINT      NOT NULL REFERENCES models (id) ON DELETE CASCADE,
    version          INT         NOT NULL CHECK (version > 0),
    stage            TEXT        NOT NULL DEFAULT 'none'
                     CHECK (stage IN ('none', 'staging', 'production', 'archived')),
    sha256           TEXT        NOT NULL CHECK (sha256 ~ '^[0-9a-f]{64}$'),
    size_bytes       BIGINT      NOT NULL CHECK (size_bytes > 0),
    metadata         JSONB       NOT NULL,
    created_by       TEXT        NOT NULL DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    stage_updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (model_id, version)
);

-- Гарантия на уровне БД: у модели не больше одной production-версии.
CREATE UNIQUE INDEX model_versions_one_production
    ON model_versions (model_id) WHERE stage = 'production';

-- +goose Down
DROP TABLE model_versions;
DROP TABLE models;
