package registry

import (
	"context"
	"fmt"
	"io"

	"github.com/minio/minio-go/v7"
	"github.com/minio/minio-go/v7/pkg/credentials"
)

// Blob хранит файлы моделей в S3-совместимом хранилище (RustFS, MinIO, AWS S3).
type Blob struct {
	client *minio.Client
	bucket string
}

type S3Config struct {
	Endpoint  string
	AccessKey string
	SecretKey string
	Bucket    string
	UseSSL    bool
}

func NewBlob(cfg S3Config) (*Blob, error) {
	c, err := minio.New(cfg.Endpoint, &minio.Options{
		Creds:  credentials.NewStaticV4(cfg.AccessKey, cfg.SecretKey, ""),
		Secure: cfg.UseSSL,
	})
	if err != nil {
		return nil, fmt.Errorf("s3 client: %w", err)
	}
	return &Blob{client: c, bucket: cfg.Bucket}, nil
}

// EnsureBucket создаёт бакет, если его ещё нет.
func (b *Blob) EnsureBucket(ctx context.Context) error {
	ok, err := b.client.BucketExists(ctx, b.bucket)
	if err != nil {
		return fmt.Errorf("check bucket: %w", err)
	}
	if ok {
		return nil
	}
	if err := b.client.MakeBucket(ctx, b.bucket, minio.MakeBucketOptions{}); err != nil {
		return fmt.Errorf("create bucket: %w", err)
	}
	return nil
}

func (b *Blob) Put(ctx context.Context, key string, r io.Reader, size int64, contentType string) error {
	_, err := b.client.PutObject(ctx, b.bucket, key, r, size, minio.PutObjectOptions{ContentType: contentType})
	if err != nil {
		return fmt.Errorf("put %s: %w", key, err)
	}
	return nil
}

// Copy копирует объект на стороне сервера, не прокачивая данные через наш сервис.
func (b *Blob) Copy(ctx context.Context, src, dst string) error {
	_, err := b.client.CopyObject(ctx,
		minio.CopyDestOptions{Bucket: b.bucket, Object: dst},
		minio.CopySrcOptions{Bucket: b.bucket, Object: src})
	if err != nil {
		return fmt.Errorf("copy %s -> %s: %w", src, dst, err)
	}
	return nil
}

// Get открывает объект на чтение. Вызывающий обязан закрыть reader.
func (b *Blob) Get(ctx context.Context, key string) (io.ReadCloser, int64, error) {
	obj, err := b.client.GetObject(ctx, b.bucket, key, minio.GetObjectOptions{})
	if err != nil {
		return nil, 0, fmt.Errorf("get %s: %w", key, err)
	}
	st, err := obj.Stat() // GetObject ленивый: ошибка «нет объекта» видна только здесь
	if err != nil {
		obj.Close()
		if minio.ToErrorResponse(err).Code == minio.NoSuchKey {
			return nil, 0, ErrNotFound
		}
		return nil, 0, fmt.Errorf("stat %s: %w", key, err)
	}
	return obj, st.Size, nil
}

func (b *Blob) Delete(ctx context.Context, key string) error {
	if err := b.client.RemoveObject(ctx, b.bucket, key, minio.RemoveObjectOptions{}); err != nil {
		return fmt.Errorf("delete %s: %w", key, err)
	}
	return nil
}
