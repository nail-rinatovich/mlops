//go:build !cgo

package model

import (
	"context"
	"errors"
)

// Заглушка для сборки без CGO (например, локальные unit-тесты на Windows без gcc).
// Реальный инференс доступен только в сборке с CGO_ENABLED=1.

var errNoCGO = errors.New("onnx runtime requires a cgo build (CGO_ENABLED=1)")

func InitRuntime(string) error { return errNoCGO }
func DestroyRuntime() error    { return nil }

type ONNXPredictor struct{}

func LoadONNX(string) (*ONNXPredictor, error) { return nil, errNoCGO }

func (*ONNXPredictor) Metadata() Metadata { return Metadata{} }
func (*ONNXPredictor) Predict(context.Context, [][]float32) ([]Prediction, error) {
	return nil, errNoCGO
}
func (*ONNXPredictor) Close() error { return nil }
