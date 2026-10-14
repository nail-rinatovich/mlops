//go:build cgo

package model

import (
	"context"
	"fmt"
	"path/filepath"

	ort "github.com/yalue/onnxruntime_go"
)

const (
	inputName = "input"
	labelName = "label"
	probsName = "probabilities"
	onnxFile  = "model.onnx"
)

// InitRuntime инициализирует глобальное окружение ONNX Runtime. Вызывается один раз на процесс.
func InitRuntime(libPath string) error {
	if libPath != "" {
		ort.SetSharedLibraryPath(libPath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return fmt.Errorf("init onnxruntime: %w", err)
	}
	return nil
}

// DestroyRuntime освобождает окружение ONNX Runtime. Вызывать после Close всех предикторов.
func DestroyRuntime() error { return ort.DestroyEnvironment() }

// ONNXPredictor — Predictor поверх ONNX Runtime. Run у ORT потокобезопасен,
// а выходные тензоры выделяются на каждый вызов, поэтому мьютекс не нужен.
type ONNXPredictor struct {
	session  *ort.DynamicAdvancedSession
	meta     Metadata
	nFeature int
}

// LoadONNX загружает model.onnx и metadata.json из dir и проверяет, что они согласованы.
func LoadONNX(dir string) (*ONNXPredictor, error) {
	meta, err := LoadMetadata(dir)
	if err != nil {
		return nil, err
	}
	path := filepath.Join(dir, onnxFile)
	if err := checkSignature(path, len(meta.Features)); err != nil {
		return nil, err
	}

	opts, err := ort.NewSessionOptions()
	if err != nil {
		return nil, fmt.Errorf("session options: %w", err)
	}
	defer opts.Destroy()
	// Параллелим на уровне запросов (горутины), а не внутри одного маленького графа.
	if err := opts.SetIntraOpNumThreads(1); err != nil {
		return nil, fmt.Errorf("set threads: %w", err)
	}

	s, err := ort.NewDynamicAdvancedSession(path, []string{inputName}, []string{labelName, probsName}, opts)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	return &ONNXPredictor{session: s, meta: meta, nFeature: len(meta.Features)}, nil
}

func checkSignature(path string, nFeature int) error {
	ins, outs, err := ort.GetInputOutputInfo(path)
	if err != nil {
		return fmt.Errorf("inspect model: %w", err)
	}
	if len(ins) != 1 || ins[0].Name != inputName {
		return fmt.Errorf("model must have single input %q, got %v", inputName, ins)
	}
	if d := ins[0].Dimensions; len(d) != 2 || d[1] != int64(nFeature) {
		return fmt.Errorf("model input shape %v does not match %d features in metadata", d, nFeature)
	}
	names := map[string]bool{}
	for _, o := range outs {
		names[o.Name] = true
	}
	if !names[labelName] || !names[probsName] {
		return fmt.Errorf("model must have outputs %q and %q, got %v", labelName, probsName, outs)
	}
	return nil
}

func (p *ONNXPredictor) Metadata() Metadata { return p.meta }

func (p *ONNXPredictor) Predict(ctx context.Context, batch [][]float32) ([]Prediction, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	n := len(batch)
	flat := make([]float32, 0, n*p.nFeature)
	for i, row := range batch {
		if len(row) != p.nFeature {
			return nil, fmt.Errorf("row %d: expected %d features, got %d", i, p.nFeature, len(row))
		}
		flat = append(flat, row...)
	}

	in, err := ort.NewTensor(ort.NewShape(int64(n), int64(p.nFeature)), flat)
	if err != nil {
		return nil, fmt.Errorf("input tensor: %w", err)
	}
	defer in.Destroy()

	outputs := []ort.Value{nil, nil} // ORT выделит сам
	if err := p.session.Run([]ort.Value{in}, outputs); err != nil {
		return nil, fmt.Errorf("run: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	labels, ok := outputs[0].(*ort.Tensor[int64])
	if !ok {
		return nil, fmt.Errorf("unexpected label output type %T", outputs[0])
	}
	probs, ok := outputs[1].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("unexpected probabilities output type %T", outputs[1])
	}

	ld, pd := labels.GetData(), probs.GetData()
	nClass := len(pd) / n
	res := make([]Prediction, n)
	for i := range n {
		pr := make([]float32, nClass)
		copy(pr, pd[i*nClass:(i+1)*nClass]) // копия: данные тензора освобождаются в defer
		res[i] = Prediction{Label: ld[i], Class: p.meta.ClassName(ld[i]), Probabilities: pr}
	}
	return res, nil
}

func (p *ONNXPredictor) Close() error { return p.session.Destroy() }
