# syntax=docker/dockerfile:1
ARG GO_VERSION=1.26
ARG ORT_VERSION=1.29.0

# --- onnxruntime: готовая сборка с GitHub releases ---
FROM debian:bookworm-slim AS ort
ARG ORT_VERSION
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
RUN set -eux; \
    case "${TARGETARCH:-amd64}" in amd64) arch=x64 ;; arm64) arch=aarch64 ;; *) exit 1 ;; esac; \
    curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-${arch}-${ORT_VERSION}.tgz" \
      | tar -xz -C /opt; \
    mkdir -p /out && cp -L /opt/onnxruntime-linux-*/lib/libonnxruntime.so.${ORT_VERSION} /out/libonnxruntime.so

# --- сборка Go (CGO нужен для onnxruntime_go) ---
FROM golang:${GO_VERSION}-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
COPY --from=ort /out/libonnxruntime.so /usr/local/lib/libonnxruntime.so
ARG RUN_TESTS=true
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    if [ "$RUN_TESTS" = "true" ]; then \
      ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so go test -race -cover ./... ; \
    fi
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /out/inference ./cmd/inference

# --- финальный образ ---
# distroless/cc-debian12 не подошёл: ORT 1.29 падает в нём с SIGSEGV в CreateOrtEnv,
# хотя все зависимости из ldd присутствуют. На bookworm-slim работает стабильно.
FROM debian:bookworm-slim
RUN useradd --system --uid 65532 --no-create-home nonroot
COPY --from=ort /out/libonnxruntime.so /usr/local/lib/libonnxruntime.so
COPY --from=build /out/inference /inference
COPY models/iris /models/iris
ENV ORT_LIB_PATH=/usr/local/lib/libonnxruntime.so \
    MODEL_DIR=/models/iris \
    INFERENCE_ADDR=:8080
EXPOSE 8080
USER 65532:65532
ENTRYPOINT ["/inference"]
