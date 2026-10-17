# syntax=docker/dockerfile:1
ARG GO_VERSION=1.26

FROM golang:${GO_VERSION}-bookworm AS build
WORKDIR /src
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download
COPY . .
# Реестру не нужен CGO, поэтому бинарник полностью статический.
RUN --mount=type=cache,target=/go/pkg/mod --mount=type=cache,target=/root/.cache/go-build \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/registry ./cmd/registry && \
    CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/mreg ./cmd/mreg

# distroless/static: без shell и libc, только наш бинарник. Здесь он работает,
# потому что в отличие от inference нет нативной библиотеки ONNX Runtime.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/registry /registry
COPY --from=build /out/mreg /mreg
EXPOSE 8090
ENTRYPOINT ["/registry"]
