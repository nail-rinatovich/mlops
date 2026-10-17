// Команда loadgen — простой нагрузочный тест для /v1/predict.
//
//	go run ./tools/loadgen -url http://localhost:8081 -c 32 -d 30s
//
// Печатает число запросов, статусы, ошибки, перцентили задержки и то,
// сколько ответов дала каждая версия модели (видно момент горячей замены).
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"sort"
	"sync"
	"time"
)

type result struct {
	status  int
	err     error
	latency time.Duration
	version string
}

func main() {
	url := flag.String("url", "http://localhost:8081", "адрес inference-сервиса")
	conc := flag.Int("c", 32, "число параллельных клиентов")
	dur := flag.Duration("d", 30*time.Second, "длительность теста")
	flag.Parse()

	body := []byte(`{"instances":[[5.1,3.5,1.4,0.2],[6.7,3.0,5.2,2.3]]}`)
	client := &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{MaxIdleConnsPerHost: *conc}}
	ctx, cancel := context.WithTimeout(context.Background(), *dur)
	defer cancel()

	results := make(chan result, 4096)
	var wg sync.WaitGroup
	for range *conc {
		wg.Go(func() {
			for ctx.Err() == nil {
				results <- shoot(client, *url+"/v1/predict", body)
			}
		})
	}
	go func() { wg.Wait(); close(results) }()

	var (
		latencies []time.Duration
		statuses  = map[int]int{}
		versions  = map[string]int{}
		errs      = map[string]int{}
		order     []string // версии в порядке первого появления
	)
	start := time.Now()
	for r := range results {
		if r.err != nil {
			errs[r.err.Error()]++
			continue
		}
		statuses[r.status]++
		latencies = append(latencies, r.latency)
		if r.version != "" {
			if versions[r.version] == 0 {
				order = append(order, r.version)
			}
			versions[r.version]++
		}
	}
	elapsed := time.Since(start)

	total := len(latencies)
	for _, n := range errs {
		total += n
	}
	slices.Sort(latencies)
	fmt.Printf("requests: %d in %s (%.0f rps), concurrency %d\n", total, elapsed.Round(time.Millisecond),
		float64(total)/elapsed.Seconds(), *conc)
	fmt.Printf("latency: p50=%s p95=%s p99=%s max=%s\n", pct(latencies, 50), pct(latencies, 95), pct(latencies, 99), pct(latencies, 100))
	codes := make([]int, 0, len(statuses))
	for c := range statuses {
		codes = append(codes, c)
	}
	sort.Ints(codes)
	for _, c := range codes {
		fmt.Printf("status %d: %d\n", c, statuses[c])
	}
	for _, v := range order {
		fmt.Printf("model version %s: %d responses\n", v, versions[v])
	}
	for e, n := range errs {
		fmt.Printf("transport error (%d): %s\n", n, e)
	}

	if len(errs) > 0 || statuses[http.StatusOK] != total {
		fmt.Println("RESULT: FAIL (есть не-200 ответы или ошибки соединения)")
		os.Exit(1)
	}
	fmt.Println("RESULT: OK (все ответы 200)")
}

func shoot(c *http.Client, url string, body []byte) result {
	t0 := time.Now()
	resp, err := c.Post(url, "application/json", bytes.NewReader(body))
	if err != nil {
		return result{err: err}
	}
	defer resp.Body.Close()
	var out struct {
		Model struct {
			Version string `json:"version"`
		} `json:"model"`
	}
	b, _ := io.ReadAll(resp.Body)
	_ = json.Unmarshal(b, &out)
	return result{status: resp.StatusCode, latency: time.Since(t0), version: out.Model.Version}
}

func pct(sorted []time.Duration, p int) time.Duration {
	if len(sorted) == 0 {
		return 0
	}
	i := (len(sorted)*p)/100 - 1
	return sorted[max(0, min(i, len(sorted)-1))].Round(10 * time.Microsecond)
}
