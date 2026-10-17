// Команда mreg — CLI для реестра моделей.
//
//	mreg push iris --dir models/iris       загрузить новую версию
//	mreg list                              список моделей
//	mreg list iris                         версии модели
//	mreg promote iris 2 --stage production сменить стадию версии
//	mreg pull iris production --out ./m    скачать версию
//
// Адрес и токен: флаги --url/--token или переменные MREG_URL/MREG_TOKEN.
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"os/user"
	"strconv"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	api "mlops/internal/registryapi"
	"mlops/internal/registryclient"
)

func main() {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	if err := rootCmd().ExecuteContext(ctx); err != nil {
		os.Exit(1)
	}
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func rootCmd() *cobra.Command {
	var baseURL, token string
	client := func() *registryclient.Client { return registryclient.New(baseURL, token, nil) }

	root := &cobra.Command{
		Use:          "mreg",
		Short:        "Клиент реестра моделей",
		SilenceUsage: true,
	}
	root.PersistentFlags().StringVar(&baseURL, "url", envOr("MREG_URL", "http://localhost:8090"), "адрес реестра (MREG_URL)")
	root.PersistentFlags().StringVar(&token, "token", os.Getenv("MREG_TOKEN"), "токен для записи (MREG_TOKEN)")

	root.AddCommand(pushCmd(client), listCmd(client), promoteCmd(client), pullCmd(client))
	return root
}

func pushCmd(client func() *registryclient.Client) *cobra.Command {
	var dir, author string
	cmd := &cobra.Command{
		Use:   "push <model> --dir <каталог>",
		Short: "Загрузить model.onnx и metadata.json как новую версию",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			v, err := client().Push(cmd.Context(), args[0], dir, author)
			if err != nil {
				return err
			}
			fmt.Printf("registered %s v%d  sha256=%s  size=%d bytes  stage=%s\n",
				v.Name, v.Version, v.SHA256[:12], v.SizeBytes, v.Stage)
			return nil
		},
	}
	defaultAuthor := "unknown"
	if u, err := user.Current(); err == nil {
		defaultAuthor = u.Username
	}
	cmd.Flags().StringVar(&dir, "dir", "", "каталог с model.onnx и metadata.json")
	cmd.Flags().StringVar(&author, "author", defaultAuthor, "кто загружает")
	_ = cmd.MarkFlagRequired("dir")
	return cmd
}

func listCmd(client func() *registryclient.Client) *cobra.Command {
	return &cobra.Command{
		Use:   "list [model]",
		Short: "Список моделей или версий модели",
		Args:  cobra.MaximumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
			defer tw.Flush()
			if len(args) == 0 {
				ms, err := client().ListModels(cmd.Context())
				if err != nil {
					return err
				}
				fmt.Fprintln(tw, "MODEL\tLATEST\tPRODUCTION\tCREATED")
				for _, m := range ms {
					prod := "-"
					if m.ProductionVersion != nil {
						prod = "v" + strconv.Itoa(*m.ProductionVersion)
					}
					fmt.Fprintf(tw, "%s\tv%d\t%s\t%s\n", m.Name, m.LatestVersion, prod, m.CreatedAt.Local().Format(time.DateTime))
				}
				return nil
			}
			vs, err := client().ListVersions(cmd.Context(), args[0])
			if err != nil {
				return err
			}
			fmt.Fprintln(tw, "VERSION\tSTAGE\tACCURACY\tF1\tSHA256\tAUTHOR\tCREATED")
			for _, v := range vs {
				fmt.Fprintf(tw, "v%d\t%s\t%s\t%s\t%s\t%s\t%s\n", v.Version, v.Stage,
					metric(v, "accuracy"), metric(v, "f1_macro"), v.SHA256[:12], v.CreatedBy,
					v.CreatedAt.Local().Format(time.DateTime))
			}
			return nil
		},
	}
}

func metric(v api.Version, key string) string {
	if m, ok := v.Metadata.Metrics[key]; ok {
		return strconv.FormatFloat(m, 'f', 4, 64)
	}
	return "-"
}

func promoteCmd(client func() *registryclient.Client) *cobra.Command {
	var stage string
	cmd := &cobra.Command{
		Use:   "promote <model> <version> --stage <stage>",
		Short: "Перевести версию в стадию none|staging|production|archived",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			version, err := strconv.Atoi(args[1])
			if err != nil {
				return fmt.Errorf("version must be a number: %w", err)
			}
			v, err := client().SetStage(cmd.Context(), args[0], version, api.Stage(stage))
			if err != nil {
				return err
			}
			fmt.Printf("%s v%d -> %s\n", v.Name, v.Version, v.Stage)
			return nil
		},
	}
	cmd.Flags().StringVar(&stage, "stage", string(api.StageProduction), "целевая стадия")
	return cmd
}

func pullCmd(client func() *registryclient.Client) *cobra.Command {
	var out string
	cmd := &cobra.Command{
		Use:   "pull <model> <version|stage> --out <каталог>",
		Short: "Скачать версию модели с проверкой SHA256",
		Args:  cobra.ExactArgs(2),
		RunE: func(cmd *cobra.Command, args []string) error {
			c := client()
			v, err := c.GetVersion(cmd.Context(), args[0], args[1])
			if err != nil {
				return err
			}
			if err := c.Download(cmd.Context(), v, out); err != nil {
				return err
			}
			fmt.Printf("downloaded %s v%d to %s (sha256 verified)\n", v.Name, v.Version, out)
			return nil
		},
	}
	cmd.Flags().StringVar(&out, "out", "", "куда скачать")
	_ = cmd.MarkFlagRequired("out")
	return cmd
}
