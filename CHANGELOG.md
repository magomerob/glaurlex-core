# Changelog

## 0.1.0 (2026-11-08)

First public release. The library is the computational core of the GLAURLex web application,
extracted into its own package. The entries below record what changed against that code.

### Added

- `glaurlex.core.io`: Salamanca TXT parsing (`parse_salamanca`, `read_salamanca`,
  `parse_stimulus_map`, `detect_topic_codes`, `decode_bytes`). The level map is a parameter
  (`level_labels`) instead of being hard-coded.
- `graph.build_cooccurrence_graph`, `graph.partition_to_map`,
  `graph_compare.pairwise_similarity_table`, `groups.select_informant_rows`,
  `inference.qq_points`, `variables.split_variables`, `variables.is_numeric_like`.

### Changed

- Every identifier, column, result key and message is in English; "topic" replaces "tema".
  Main renames:

  | Before | Now |
  |---|---|
  | `estadisticas_df(df, informantes_df)` | `type_statistics(df, informant_ids=None)` |
  | columns `disponibilidad`, `aparición`, `freq_rel`, `freq_acum` | `availability`, `appearance`, `rel_freq`, `cum_freq` |
  | `informant_metrics` keys `total_disp`, `mean_disp` | `total_availability`, `mean_availability` |
  | `CODIGO_INFORMANTE`, `user_id` | `informant_id` (informants and topic frames) |
  | Salamanca columns `EXPERIMENTO_ID`, `NIVEL_CODIGO`, `NIVEL` | `experiment_id`, `level_code`, `level` |
  | Salamanca topic names `tema_<code>` | `topic_<code>` (`topic_prefix` parameter) |
  | `process_salamanca`, `detect_theme_codes` | `parse_salamanca`, `detect_topic_codes` |
  | `bigrams_for_tema` | `bigrams_for_topic` |
  | `describe_series` keys `media`, `mediana`, `rango`, `varianza`, `asimetría`, `curtosis` | `mean`, `median`, `range`, `variance`, `skewness`, `kurtosis` |
  | `PROFILE_METRICS` (dict of Spanish labels) `n_nodos`, `n_aristas`, `densidad`, `grado_prom`, `fuerza_prom`, `clustering_prom`, `diametro`, `long_camino_prom`, `componentes`, `frac_mayor_componente`, `asortatividad`, `n_comunidades`, `modularidad` | tuple of keys `n_nodes`, `n_edges`, `density`, `avg_degree`, `avg_strength`, `avg_clustering`, `diameter`, `avg_path_length`, `components`, `largest_component_fraction`, `assortativity`, `n_communities`, `modularity` |
  | `ONLY_A`, `ONLY_B`, `BOTH` = `"solo A"`, `"solo B"`, `"ambos"` | `"only_a"`, `"only_b"`, `"both"` |
  | permutation keys `observado`, `media_nula`, `sd_nula`, `nula`, `jaccard_nodos`, `jaccard_aristas` | `observed`, `null_mean`, `null_sd`, `null`, `jaccard_nodes`, `jaccard_edges` |

- Errors and notes in results are short codes (`insufficient_data`, `zero_path_length`, ...)
  instead of sentences.
- `small_world_indices` returns a `SmallWorld` dataclass and raises `ValueError` on a directed,
  disconnected or single-node graph.
- Long-running functions report progress only through `progress_cb(done, total)`; `status_cb` is
  removed. `graph_stats` no longer takes `include_small_world`; call `small_world_indices`.
- `apply_group` raises `ValueError` on an unknown operator or a malformed `{"op": ...}` rule.

### Fixed

- `apply_group` understands `{"op": "between", "min": a, "max": b}` and `{"op": ">", "value": v}`
  rules and the `<` and `>` operators. Before, such rules were ignored, so groups that used them
  silently contained every informant; they now filter.
- Rows with equal sort keys come out in the same order on every platform: `type_statistics` breaks
  availability ties by `type`, `edge_weight_divergence` by edge, and `pairwise_similarity_table`
  keeps pair order. Before, the order of tied rows, and with it their `cum_freq` in
  `type_statistics`, could depend on the CPU.
- Post-hoc comparisons in `compare_groups` test the right pairs of levels when a level has fewer
  than two observations.
- `decode_bytes` tries utf-8-sig, cp1252 and latin-1 in that order. Statistical encoding detection
  misread Spanish cp1252 files (`niño` became `ni±o`).
- Per-informant availability is summed in sorted order, so informants with the same responses in a
  different order tie exactly. This can change rank-test p-values (exact instead of asymptotic
  Mann-Whitney).
