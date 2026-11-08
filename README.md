# glaurlex

Computational core of GLAURLex for lexical availability studies (*disponibilidad léxica*):
availability indices, bigram co-occurrence graphs, community detection, structural graph
comparison and group inference against sociolinguistic variables.

## Installation

```bash
pip install glaurlex
```

Python 3.10 or later. Runtime dependencies: numpy, pandas, scipy, networkx, igraph, leidenalg.

## Data model

| Frame | Columns |
|---|---|
| Topic | `informant_id`, `pos` (0-indexed position in the informant's list), `type` |
| Informants | `informant_id` plus any sociolinguistic variables |

## Example

```python
import pandas as pd

from glaurlex.core.graph import build_cooccurrence_graph, community_leiden, node_stats
from glaurlex.core.inference import informant_metrics
from glaurlex.core.stats import type_statistics

topic = pd.DataFrame(
    {
        "informant_id": [1, 1, 1, 2, 2, 3, 3, 3],
        "pos": [0, 1, 2, 0, 1, 0, 1, 2],
        "type": ["dog", "cat", "horse", "cat", "dog", "dog", "horse", "cow"],
    }
)
informants = pd.DataFrame({"informant_id": [1, 2, 3], "sex": ["F", "M", "F"]})

type_statistics(topic)                      # availability, frequency, appearance per type
graph = build_cooccurrence_graph(topic, directed=False)
node_stats(graph)                           # degree, strength, centralities, components
community_leiden(graph)                     # Leiden communities, seed 42

informant_metrics(topic, informants)       # tokens, types, TTR, entropy, availability per informant
```

Salamanca-format TXT files are parsed with `glaurlex.core.io.read_salamanca`.

## Modules

| Module | Contents |
|---|---|
| `glaurlex.core.stats` | Per-type frequency, position and availability statistics |
| `glaurlex.core.graph` | Bigram graphs, node statistics, Leiden communities, small-world indices |
| `glaurlex.core.graph_compare` | Node/edge overlap, structural profiles, partition similarity, rarefaction and permutation tests |
| `glaurlex.core.inference` | Per-informant metrics, descriptive statistics, group comparisons, correlation |
| `glaurlex.core.groups` | Informant filtering rules |
| `glaurlex.core.variables` | Ordinal level configuration, variable classification |
| `glaurlex.core.metrics_catalog` | Metadata for every metric |
| `glaurlex.core.io` | Salamanca TXT parsing |

## Development

```bash
python -m pip install -e . --group dev
python -m pytest
python -m ruff check . && python -m ruff format --check .
python -m mypy
```

## Funding and authorship

Proyecto PID2022-137337NB-C21 financiado por MICIU/AEI/10.13039/501100011033 y por FEDER, UE.

Developed by Marcos Gómez Robres (mgrobres@gmail.com / magomerob@unirioja.es).

## Citation

TODO

## License

TODO
