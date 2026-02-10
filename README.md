# Layer Toolkit

Layer Toolkit streamlines layered-material VASP workflows by turning common setup and post-processing steps into repeatable CLI operations.

It is designed for studies where you want to generate and compare multiple slab/layer-count variants quickly, while keeping prototype selection and analysis output consistent across runs.

Core workflows:
- Generate layered BCC/HCP inputs from Materials Project prototypes.
- Stage `relax/` and `scf/` directories with `POSCAR`, `INCAR`, `POTCAR`, and batch job scripts.
- Analyze bonding and ELF data, including top-N ELF hotspot locations and nearest-atom distances.

## Installation

```bash
pip install -e .
```

## Configuration

All external dependencies (POTCAR directory, VASP executable, scheduler settings, Materials Project API key) are configured via `config.json` in the project root. Copy `config.example.json` and fill in the relevant paths:

```bash
cp config.example.json config.json
```

You can also point the toolkit at another configuration file with the `LAYER_TOOLKIT_CONFIG` environment variable or the CLI `--config` option.

## Command Line Interface

The `layer-toolkit` console script exposes the main workflows:

- Generate layer inputs
  ```bash
  layer-toolkit --config config.json generate-layers --element Fe --structure bcc --layers 1 2 3
  layer-toolkit --config config.json generate-layers --element Fe --structure bcc --layers 2 4 --require-stable --max-energy-above-hull 0.02
  layer-toolkit --config config.json generate-layers --element Fe --structure bcc --layers 3 --material-id mp-13
  ```
- Analyze bonds in POSCAR-like files
  ```bash
  layer-toolkit analyze-bonds --input ./structures --pattern '*.vasp'
  ```
- Analyze ELF outputs (single file or directory)
  ```bash
  layer-toolkit analyze-elf --file ELFCAR --top-n 5
  layer-toolkit analyze-elf --directory ./ --prefix ELFCAR_ --top-n 5 --min-separation-frac 0.08 --hotspots-output elfcar_hotspots.dat
  ```

Each subcommand provides `--help` for detailed options.

Prototype-selection controls for `generate-layers`:
- `--material-id` targets a specific Materials Project entry directly.
- `--require-stable` restricts results to stable entries.
- `--max-energy-above-hull` limits candidates by eV/atom above hull.

Directory-mode outputs for `analyze-elf`:
- `elfcar_data.dat` (summary metrics)
- `elfcar_coords.dat` (max-ELF coordinates)
- `elfcar_hotspots.dat` (top-N hotspot table)

## Python API

The package exposes the same functionality for scripting:

```python
from layer_toolkit.config import load_settings
from layer_toolkit.generation.layers import LayerGenerator, LayerGenerationRequest

settings = load_settings("config.json")
request = LayerGenerationRequest(element="Fe", structure_type="bcc", layer_counts=[1, 2, 3])
LayerGenerator(settings).run(request)
```

Bond and ELF analysis helpers live under `layer_toolkit.analysis`.

## Features

- Generates BCC and HCP layered structure inputs from Materials Project prototypes with configurable vacuum spacing and automatic layer-count deduplication.
- Supports reproducible prototype selection with explicit `--material-id` targeting and optional Materials Project stability/energy-above-hull filters.
- Creates per-layer `relax/` and `scf/` work directories with ready-to-run `POSCAR`, `INCAR`, `POTCAR`, and batch job scripts.
- Uses configurable templates and centralized JSON settings for scheduler directives, executable paths, and environment-based overrides.
- Analyzes POSCAR-like files to report unique in-plane and interlayer bond statistics across unit, primitive, and 3x3x1 supercell views.
- Analyzes ELFCAR data in single-file (JSON) or batch mode, reporting max/average ELF values and top-N hotspot coordinates with nearest-atom distances.
