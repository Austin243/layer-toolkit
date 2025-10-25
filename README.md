# Layer Toolkit

Layer Toolkit converts the original collection of single-file scripts into a reusable Python package for building layered VASP inputs and analysing electronic localisation function (ELF) and bonding characteristics.

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
  ```
- Analyse bonds in POSCAR-like files
  ```bash
  layer-toolkit analyze-bonds --input ./structures --pattern '*.vasp'
  ```
- Analyse ELF outputs (single file or directory)
  ```bash
  layer-toolkit analyze-elf --file ELFCAR
  layer-toolkit analyze-elf --directory ./ --prefix ELFCAR_
  ```

Each subcommand provides `--help` for detailed options.

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

## Legacy Script Compatibility

The original scripts (`2D_layers.py`, `bond_analysis.py`, `elf_analysis.py`, `max_ELF.py`) now act as thin wrappers around the new package for backwards compatibility.
