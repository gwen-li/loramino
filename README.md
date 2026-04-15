# loramino

## Set up

```bash
conda create -n loramino python=3.10
conda activate loramino
pip install -r requirements.txt
```

## Usage

```bash
python main.py --config_file config.json --base_model pythia-14m --verbose
```

### Run tiny test on local machine

```bash
python main.py --config_file config_tiny.json --base_model pythia-14m --verbose
```