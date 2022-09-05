# GraphLogCore Generator (GLC)

Graph generator using simple first order logic.

Common graph generator pipeline for the following projects:

- [CLUTRR](https://github.com/facebookresearch/clutrr)
- [GraphLog](https://github.com/facebookresearch/graphlog)

## History

Over the years I have been developing FOL-inspired graph generation using several projects (notably CLUTRR, and GraphLog). This project is built from the need to develop a common generation logic of the underlying graphs used in these projects, incorporating the lessons learned and suggestions from numerous reviewer feedbacks.

## Requirements

```
pip install -r requirements.txt
```

## Usage

The file `graph_config.yaml` contains the parameters requires for graph generation. We use Hydra which allows us to also configure the parameters from command line.

To use the rules defined in CLUTRR (Sinha et al. 2019), you can specify the path to the rules:

```
python glc.py rule_store=rule_bases/clutrr
```

A sample generator with CLUTRR graphs is provided in [graph_config.yaml](graph_config.yaml)

## Features

- Use any arbitrary compositional rule base of the form (a,b) -> c
- Fast generation
- Logically validate the generated graphs using Proof traces
- Split graphs by unique "descriptors"
- Generate world graph on a given set of rules



## Things to do

- Documentation to support GLC development

## Test

```
python -m pytest tests
```

## License

"GLC" is CC-BY-NC 4.0 (Attr Non-Commercial Inter.) licensed, as found in the LICENSE file.
