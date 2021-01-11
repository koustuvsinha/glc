# GraphLogCore Generator (GLC)

Graph generator using simple first order logic.

Common graph generator pipeline for the following projects:

- [CLUTRR](https://github.com/facebookresearch/clutrr)
- [GraphLog](https://github.com/facebookresearch/graphlog)

## Features

- Use any arbitrary compositional rule base of the form (a,b) -> c
- Fast generation
- Logically validate the generated graphs using Proof traces
- Split graphs by unique "descriptors"
- Generate world graph on a given set of rules

## History

Over the years I have been developing FOL-inspired graph generation using several projects (CLUTRR, and most recently GraphLog, and several ongoing ones). Since these projects share a common ground - generation logic of the graphs - it makes no sense to develop them separately. Incorporating the knowledge learned from all these projects, and from many reviewer feedbacks, I'm consolidating the core graph generation logic into this one single location. This will potentially help users of all these projects, and will help me debug stuff faster.

## Requirements

```
pip install -r requirements.txt
```

## Usage

The file `graph_config.yaml` contains the parameters requires for graph generation. We use Hydra which allows us to also configure the parameters from command line.

The script currently expects a `rule_{world_id}.json` in the `save_loc` folder. I'll update it later to be more generic.

```
python glc.py save_loc=/scratch/koustuvs/clutrr_2.0 world_id=0 world_prefix=rule
```

## Things to do

- Currently the noise is a combination of dangling, disconnected and supporting noise. Need to split it in constituent parts.
- Documentation to support GLC development

## Test

```
python -m pytest tests
```

## License

"GLC" is CC-BY-NC 4.0 (Attr Non-Commercial Inter.) licensed, as found in the LICENSE file.