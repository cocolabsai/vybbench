# VybBench

**Almost isn't good enough.** VybBench measures how well models maintain
design fidelity by comparing how effectively they generate specific
fonts and colors, and perform non-destructive edits.

Our dataset consists of 145 fonts, 100 colors, and 200 edits. We have each model
perform each task three times and score each generation individually.

We compare top image generation models:
ChatGPT Image (gpt-image-2.5-sunburst), Meta Muse, and Recraft v4.1. We also report
the results for [Vyb 1.0](https://cocolabs.ai/vyb).

Full Disclosure: VybBench was built by Coco Labs, the makers of Vyb. The full
methodology and rationale are in
[the launch post](https://cocolabs.ai/blog/introducing-vybbench).

This repo contains the code for scoring any model. The prompts and the results 
from our runs are available in the HuggingFace Dataset: 
(https://huggingface.co/datasets/cocolabs/vybbench). They are downloaded automatically.

## Reproduce

```
pip install -e .
vybbench score
```

## Add your model

We have made it easy for you to test any model simply by doing the following:

```python
from vybbench.model import Model, Start

class MyModel(Model):
    name = "mymodel"

    def generate(self, prompt: str, width: int, height: int) -> bytes: ...
    def edit(self, instruction: str, start: Start) -> bytes: ...
```

```
vybbench run --model mypkg.mymodel:MyModel
vybbench judge
vybbench score
```

Our judging and scoring is based on the returned PNGs. Starts contain both the starting
png and the starting svg. Use whichever is better for your model.

We are happy to include your model. Please create an issue with the model code.

## Citation

```bibtex
@misc{vybbench2026,
  title = {VybBench 1.0: A Design Fidelity Benchmark},
  author = {{Coco Labs}},
  year = {2026},
  url = {https://huggingface.co/datasets/cocolabs/vybbench}
}
```
