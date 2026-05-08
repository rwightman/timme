# AGENT.md

## Project Shape
- timme splits models into `ImageClassifier(encoder, head)`.
- SSL/task code should use `timme.create_encoder(...)` and bare `ImageEncoder`s.
- Architecture configs live in family `*_CFGS`; pretrained metadata currently comes from timm's registry.

## Build/Test Commands
- Install: `python -m pip install -e .`
- Run tests: `pytest tests/`
- Filter tests: `pytest -k "substring-to-match" tests/`

## Code Style Guidelines
- Line length: 120 chars
- Indentation: 4-space hanging indents, arguments should have an extra level of indent, use 'sadface' (closing parenthesis and colon on a separate line)
- Typing: Use PEP484 type annotations in function signatures
- Docstrings: Google style (do not duplicate type annotations and defaults)
- Imports: Standard library first, then third-party, then local
- Function naming: snake_case
- Class naming: PascalCase
- Error handling: Use try/except with specific exceptions
- Conditional expressions: Use parentheses for complex expressions

## Model Additions
- Add variants through the family config dict, `WeightLayout`, builder, and checkpoint filter path.
- For timm parity, compare timme outputs to timm `forward_features`.

## Training And Validation Apps
- Native apps use `train_{task}.py` / `eval_{task}.py`: `train_cls`, `eval_cls`, `train_ssl`, `eval_knn`.