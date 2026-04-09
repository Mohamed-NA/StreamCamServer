Runtime model artifacts are not committed directly.

Place the unpacked runtime files here for local development, or install a release bundle with:

```bash
uv run scripts/install_model_bundle.py dist/model-bundle.tar.gz
```

To create a release bundle from locally exported runtime models:

```bash
uv run scripts/package_model_bundle.py
```
