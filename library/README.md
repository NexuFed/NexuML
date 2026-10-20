# nexuml-library

Reusable layers, data sources, evaluation algorithms, and scenarios for the [NexuML](https://github.com/NexuFed/NexuML) framework.

Install the library directly:

```bash
uv pip install nexuml-library
```

Or install it through the framework convenience extra:

```bash
uv pip install "nexuml[library]"
```

The package advertises itself through the `nexuml.libraries` entry-point group, so installed components and scenarios are discovered automatically. Feature-specific dependencies are available through the `audio`, `data`, `pretrained`, `eval`, and `all` extras.
