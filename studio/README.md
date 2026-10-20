# NexuML Studio

A local, single-user workbench over your installed NexuML. Edit ordinary NexuML YAML,
run training in your selected Python environment, and inspect real results.

## Install and launch

Create or activate your project's Python environment, then run:

```sh
pip install "nexuml[api,library]"
npm install -g @nexufed/nexuml-studio
nexuml-studio .
```

If that environment is not on PATH, use
`nexuml-studio --python /path/to/project/.venv/bin/python .`.
Studio opens the browser; it never installs or synchronizes Python packages.
Keep the terminal running. Ctrl-C stops owned local processes; closing the browser does not.

## Try a CPU example

1. Click **Try an example** to load a one-epoch CPU draft with locally generated data.
   Loading calls trusted installed Python but does not start training.
2. Inspect **Pipeline** and **Training**; **Save** writes portable YAML.
3. Choose **Run… → Review selection → Run on Local**.
4. Open **Runs → Metrics / Logs / Artifacts / Configuration**. Reload and select the
   retained operation to inspect it without relaunching.

[Tutorial, runtime selection, troubleshooting and hosting](https://nexufed.github.io/NexuML/dev/studio/)
· [Manual API](https://nexufed.github.io/NexuML/dev/how-to/local-api/)
· [Contributor setup and checks](https://nexufed.github.io/NexuML/dev/development/install/)
