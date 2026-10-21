# Tutorials

Hands-on walkthroughs of PRISM on a scientific application, grouped by
application area. Each one walks a single configuration from environment and
data through training.

Tutorial pages live here, in `docs/tutorials/<area>/`. Notebooks live in
[`applications/<area>/`](../../applications/), next to the scripts they call.

For reference material on the same applications, such as pipeline designs and
evaluation invocations, see [`docs/applications/`](../applications/).

| Area | Tutorial | Modalities | Platform |
|---|---|---|---|
| Biology | [BioReason: train a DNA + language model](../../applications/bioreason/bioreason_demo.ipynb) (notebook): projector alignment, SFT, GRPO, and KEGG evaluation, submitted as chained PBS jobs | DNA + text | Aurora |
| Materials science | [Jointly train on material descriptions and crystal structures](material_science/materials_prism_tutorial.md): predict a material property from a CIF crystal graph and a text description | Crystal graph + text | CUDA, XPU, or CPU |
| Time series | [SciTS forecasting with Intern-S2 Preview 397B](timeseries/README.md): environment, data, training, and held-out evaluation of a quantile forecast head on a frozen OLMo-1B, with a companion [notebook](../../applications/timeseries/olmo_scits_forecast.ipynb) | Time series + text | Aurora |

## Adding a tutorial

- Put Markdown pages in `docs/tutorials/<area>/`, where `<area>` is the
  application's directory under `applications/`, or the name that directory
  would take if the application has no scripts there yet.
- Put a notebook in `applications/<area>/`, next to the scripts it runs, and
  link to it from this page.
- Add a row to the table above and to the Tutorials section of
  [`docs/index.md`](../index.md).
- Run `make docs` before pushing. It checks relative links in Markdown files
  the same way CI does. Links inside notebooks are not checked, so test those
  by hand after moving a notebook.
