"""Regenerate docs/configuration.md from the pydantic models: python docs/gen_config_reference.py"""

from pathlib import Path

from pydantic import BaseModel

from hiprofile.config import Config
from hiprofile.simulate import SimConfig


def table(model: type[BaseModel], prefix: str = "", depth: int = 0) -> list[str]:
    rows = []
    for name, field in model.model_fields.items():
        ann = field.annotation
        if depth == 0 and isinstance(ann, type) and issubclass(ann, BaseModel) and model is Config:
            rows += table(ann, f"{prefix}{name}.", depth + 1)
            continue
        default = field.get_default(call_default_factory=True)
        if isinstance(default, BaseModel):
            default = default.model_dump(exclude_none=True)
        tname = str(ann) if "[" in str(ann) or "|" in str(ann) else getattr(ann, "__name__", str(ann))
        tname = tname.replace("typing.", "").replace("hiprofile.config.", "").replace("|", "\\|")
        desc = (field.description or "").replace("|", "\\|")
        rows.append(f"| `{prefix}{name}` | `{tname}` | `{default}` | {desc} |".replace("`None`", "`null`"))
    return rows


def main() -> None:
    head = ["| key | type | default | description |", "|---|---|---|---|"]
    text = [
        "# Configuration reference",
        "",
        "Generated from `hiprofile.config` by `docs/gen_config_reference.py`. Every key is optional.",
        "The validated configuration, the code version and the seed are written into every result file.",
        "",
        "## Fit configuration (`hiprofile fit --config`)",
        "",
        *head,
        *table(Config),
        "",
        "Prior ranges (`priors.v_ctr`, `priors.delta_v`, ...) take `kind` (`uniform`, `loguniform`,",
        "`normal`), `low`/`high` or `mean`/`std`.",
        "",
        "## Simulation configuration (`hiprofile simulate --config`)",
        "",
        *head,
        *table(SimConfig),
        "",
    ]
    Path(__file__).with_name("configuration.md").write_text("\n".join(text))


if __name__ == "__main__":
    main()
