import sys
from configs.config import PROJECT

sys.path.append(f"{PROJECT}/external/VIVE3D")
import json
from pathlib import Path
from lib import Editor
import click


@click.command()
@click.option(
    "--settings",
    type=click.Path(exists=True),
    default=f"{PROJECT}/configs/edit.json",
    help="Path to JSON config file.",
)
@click.option(
    "--model",
    type=click.Path(exists=True),
    default=f"{PROJECT}/outputs/example/model.pkl",
    help="Path to tuned model",
)
@click.option(
    "--edit",
    "-e",
    "edits",
    nargs=2,
    multiple=True,
    type=(str, float),
    help="Pair: --edit TYPE STRENGTH (repeatable).",
    default=[("glasses", +1.3), ("sentiment", +0.3)],
)
@click.option(
    "--out", type=click.Path(), default=f"{PROJECT}/outputs/example", help="Output path"
)
@click.option("--source", "-s", type=int, help="ID of the source identity", default=1)
@click.option("--target", "-t", type=int, help="ID of the target identity", default=0)
def cli(settings, model, out, edits, source, target):
    with open(settings) as f:
        settings = json.load(f)
    edits = list(edits)
    settings["load_path"] = model
    settings["save_path"] = out
    boundaries, names = extract_boundaries(settings["boundary_path"])
    settings["boundary_path"] = boundaries
    settings["target_axis"] = [(names.index(n), s) for n, s in edits]
    settings["main_model"] = source
    settings["target_model"] = target
    g = Editor(settings)
    g()


def extract_boundaries(path):
    p = Path(path)
    if not p.is_dir():
        raise ValueError(f"Not a directory: {path}")

    files = sorted(x for x in p.iterdir() if x.is_file() and x.suffix == ".npy")
    full_paths = [str(f.resolve()) for f in files]
    names_no_ext = [f.stem for f in files]
    return full_paths, names_no_ext


if __name__ == "__main__":
    cli()
