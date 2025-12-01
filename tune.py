import json
import click
from configs.config import PROJECT, LOG
import sys

# import wandb
#
# if LOG:
#     wandb.login()

sys.path.append(f"{PROJECT}/external/VIVE3D")
from lib import KeyframeModelTuner


@click.command()
@click.option(
    "--settings",
    type=click.Path(exists=True),
    default=f"{PROJECT}/configs/tune.json",
    help="Path to JSON config file.",
)
@click.option(
    "--id1",
    type=click.Path(exists=True),
    default=f"{PROJECT}/example/001.mp4",
    help="Path to video of identity 1",
)
@click.option(
    "--id2",
    type=click.Path(exists=True),
    default=f"{PROJECT}/example/002.mp4",
    help="Path to video of identity 2",
)
@click.option(
    "--out", type=click.Path(), default=f"{PROJECT}/outputs/example", help="Output path"
)
def cli(settings, id1, id2, out):
    with open(settings) as f:
        settings = json.load(f)
    settings["load_path"] = [id1, id2]
    settings["save_path"] = out
    k = KeyframeModelTuner(settings)
    k()


if __name__ == "__main__":
    cli()
