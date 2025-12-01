import json
from torch import Tensor
from typing import List, Tuple, Union


class DictOperations:

    @staticmethod
    def add(dicts: List[dict]):
        ds = [DictOperations.flatten(d) for d in dicts]
        summed = {}
        for key in ds[0].keys():
            s = sum([d[key] for d in ds])
            summed[key] = s
        return DictOperations.unflatten(summed)

    @staticmethod
    def scale(d: dict, scalar: Union[int, float]):
        d = DictOperations.flatten(d)
        scaled = {}
        for key, value in d.items():
            scaled[key] = value * scalar
        return DictOperations.unflatten(scaled)

    @staticmethod
    def concatenate_values(dicts: List[dict]):
        ds = [DictOperations.flatten(d) for d in dicts]
        concatenated = {}
        for key in ds[0].keys():
            s = [d[key] for d in ds]
            concatenated[key] = s
        return DictOperations.unflatten(concatenated)

    @staticmethod
    def average(dicts):
        summed = DictOperations.add(dicts)
        avg = DictOperations.scale(summed, 1 / len(dicts))
        return avg

    @staticmethod
    def at(d: Tuple[dict, List], path: List[Tuple[str, int]]):
        for sub_key in path:
            d = d[sub_key]
        return d

    @staticmethod
    def insert(d: dict, path: List[Tuple[str, int]], value):
        p = path[:-1]
        key = path[-1]
        for sub_key in p:
            d.setdefault(sub_key, {})
            d = d[sub_key]
        d[key] = value

    @staticmethod
    def list_to_str_path(path: List[Tuple[int, str]]) -> str:
        return ".".join(path)

    @staticmethod
    def str_to_list_path(path: str) -> List[Tuple[int, str]]:
        if path[0] == ".":
            path = path[1:]
        list_path = path.split(".")
        for i, element in enumerate(list_path):
            try:
                int_variable = int(element)
                list_path[i] = int_variable
            except ValueError:
                continue
        return list_path

    @staticmethod
    def flatten(d: dict, running_key: str = "") -> dict:
        flattened = {}
        for key, value in d.items():
            new_running_key = running_key + f".{key}"
            if isinstance(value, dict):
                flattened.update(
                    DictOperations.flatten(value, running_key=running_key + f".{key}")
                )
            else:
                flattened[new_running_key] = value

        return flattened

    @staticmethod
    def unflatten(d: dict) -> dict:
        unflattened = {}
        for flat_key, value in d.items():
            path = DictOperations.str_to_list_path(flat_key)
            DictOperations.insert(unflattened, path, value)
        return unflattened

    @staticmethod
    def serialize(d: dict, round_floats=3) -> dict:
        if (isinstance(d, list) or isinstance(d, tuple)) and len(d) > 1:
            return [DictOperations.serialize(l, round_floats=round_floats) for l in d]
        elif isinstance(d, list) or isinstance(d, tuple):
            return DictOperations.serialize(d[0], round_floats=round_floats)
        elif isinstance(d, Tensor):
            return DictOperations.serialize(d.tolist(), round_floats=round_floats)
        elif isinstance(d, dict):
            return dict(
                [
                    (key, DictOperations.serialize(value, round_floats=round_floats))
                    for key, value in d.items()
                ]
            )
        elif isinstance(d, float) and round_floats > 0:
            return round(d, round_floats)
        else:
            return d

    @staticmethod
    def load(path) -> dict:
        with open(path, "r") as file:
            d = json.load(file)
        return d

    @staticmethod
    def save(d: dict, path: str, name: str = "dict", indent: bool = False):
        with open(f"{path}/{name}.json", "w") as file:
            if indent:
                json.dump(d, file, indent=4)
            else:
                json.dump(d, file)
