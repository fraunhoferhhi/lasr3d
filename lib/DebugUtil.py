import torch
from torch import Tensor
from configs.config import PROJECT, DEBUG
import cv2
from external.VIVE3D.vive3D.util import tensor_to_image


def save_debug_image(img, step=None, name="debug", from_tensor=True, insist_debug=True):
    if insist_debug and not DEBUG:
        return
    if step is None:
        full_name = name
    else:
        index = str(step).zfill(4)
        full_name = f"{name}_{index}"
    if from_tensor:
        image = tensor_to_image(img)
    else:
        image = img
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(
        f"{PROJECT}/outputs/debug/{full_name}.png",
        image,
    )


class DebugTool:

    last_cuda_values = {"reserved": 0, "alloc": 0}

    @staticmethod
    def c():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        diff_alloc = alloc - DebugTool.last_cuda_values["alloc"]
        diff_reserved = reserved - DebugTool.last_cuda_values["reserved"]
        DebugTool.last_cuda_values = {"alloc": alloc, "reserved": reserved}

        return f"alloc: {alloc:.2f}({diff_alloc:.2f}), res: {reserved:.2f}({diff_reserved:.2f})"

    def c_stat():
        alloc = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3

        return f"alloc: {alloc:.2f}, res: {reserved:.2f}"

    def sizeof(tensor: Tensor):
        sz = tensor.element_size() * tensor.nelement() / 1024 ** 3
        device = tensor.device
        return f"{sz:.2f}GB, {device}"
