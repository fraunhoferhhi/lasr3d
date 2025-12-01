import sys
import random
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from typing import List, Dict, Union
from torch import Tensor
import numpy as np
from configs.config import *
from abc import ABC, abstractmethod
import bisect
import torch


class KeyFrameSelector(ABC):
    def __init__(self) -> None:
        super().__init__()
        self._additional_information: Dict = None

    @abstractmethod
    def select(self, attributes: Dict) -> List[int]:
        pass

    def get_additional_information(self) -> Dict:
        return self._additional_information

    # make sure the keyframes are sorted in ascending order
    @staticmethod
    def get_neighbours(
        keyframes: List[int], frame_id: Union[int, float]
    ) -> Dict[str, np.array]:
        if int(frame_id) == frame_id:
            frame_id = int(frame_id)
        right_neighbour_index = bisect.bisect_left(keyframes, frame_id)
        assert frame_id > -1 and right_neighbour_index < len(
            keyframes
        ), f"Frame {frame_id} in keyframes {keyframes}, right neighbour index: {right_neighbour_index}: No element found"

        if keyframes[right_neighbour_index] == frame_id:
            return {
                "neighbours": np.asarray([right_neighbour_index]),
                "weights": np.asarray([1.0]),
            }

        neighbours_indices = np.asarray(
            [right_neighbour_index - 1, right_neighbour_index], dtype=int
        )
        neighbours = np.asarray(
            [keyframes[right_neighbour_index - 1], keyframes[right_neighbour_index]],
            dtype=int,
        )
        distances = np.abs(neighbours - frame_id)
        distances = sum(distances) - distances
        weights = distances / sum(distances)

        return {"neighbours": neighbours_indices, "weights": weights}

    @staticmethod
    def get_parameters_from_neighbours(neighbours, data_tensor):
        neighbour_indices = neighbours["neighbours"]
        neighbour_weights = neighbours["weights"]
        neighbour_ws = data_tensor[neighbour_indices, ...]
        if len(neighbour_weights) == 1:
            neighbour_ws = neighbour_ws[0, ...] * neighbour_weights[0]
        else:
            neighbour_ws = (
                neighbour_ws[0, ...] * neighbour_weights[0]
                + neighbour_ws[1, ...] * neighbour_weights[1]
            )
        return neighbour_ws

    @staticmethod
    def get_w_offsets_and_poses_from_neighbours(
        neighbours, w_latent_offsets, pitches, yaws
    ):
        neighbour_indices = neighbours["neighbours"]
        neighbour_weights = neighbours["weights"]
        neighbour_ws = w_latent_offsets[neighbour_indices, :, :]
        neighbour_yaws = yaws[neighbour_indices]
        neighbour_pitches = pitches[neighbour_indices]
        if len(neighbour_weights) == 1:
            neighbour_ws = neighbour_ws[0, :, :] * neighbour_weights[0]
            neighbour_yaws = neighbour_yaws[0] * neighbour_weights[0]
            neighbour_pitches = neighbour_pitches[0] * neighbour_weights[0]
        else:
            neighbour_ws = (
                neighbour_ws[0, :, :] * neighbour_weights[0]
                + neighbour_ws[1, :, :] * neighbour_weights[1]
            )
            neighbour_yaws = (
                neighbour_yaws[0] * neighbour_weights[0]
                + neighbour_yaws[1] * neighbour_weights[1]
            )
            neighbour_pitches = (
                neighbour_pitches[0] * neighbour_weights[0]
                + neighbour_pitches[1] * neighbour_weights[1]
            )
        return neighbour_ws, neighbour_pitches, neighbour_yaws


class LandmarkKeyframeSelector(KeyFrameSelector):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

    def get_landmark_indices(self, region: str) -> None:
        assert region in ["mouth", "eye_left", "eye_right"]
        if region == "mouth":
            upper_index = 62
            lower_index = 66
        if region == "eye_left":
            upper_index = 37
            lower_index = 41
        elif region == "eye_right":
            upper_index = 43
            lower_index = 47
        return lower_index, upper_index

    def get_y_position(self, landmarks, region) -> None:
        up_id, low_id = self.get_landmark_indices(region=region)
        upper_y = landmarks[:, up_id, 1]
        lower_y = landmarks[:, low_id, 1]
        return lower_y, upper_y

    def get_opening(self, lower_y, upper_y):
        opening_distance = upper_y - lower_y
        interpolated_opening = np.interp(
            opening_distance, (opening_distance.min(), opening_distance.max()), (0, 1)
        )
        return interpolated_opening, opening_distance

    def select(self, attributes: Dict) -> List[int]:
        extreme_keyframes = self.get_extreme_keyframes(
            landmarks=attributes["video_dict"]["landmarks"]
        )
        indices = np.argsort(extreme_keyframes)[-self.num_frames :][::-1]
        indices = indices.tolist()
        indices = sorted(indices)
        indices[0] = 0
        indices[-1] = len(attributes["video_dict"]["frames"]) - 1
        return indices

    def get_landmark_opening_relation(self, landmarks: Tensor, region: str) -> np.array:
        lower, upper = self.get_y_position(landmarks, region)
        normalized_opening, _ = self.get_opening(lower, upper)
        return normalized_opening

    def get_landmark_opening_variance(self, landmarks: Tensor, region: str) -> np.array:
        normalized_opening = self.get_landmark_opening_relation(landmarks, region)
        opening_derivation = (0.5 - normalized_opening) ** 2
        return opening_derivation

    def get_extreme_keyframes(self, landmarks: Tensor) -> Tensor:
        mouth = self.get_landmark_opening_variance(landmarks, "mouth")
        eye_left = self.get_landmark_opening_variance(landmarks, "eye_left")
        eye_right = self.get_landmark_opening_variance(landmarks, "eye_right")

        return 2 * mouth + eye_left + eye_right


class EquidistantKeyframeSelector(KeyFrameSelector):
    def __init__(self, num_frames):
        super().__init__()
        self.num_frames = num_frames

    def select(self, attributes: Dict) -> List[int]:
        len_video = len(attributes["video_dict"]["frames"])
        indices = np.linspace(
            0, len_video - 1, self.num_frames, endpoint=True, dtype=int
        ).tolist()
        return indices


class DifferentialKeyframeSelector(LandmarkKeyframeSelector):
    def __init__(self, num_frames):
        super().__init__(num_frames)
        self.num_frames = num_frames
        self._additional_information = {}

    def calculate_gradient(self, landmarks):
        lower_mouth_y, upper_mouth_y = self.get_y_position(landmarks, "mouth")
        interpolated_mouth_opening, mouth_opening = self.get_opening(
            lower_mouth_y, upper_mouth_y
        )

        gradient = np.gradient(mouth_opening)
        gradient = np.abs(gradient)
        gradient = gradient / sum(gradient)

        self._additional_information["mouth_opening"] = interpolated_mouth_opening
        self._additional_information["mouth_opening_gradient"] = gradient

    def select(self, attributes: Dict) -> List[int]:
        landmarks = attributes["video_dict"]["landmarks"]
        last_index = len(landmarks) - 1
        self.calculate_gradient(landmarks)
        gradients = self._additional_information["mouth_opening_gradient"]
        threshold = 1 / self.num_frames
        accumulated_weight = 0
        keyframes = [0]
        for i, gradient in enumerate(gradients):
            accumulated_weight += gradient
            if accumulated_weight >= threshold:
                keyframes.append(i)
                accumulated_weight -= threshold

        keyframes = keyframes[: self.num_frames]
        keyframes[-1] = last_index
        other = random.sample(
            list(set(range(last_index)) - set(keyframes)),
            self.num_frames - len(keyframes),
        )
        keyframes += other
        return sorted(keyframes)


def test_get_neighbours():
    keyframe_selector: KeyFrameSelector = EquidistantKeyframeSelector(num_frames=10)
    frames = torch.zeros((104, 1))
    attributes = {"video_frames": {"frames": frames}}
    keyframes = keyframe_selector.select(attributes=attributes)

    x = keyframe_selector.get_neighbours(keyframes, 27)
    y = keyframe_selector.get_neighbours(keyframes, 60)
    z = keyframe_selector.get_neighbours(keyframes, 103)
    return x, y, z


class InitialInversionSelector(LandmarkKeyframeSelector):
    def __init__(self, landmarks):
        self.lm = landmarks
        self.num_frames = landmarks.shape[0]

    def select(self):
        mouth_opening = self.get_landmark_opening_relation(self.lm, "mouth")
        eye_left_opening = self.get_landmark_opening_relation(self.lm, "eye_left")
        eye_right_opening = self.get_landmark_opening_relation(self.lm, "eye_right")
        eye_opening = eye_left_opening + eye_right_opening
        frame_indices = [
            np.argmin(mouth_opening),
            np.argmax(mouth_opening),
            np.argmin(eye_opening),
            np.argmax(eye_opening),
        ]
        other = random.sample(
            list(set(range(self.num_frames)) - set(frame_indices)),
            5 - len(frame_indices),
        )
        return frame_indices + other
