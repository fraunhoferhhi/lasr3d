from configs.config import PROJECT
import torch
from vive3D.segmenter import Segmenter
from .CGSGANGenerator import CGSGANGenerator
from datetime import datetime
import pickle
from torch import Tensor
from .KeyframeSelector import (
    DifferentialKeyframeSelector,
    EquidistantKeyframeSelector,
    LandmarkKeyframeSelector,
)
from vive3D.util import tensor_to_image, image_to_tensor
import numpy as np
import bz2
import cv2
import os
import pickle
from .Pipelines import (
    TuningPipeline,
    InversionPipeline,
)
from .State import CGSGANPosesAnalysed, State
from typing import Dict, List, Type, Union
from torch import Tensor
from vive3D.video_tool import VideoTool


"""
The VideoModelObject is a data structure to save a tuned model that was made to replicate a specific video, first tuned on selected tensors and then later on fine-tuned on specific keyframes. It saves all together at one place.
"""


class VideoModelObject:
    def __init__(
        self,
        video_dict: Dict,
        original_person_codes: Dict,
        keyframes: Union[List, None] = None,
        w_person: Union[Tensor, None] = None,
        w_offsets: Union[Tensor, None] = None,
        selected_face_tensors: Union[Tensor, None] = None,
        yaws: Union[Tensor, None] = None,
        pitches: Union[Tensor, None] = None,
        model=None,
        additional_information: Dict = None,
        name: str = "",
        verbose: bool = False,
        device: str = "cuda:0",
    ) -> None:
        self.device = device
        self._generator = None
        self.initialize_generator()

        if "face_tensors" not in video_dict.keys() or video_dict["face_tensors"] is None:
            frames = video_dict["frames"]
            face_tensors = [image_to_tensor(frame) for frame in frames]
            face_tensors = torch.stack(face_tensors,dim=0)
            video_dict["face_tensors"] = face_tensors

        person_codes = self.to_person_codes(
            w_person, w_offsets, yaws, pitches, selected_face_tensors
        )
        self.dict = {
            "video_dict": video_dict,
            "person_codes": person_codes,
            "original_person_codes": original_person_codes,
            "keyframes": keyframes,
            "model": model,
            "additional_information": additional_information,
        }
        if self.dict["additional_information"] is None:
            self.dict["additional_information"] = {}
        if name == "":
            now = datetime.now()
            now = now.strftime("%Y%m%d_%H%M")
            self.name = f"object_{now}"
        else:
            self.name = name
        self.verbose = verbose
        self.len_video = self.get_video_length(video_dict)
        # self.__vive3d = None
        self.__keyframe_selector = None
        self.segmenter = Segmenter(
            device=self.device, path=f"{PROJECT}/models/79999_iter.pth"
        )

    @staticmethod
    def read_video(source_path, start_sec, end_sec, fps):
        vid = VideoTool(source_path, None)
        frames = vid.extract_frames_from_video(start_sec, end_sec, 1 / fps)
        return {"frames": frames}

    def initialize_generator(self, model_type: str = "EG3D_Generator"):
        assert self._generator is None
        model_path = f"{PROJECT}/models/cgs-gan-512.pkl"
        self._generator = CGSGANGenerator(model_path, self.device)
        self._generator.set_camera_parameters(
            focal_length=3.8, cam_pivot=[0, 0.05, 0.2]
        )

    def get_generator(self):
        assert self._generator is not None
        return self._generator

    def set_generator(self, generator):
        self._generator = generator

    def has_additional_information(self) -> bool:
        return (
            "additional_information" in self.get_dict().keys()
            and self.get_dict()["additional_information"] is not None
        )

    def get_state(self) -> Type["State"]:
        state: State = State.get_min_state(self)
        while state < State.get_max_state():
            old_state = state
            state += 1
            if not state.fullfills_state_requirements():
                return old_state
        return state

    def upgrade_to_state(self, new_state: Type["State"], options: Dict):
        current_state = self.get_state()
        assert (
            current_state <= new_state
        ), "Cannot upgrade to a lower state, use reset_to_state()"
        while current_state < new_state:
            current_state += 1
            current_state.upgrade(options)
            if self.verbose:
                print(f"State upgraded to {current_state}")

    # hard: deletes everything that is in the states above, eventhough it might not even exist
    def reset_to_state(self, new_state: Type["State"], hard: bool = False):
        current_state = (
            State.get_max_state(self, multiple_videos=False)
            if hard
            else self.get_state()
        )
        assert new_state <= current_state, "Cannot reset to a higher state"

        while current_state > new_state:
            try:
                current_state.reset()
                current_state -= 1
            except:
                new_state = State.get_min_state(self)
            assert self.get_state() == current_state or hard
            if self.verbose:
                print(f"State downgraded to {current_state}")

    def move_to(self, device: str = "cuda:0"):
        self.dict = VideoModelObject._to(self.dict, device)

    @staticmethod
    def _to(object, device: str):
        if isinstance(object, torch.Tensor):
            return object.to(device)
        elif isinstance(object, dict):
            return {
                key: VideoModelObject._to(value, device)
                for key, value in object.items()
            }
        elif isinstance(object, (list, tuple)):
            return [VideoModelObject._to(item, device) for item in object]
        else:
            return object

    def set_keyframe_selector(self, selector_type: str, num_keyframes: int):
        assert (
            self.__keyframe_selector is None
        ), "You can only reference one Keyframe Selector"
        valid_types = ["equidistant", "differential", "landmark"]
        assert selector_type in valid_types, "Unknown Keyframe Selector Type"

        if selector_type == "equidistant":
            self.__keyframe_selector = EquidistantKeyframeSelector(num_keyframes)
        if selector_type == "landmark":
            self.__keyframe_selector = LandmarkKeyframeSelector(num_keyframes)
        if selector_type == "differential":
            self.__keyframe_selector = DifferentialKeyframeSelector(num_keyframes)

    def delete_keyframe_selector(self):
        assert not self.keyframe_selector_is_none()
        self.__keyframe_selector = None

    def keyframe_selector_is_none(self) -> bool:
        return self.__keyframe_selector is None

    def get_keyframe_selector(self):
        assert self.__keyframe_selector is not None, "Uninitialized Keyframe Selector"
        return self.__keyframe_selector

    @staticmethod
    def load_video_model(path: str, new_name: str = ""):
        if new_name != "":
            name = new_name
        else:
            name = os.path.splitext(os.path.basename(path))[0]

        with open(path, "rb") as file:
            loaded_file = pickle.load(file)

        if isinstance(loaded_file, dict):
            return VideoModelObject.initialize_from_dict(
                value_dict=loaded_file, name=name
            )
        else:
            multiple_video_model =  MultipleVideoModelObject.initialize_from_list(
                dict_list=loaded_file, name=name
            )
            for v in multiple_video_model.video_models[1:]:
                v.get_dict()["model"] = multiple_video_model.get_dict()["model"]
            return multiple_video_model

    def save_video_model(self, path: str):
        full_path = f"{path}/{self.name}.pkl"
        self.get_dict()["video_dict"]["face_tensors"] = None
        with open(full_path, "wb") as file:
            pickle.dump(self.dict, file)

    def override_person_anchor(self, new_person_offset: torch.Tensor):
        current_state = self.get_state()
        device = self.get_dict()["person_codes"]["w_person"].device
        new_person_offset = new_person_offset.to(device)
        assert current_state > State.get_state(
            "SpectreAnalysed"
        ), "not in the state to override person codes"
        self.get_dict()["person_codes"]["w_person"] = new_person_offset

    def override_pose(self, new_pose):
        current_state = self.get_state()
        device = self.get_dict()["additional_information"]["poses"].device
        new_pose = new_pose.to(device)
        assert (
            current_state >= CollectivelyPersonAnalysed()  # type: ignore
        ), "not in the state to override person codes"
        self.get_dict()["additional_information"]["poses"] = new_pose

    def to_person_codes(
        self, w_person, w_offsets, yaws, pitches, selected_face_tensors
    ):
        if all(
            [
                e is None
                for e in [w_person, w_offsets, yaws, pitches, selected_face_tensors]
            ]
        ):
            return None
        if len(w_person.shape) == 3 and w_person.shape[0] == 1:
            w_person = w_person.squeeze(dim=0)
        return {
            "w_person": w_person,
            "w_offsets": w_offsets,
            "yaws": yaws,
            "pitches": pitches,
            "selected_face_tensors": selected_face_tensors,
        }

    @classmethod
    def initialize_self(
        cls,
        video_dict,
        person_codes: Union[Dict, None] = None,
        original_person_codes: Dict = None,
        model=None,
        keyframes: List[int] = None,
        additional_information: Dict = None,
        name: str = "",
    ):
        if person_codes is not None:
            return cls(
                video_dict=video_dict,
                keyframes=keyframes,
                additional_information=additional_information,
                original_person_codes=original_person_codes,
                w_person=person_codes["w_person"],
                w_offsets=person_codes["w_offsets"],
                selected_face_tensors=person_codes["selected_face_tensors"],
                yaws=person_codes["yaws"] if "yaws" in person_codes.keys() else None,
                pitches=(
                    person_codes["pitches"] if "yaws" in person_codes.keys() else None
                ),
                model=model,
                name=name,
            )
        else:
            return cls(
                video_dict=video_dict,
                keyframes=keyframes,
                original_person_codes=original_person_codes,
                additional_information=additional_information,
                name=name,
            )

    @classmethod
    def initialize_from_dict(cls, value_dict: Dict, name: str = ""):
        return cls.initialize_self(**value_dict, name=name)

    @classmethod
    def initialize_from_video(cls, video_path, analyzer_options, name: str=""):
        video_dict = VideoModelObject.read_video(video_path, **analyzer_options)
        return cls.initialize_self(video_dict=video_dict, name=name)

    def get_dict(self):
        return self.dict

    def set_dict(self, value: dict):
        self.dict = value

    def delete_original_frames(self):
        self.get_dict()["video_dict"]["frames"] = []

    def replace_until(
        self, other: Type["VideoModelObject"], state: Type["State"]
    ) -> None:
        cur_state = State.get_min_state(self)
        while cur_state <= state:
            cur_state.replace(other)
            cur_state += 1

    #
    def get_video_length(self, video_dict: Dict) -> int:
        v_2 = None
        if isinstance(video_dict, list):
            video_dict = video_dict[0]
            v_2 = video_dict[1]
        frames = video_dict["face_tensors"]
        length = frames.shape[0]
        if v_2 is not None and len(v_2) != length:
            print("WARNING, Videos seem to have different lengths")
        return length


class MultipleVideoModelObject(VideoModelObject):
    def __init__(
        self,
        video_models: List[VideoModelObject],
        name: str = "",
        main_model_idx: int = 0,
    ) -> None:
        self.video_models: List[VideoModelObject] = video_models
        self.set_main_model(main_model_idx)
        self.pipeline = None
        self.name = name

    def set_main_model(self, index: int) -> None:
        assert index < len(self.video_models) and index >= 0, "Model index out of range"
        self._main_model_idx = index

    def get_main_model(self) -> int:
        return self._main_model_idx

    def init_pipeline(self):
        main_generator = self.get_generator()
        self.pipeline = TuningPipeline(
            video_model=self, generator=main_generator, device="cuda:0"
        )

    def init_inversion_pipeline(self):
        main_generator = self.get_generator()
        self.pipeline = InversionPipeline(
            video_model=self, generator=main_generator, device="cuda:0"
        )

    def get_state(self) -> State:
        min_seperate_state = State.get_max_state(
            video_model=self, multiple_videos=False
        )
        for video_model in self.video_models:
            curr_state = video_model.get_state()
            if curr_state < min_seperate_state:
                min_seperate_state = curr_state

        old_state = min_seperate_state
        if old_state < State.get_state("KeyframesAnalysed"):
            return old_state

        collective_state: State = State.get_max_state(
            video_model=self, multiple_videos=False
        )
        collective_state += 1

        while collective_state < State.get_max_state(multiple_videos=True):
            if not collective_state.fullfills_state_requirements():
                return old_state
            old_state = collective_state
            collective_state += 1

        return (
            collective_state
            if collective_state.fullfills_state_requirements()
            else old_state
        )

    def individual_state_change(
        self, new_state: State, options: Dict, downgrade_individuals_ok: bool
    ):
        for v in self.video_models:
            individual_state: State = v.get_state()
            if individual_state == new_state:
                continue

            if individual_state > new_state and downgrade_individuals_ok:
                v.reset_to_state(new_state, hard=True)
            else:
                v.upgrade_to_state(new_state, options)

    def upgrade_to_state(
        self, new_state: State, options: Dict, downgrade_individuals_ok: bool = True
    ):
        current_state = self.get_state()
        assert (
            current_state < new_state
        ), "Cannot upgrade to a lower state, use reset_to_state()"
        max_individual_state = State.get_max_individual_state(self)
        max_seperately_upgradable: State = min(new_state, max_individual_state)
        self.individual_state_change(
            max_seperately_upgradable, options, downgrade_individuals_ok
        )
        current_state = self.get_state()
        if new_state > current_state:
            current_state = max(max_individual_state, self.get_state())
        while new_state > current_state:
            current_state += 1
            current_state.upgrade(options)

    def reset_to_state(self, new_state: State, hard: bool = False):
        current_state = (
            State.get_max_state(self, multiple_videos=True)
            if hard
            else self.get_state()
        )
        assert new_state <= current_state, "Cannot reset to a higher state"

        while (
            current_state > new_state
            and current_state >= State.get_max_individual_state(self)
        ):
            current_state.reset()
            current_state -= 1
        if current_state > new_state:
            self.individual_state_change(new_state, None, downgrade_individuals_ok=True)

    def move_to(self, device: str = "cuda:0"):
        self.video_models = VideoModelObject._to(self.video_models, device)

    def get_generator(self):
        return self.video_models[self._main_model_idx].get_generator()

    def set_generator(self, generator):
        for v in self.video_models:
            v.set_generator(generator)

    @staticmethod
    def initialize_from_list(dict_list: List[dict], name: str):
        video_models = [VideoModelObject.initialize_from_dict(d) for d in dict_list]
        return MultipleVideoModelObject(video_models, name)

    def save_video_model(self, path: str):
        if path[-4:] != ".pkl":
            full_path = f"{path}/{self.name}.pkl"
        else:
            full_path = path
        os.makedirs(path, exist_ok=True)
        video_model_dicts = [v.get_dict() for v in self.video_models]
        for d in video_model_dicts:
            d["video_dict"]["face_tensors"] = None

        with open(full_path, "wb") as file:
            pickle.dump(video_model_dicts, file)

    def get_dict(self):
        return self.video_models[self._main_model_idx].dict

    def save_frames_for_external(self, path: str):
        for i, v in enumerate(self.video_models):
            os.makedirs(f"{path}/{i}", exist_ok=True)
            frames = v.get_dict()["video_dict"]["frames"]
            max_number = len(str(len(frames)))
            for j, image in enumerate(frames):
                cv2.imwrite(f"{path}/{i}/img_{str(j).zfill(max_number)}.png", image)

    def delete_original_frames(self):
        for v in self.video_models:
            v.delete_original_frames()

    def replace_until(
        self, other: Type["MultipleVideoModelObject"], state: State
    ) -> None:
        max_individual_state = State.get_max_individual_state()
        end_state = min(max_individual_state, state)

        for v_self, v_other in zip(self.video_models, other.video_models):
            v_self.replace_until(v_other, end_state)

        if end_state == state:
            return

        while end_state < state:
            end_state += 1
            end_state.replace(self, other)

    def save_neutral_faces(self, save_path: str, name: str = "neutral_face"):
        g = self.get_generator()
        pose = g.get_neutral_camera()
        code_0 = (
            self.video_models[0].get_dict()["person_codes"]["w_person"].unsqueeze(dim=0)
        )
        code_1 = (
            self.video_models[1].get_dict()["person_codes"]["w_person"].unsqueeze(dim=0)
        )
        source_image = g.generate(w=code_0, camera_params=pose)
        target_image = g.generate(w=code_1, camera_params=pose)
        source_debug_img = tensor_to_image(source_image)
        source_debug_img = cv2.cvtColor(source_debug_img, cv2.COLOR_BGR2RGB)

        target_debug_img = tensor_to_image(target_image)
        target_debug_img = cv2.cvtColor(target_debug_img, cv2.COLOR_BGR2RGB)

        grid_img = np.concatenate((source_debug_img, target_debug_img), axis=1)
        cv2.imwrite(
            f"{save_path}/{name}.png",
            grid_img,
        )
