import numpy as np
from vive3D.util import tensor_to_image
from .SpectreUser import SpectreUser
import copy
from typing import Type, Dict, List
from abc import ABC, abstractmethod
import torch
from .KeyframeSelector import (
    InitialInversionSelector,
)
from configs.config import *
import random

class State(ABC):
    VALID_STATES = [
        "VideoAnalysed",
        "CGSGANPosesAnalysed",
        "SpectreAnalysed",
        "KeyframesAnalysed",
        "CollectivelyPersonAnalysed",
        "CollectivelyKeyframesInverted",
        "CollectivelyModelsTuned",
    ]

    @property
    @abstractmethod
    def META_INFO() -> dict:
        pass

    def __init__(self, video_model= None) -> None:
        super().__init__()
        self.video_model= video_model
        self.order: int = self.VALID_STATES.index(self.__class__.__name__)

    @staticmethod
    def get_min_state(video_model= None) -> Type["State"]:
        min_state = State.VALID_STATES[0]
        min_state_class = globals()[min_state]
        return min_state_class(video_model)

    @staticmethod
    def get_max_state(
        video_model= None, multiple_videos: bool = False
    ) -> Type["State"]:
        if not multiple_videos:
            return State.get_max_individual_state(video_model)

        max_state = State.VALID_STATES[-1]
        max_state_class = globals()[max_state]
        return max_state_class(video_model)

    @staticmethod
    def get_max_individual_state(video_model= None) -> Type["State"]:
        filtered_states = [
            globals()[s]
            for s in State.VALID_STATES
            if globals()[s].META_INFO["is_individual"]
        ]
        max_state = filtered_states[-1]
        return max_state(video_model)

    @staticmethod
    def get_state(name: str, video_model= None) -> Type["State"]:
        assert name in State.VALID_STATES
        state_class = globals()[name]
        return state_class(video_model)

    def __eq__(self, value: Type["State"], /) -> bool:
        return self.order == value.order

    def __lt__(self, value: Type["State"]) -> bool:
        return self.order < value.order

    def __le__(self, value: Type["State"]) -> bool:
        return self.order <= value.order

    def __gt__(self, value: Type["State"]) -> bool:
        return self.order > value.order

    def __ge__(self, value: Type["State"]) -> bool:
        return self.order >= value.order

    def __iadd__(self, other: int) -> Type["State"]:
        new_order: int = self.order + other
        assert new_order < len(State.VALID_STATES), "State Order out of Range"
        assert new_order >= 0, "State Order out of Range"
        new_state_class = State.VALID_STATES[new_order]
        new_state_class = globals()[new_state_class]
        return new_state_class(self.video_model)

    def __isub__(self, other: int) -> Type["State"]:
        new_order: int = self.order - other
        assert new_order < len(State.VALID_STATES), "State Order out of Range"
        assert new_order >= 0, "State Order out of Range"
        new_state_class = State.VALID_STATES[new_order]
        new_state_class = globals()[new_state_class]
        return new_state_class(self.video_model)

    def __repr__(self) -> str:
        return State.VALID_STATES[self.order]

    @abstractmethod
    def fullfills_state_requirements(self) -> bool:
        pass

    @abstractmethod
    def upgrade(self, opts: Dict):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def replace(self, other):
        pass


class VideoAnalysed(State):

    META_INFO = {"is_individual": True, "is_tuned": False}

    def fullfills_state_requirements(self) -> bool:
        return True

    def upgrade(self, opts):
        return opts

    def reset(self):
        raise TypeError("Cannot reset any furhter")

    def replace(self, other):
        other_params = other.get_dict()["video_dict"]
        self.video_model.get_dict()["video_dict"] = other_params


class CollectivelyPersonAnalysed(State):
    META_INFO = {"is_individual": False, "is_tuned": False}

    def create_selection(self, opts: Dict):
        if isinstance(opts["selection"], List):
            selection = opts["selection"]
        elif opts["selection"] == "select":
            selection = []
            for video_model in self.video_model.video_models:
                landmarks = video_model.get_dict()["video_dict"]["landmarks"]
                selector = InitialInversionSelector(landmarks)
                sel = selector.select()
                sel.sort()
                selection.append(sel)

        else:
            assert opts["selection"] == "random"
            num_frames = len(
                self.video_model.video_models[0].get_dict()["video_dict"]["frames"]
            )
            sel = random.sample(range(num_frames), 4)
            sel.sort
            selection = [sel for _ in self.video_model.video_models]
        return selection

    def upgrade(self, opts: Dict):
        opts = opts["analyse_person"]
        selection = self.create_selection(opts)
        self.video_model.init_inversion_pipeline()
        w_person, w_offsets = self.video_model.pipeline.get_average_person(
            len(self.video_model.video_models), len(selection[0])
        )
        input = {"w_person": [], "w_offsets": [], "poses": [], "keyframes": []}
        target = {
            "images": [],
            "face_segmentation": [],
            "landmarks": [],
            "face": [],
            "num_face_px": [],
        }
        for i, v in enumerate(self.video_model.video_models):
            state_dict = v.get_dict()
            keyframes = selection[i]
            face_segmentations = v.segmenter.get_eyes_mouth_BiSeNet(
                state_dict["video_dict"]["face_tensors"][keyframes].to(v.device),
                dilate=8,
            ).any(dim=0)
            images = (
                state_dict["video_dict"]["face_tensors"][keyframes]
                .clone()
                .detach()
                .to(v.device)
            )
            input["keyframes"].append(keyframes)
            input["w_person"].append(w_person[i])
            input["w_offsets"].append(w_offsets[i])
            input["poses"].append(v.get_dict()["additional_information"]["poses"])
            target["face_segmentation"].append(face_segmentations)
            target["images"].append(images)
            target["face"].append(images * face_segmentations)
            target["num_face_px"].append(face_segmentations.sum())
            target["landmarks"].append(state_dict["video_dict"]["landmarks"][keyframes])

        w_person, w_offsets, _ = self.video_model.pipeline(input, target, opts)
        for i, v in enumerate(self.video_model.video_models):
            w_p = w_person[i].clone().detach()
            if w_p.dim() == 3:
                w_p = w_p.squeeze(dim=0)
            v.get_dict()["person_codes"] = {
                "w_person": w_p,
                "w_offsets": w_offsets[i].clone().detach(),
                "selected_face_tensors": images.clone().detach(),
            }
            v.get_dict()["additional_information"]["initial_selection"] = selection
            v.get_dict()["original_person_codes"] = copy.deepcopy(
                v.get_dict()["person_codes"]
            )

    def fullfills_state_requirements(self) -> bool:
        for v in self.video_model.video_models:
            v_dict = v.get_dict()
            selection = (
                "initial_selection" not in v_dict["additional_information"]
                or v_dict["additional_information"]["initial_selection"] is not None
            )
            if not (v_dict["person_codes"] is not None and selection):
                return False
        return True

    def replace(self, other):
        from .VideoModelObject import MultipleVideoModelObject
        assert isinstance(other, MultipleVideoModelObject)
        other: MultipleVideoModelObject = other
        for v_self, v_other in zip(self.video_model.video_models, other.video_models):
            own_dict = v_self.get_dict()
            other_dict = v_other.get_dict()
            own_dict["person_codes"] = other_dict["person_codes"]
            own_dict["original_person_codes"] = other_dict["original_person_codes"]
            own_dict["additional_information"]["initial_selection"] = other_dict[
                "additional_information"
            ]["initial_selection"]

    def reset(self):
        for v in self.video_model.video_models:
            v_dict = v.get_dict()
            v_dict["person_codes"] = None
            v_dict["original_person_codes"] = None
            v_dict["additional_information"]["initial_selection"] = None


class CGSGANPosesAnalysed(State):
    META_INFO = {"is_individual": True, "is_tuned": False}

    def fullfills_state_requirements(self) -> bool:
        return (
            self.video_model.has_additional_information()
            and "poses" in self.video_model.get_dict()["additional_information"].keys()
            and self.video_model.get_dict()["additional_information"]["poses"]
            is not None
        )

    def reset(self):
        self.video_model.get_dict()["additional_information"]["poses"] = None
        self.video_model.get_dict()["additional_information"][
            "inverse_transformation"
        ] = None

    def replace(self, other):
        self.video_model.get_dict()["additional_information"][
            "poses"
        ] = other.get_dict()["additional_information"]["poses"]
        self.video_model.get_dict()["additional_information"][
            "inverse_transformation"
        ] = other.get_dict()["additional_information"]["inverse_transformation"]
        self.video_model.get_dict()["video_dict"]["frames"] = other.get_dict()[
            "video_dict"
        ]["frames"]
        self.video_model.get_dict()["video_dict"]["landmarks"] = other.get_dict()[
            "video_dict"
        ]["landmarks"]

    @staticmethod
    def format_tensors(images):
        face_tensors = torch.stack(images)
        face_tensors = face_tensors.permute(0, 3, 1, 2)
        return face_tensors

    def format_landmarks(self, landmarks):
        return np.stack(landmarks)

    def format_poses(self, poses):
        new_poses = []
        for pose in poses:
            extr = pose[:16].reshape((4, 4))
            intr = pose[16:].reshape((3, 3))
            new_poses.append(
                {
                    "extrinsics": extr,
                    "intrinsics": intr,
                }
            )
        return new_poses

    def format_frames(self, image_tensors):
        images = tensor_to_image(image_tensors)
        num_images = images.shape[0]
        images = [images[i] for i in range(num_images)]
        return images

    def upgrade(self, opts: Dict = None):
        opts = opts["analyse_poses"]
        images = self.video_model.get_dict()["video_dict"]["frames"]
        from preprocess_cgsgan.Preprocess import Preprocessor

        processor = Preprocessor()
        masks, cams, images, landmarks = processor(
            images, target_size=opts["output_size"]
        )
        images = processor.mask_all_images(images, masks)
        face_tensors = self.format_tensors(images)
        self.video_model.get_dict()["video_dict"]["frames"] = self.format_frames(
            face_tensors
        )
        self.video_model.get_dict()["video_dict"]["face_tensors"] = face_tensors
        self.video_model.get_dict()["video_dict"]["landmarks"] = self.format_landmarks(
            landmarks
        )
        self.video_model.get_dict()["additional_information"]["poses"] = (
            self.format_poses(cams)
        )


class KeyframesAnalysed(State):
    META_INFO = {"is_individual": True, "is_tuned": False}

    def fullfills_state_requirements(self) -> bool:
        return self.video_model.get_dict()["keyframes"] is not None

    def upgrade(self, opts: Dict):
        exists_ok: bool = False
        num_total_frames = self.video_model.get_dict()["video_dict"][
            "face_tensors"
        ].shape[0]
        opts = opts["analyse_keyframes"]
        assert (
            opts["num_keyframes"] <= num_total_frames
        ), "Cannot find more Keyframes than Frames in the video"
        if opts["num_keyframes"] < 1:
            opts["num_keyframes"] = int(opts["num_keyframes"] * num_total_frames)
        if self.video_model.keyframe_selector_is_none() or not exists_ok:
            self.video_model.set_keyframe_selector(**opts)
        keyframe_selector = self.video_model.get_keyframe_selector()
        keyframes = keyframe_selector.select(self.video_model.get_dict())
        self.video_model.get_dict()["keyframes"] = keyframes
        debug_info = keyframe_selector.get_additional_information()

        additional_info = self.video_model.get_dict()["additional_information"]
        self.video_model.get_dict()["additional_information"] = {
            **(additional_info or {}),
            **(debug_info or {}),
        }

    def reset(self):
        self.video_model.get_dict()["keyframes"] = None

    def replace(self, other):
        self.video_model.get_dict()["keyframes"] = other.get_dict()["keyframes"]


class CollectivelyKeyframesInverted(State):
    META_INFO = {"is_individual": False, "is_tuned": False}

    def __init__(self, video_model= None) -> None:
        super().__init__(video_model)
        from .VideoModelObject import MultipleVideoModelObject
        assert video_model is None or isinstance(video_model, MultipleVideoModelObject)
        self.video_model: MultipleVideoModelObject = self.video_model

    def fullfills_state_requirements(self) -> bool:
        for v in self.video_model.video_models:
            info = v.get_dict()["additional_information"]
            if "inverted_w" not in info.keys() or info["inverted_w"] is None:
                return False

        return True

    def reset(self):
        for v in self.video_model.video_models:
            v.get_dict()["additional_information"]["inverted_w"] = None

    def upgrade(self, opts: Dict):
        self.video_model.init_inversion_pipeline()
        opts = (
            opts["inversion"]
            if "inversion" in opts.keys()
            else opts["tune_model"]["inversion"]
        )
        input = {"w_person": [], "w_offsets": [], "poses": [], "keyframes": []}
        target = {
            "images": [],
            "face_segmentation": [],
            "landmarks": [],
            "face": [],
            "num_face_px": [],
        }
        for v in self.video_model.video_models:
            state_dict = v.get_dict()
            keyframes = state_dict["keyframes"]
            face_segmentations = v.segmenter.get_eyes_mouth_BiSeNet(
                state_dict["person_codes"]["selected_face_tensors"].to(v.device),
                dilate=8,
            ).any(dim=0)
            images = (
                state_dict["video_dict"]["face_tensors"][keyframes]
                .clone()
                .detach()
                .to(v.device)
            )

            input["keyframes"].append(keyframes)
            w_p = state_dict["person_codes"]["w_person"]
            if w_p.dim() == 3:
                w_p = w_p.squeeze(dim=0)
            input["w_person"].append(w_p)
            input["w_offsets"].append(state_dict["person_codes"]["w_offsets"])
            input["poses"].append(state_dict["additional_information"]["poses"])
            target["face_segmentation"].append(face_segmentations)
            target["images"].append(images)
            target["face"].append(images * face_segmentations)
            target["num_face_px"].append(face_segmentations.sum())
            target["landmarks"].append(state_dict["video_dict"]["landmarks"][keyframes])

        w_person, w_offsets, poses = self.video_model.pipeline(input, target, opts)
        for i, v in enumerate(self.video_model.video_models):
            v.get_dict()["additional_information"]["inverted_w"] = w_offsets[i]
            v.get_dict()["additional_information"]["poses"] = poses[i]
            v.get_dict()["person_codes"]["w_person"] = w_person[i]

    def replace(self, other):
        assert isinstance(other, MultipleVideoModelObject) # type: ignore
        other: MultipleVideoModelObject = other # type: ignore
        for v_self, v_other in zip(self.video_model.video_models, other.video_models):
            v_self.get_dict()["additional_information"][
                "inverted_w"
            ] = v_other.get_dict()["additional_information"]["inverted_w"]

class CollectivelyModelsTuned(State):
    META_INFO = {"is_individual": False, "is_tuned": True}

    def __init__(self, video_model= None) -> None:
        super().__init__(video_model)
        from .VideoModelObject import MultipleVideoModelObject
        assert video_model is None or isinstance(video_model, MultipleVideoModelObject)
        self.video_model= self.video_model

    def fullfills_state_requirements(self) -> bool:
        state_dict = self.video_model.get_dict()
        # len_fitting = len(state_dict["person_codes"]["selected_face_tensors"]) == len(
        #     state_dict["keyframes"]
        # )
        return (
            self.video_model.video_models[0].get_dict()["model"] is not None
            # and len_fitting
        )

    def upgrade(self, opts: Dict):
        self.video_model.init_pipeline()
        opts = opts["tune_model"]
        input = {"w_person": [], "w_offsets": [], "keyframes": [], "poses": []}
        target = {"images": []}
        hyperparameters = opts

        for v in self.video_model.video_models:
            v_dict = v.get_dict()
            keyframes = v_dict["keyframes"]
            input["keyframes"].append(keyframes)
            input["poses"].append(v_dict["additional_information"]["poses"])
            target["images"].append(v_dict["video_dict"]["face_tensors"])
            w_person = v_dict["person_codes"]["w_person"]
            if len(w_person.shape) == 3:
                w_person = w_person.squeeze(dim=0)
            input["w_person"].append(w_person)
            input["w_offsets"].append(v_dict["additional_information"]["inverted_w"])

        tuned_offsets, tuned_model = self.video_model.pipeline(input, target, hyperparameters)
        self.video_model.set_generator(tuned_model)

        for idx, v in enumerate(self.video_model.video_models):
            d = v.get_dict()
            d["model"] = tuned_model
            keyframes = d["keyframes"]
            # new_selected_face_tensors = d["video_dict"]["face_tensors"][
            #     keyframes, :, :, :
            # ]

            d["person_codes"] = self.video_model.to_person_codes(
                d["person_codes"]["w_person"],
                w_offsets=tuned_offsets[idx],
                yaws=None,
                pitches=None,
                selected_face_tensors=None,
            )

    def reset(self):
        self.video_model.get_dict()["model"] = None
        self.reset_person_codes()

    def reset_person_codes(
        self,
    ):
        for video_model in self.video_model.video_models:
            d = video_model.get_dict()
            d["person_codes"] = d["original_person_codes"]
            d["model"] = None
            if video_model.get_vive3d() is not None:
                video_model.get_vive3d().person_codes = d["person_codes"]

    def replace(self, other):
        self.video_model.get_dict()["model"] = other.get_dict()["model"]


class SpectreAnalysed(State):
    META_INFO = {"is_individual": True, "is_tuned": False}

    def __init__(self, video_model):
        State.__init__(self, video_model)

    def fullfills_state_requirements(self) -> bool:
        return (
            self.video_model.has_additional_information()
            and "spectre"
            in self.video_model.get_dict()["additional_information"].keys()
            and self.video_model.get_dict()["additional_information"]["spectre"]
            is not None
        )

    def upgrade(self, opts: Dict):
        self.spectre = SpectreUser()
        assert (
            "analyse_spectre" not in opts.keys() or not opts["analyse_spectre"].keys()
        )
        frames = self.video_model.get_dict()["video_dict"]["frames"]
        landmarks = self.video_model.get_dict()["video_dict"]["landmarks"]
        with torch.no_grad():
            spectre_dict = self.spectre.get_encoding(frames, landmarks, debug=False)
        self.video_model.get_dict()["additional_information"]["spectre"] = spectre_dict

    def reset(self):
        self.video_model.get_dict()["additional_information"]["spectre"] = None

    def replace(self, other):
        self.video_model.get_dict()["additional_information"][
            "spectre"
        ] = other.get_dict()["additional_information"]["spectre"]

