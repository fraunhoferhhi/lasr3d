import sys
from configs.config import PROJECT
from typing import Dict, Union, Tuple
import torch
import os
from typing import List
import cv2
import numpy as np
from vive3D.util import tensor_to_image, image_to_tensor
from vive3D.visualizer import Visualizer
from .VideoModelObject import VideoModelObject, MultipleVideoModelObject
from .State import State
from .KeyframeSelector import KeyFrameSelector


class KeyframeModelUser:
    def __init__(self, settings: Dict):
        self.settings = settings
        main_model_idx = (
            0 if "main_model" not in settings.keys() else settings["main_model"]
        )
        not_tuned_ok = (
            False if "not_tuned_ok" not in settings.keys() else settings["not_tuned_ok"]
        )
        self.video_model: VideoModelObject = self.init_video_model(
            not_tuned_ok=not_tuned_ok,
            model_type=settings["model_type"],
            path=settings["load_path"],
            main_model_idx=main_model_idx,
        )
        self.image_scale = None
        self.image_offset = None

    def transform_point_from_original_size(self, p: np.array) -> np.array:
        if self.image_scale is None or self.image_offset is None:
            return p
        p_new = p * self.image_scale[::-1]
        p_new = p_new + self.image_offset[::-1]
        return p_new

    @staticmethod
    def init_video_model(
        path: str,
        not_tuned_ok: bool = False,
        model_type: str = "EG3D_Generator",
        main_model_idx: int = 0,
    ) -> VideoModelObject:
        video_model: VideoModelObject = VideoModelObject.load_video_model(path=path)
        if isinstance(video_model, MultipleVideoModelObject):
            video_model.set_main_model(main_model_idx)
        video_model.move_to(device="cuda:0")
        loaded_state: State = video_model.get_state()
        assert (
            loaded_state.META_INFO["is_tuned"] or not_tuned_ok
        ), "Model isn't in the correct state to generate videos"
        return video_model

    def generate_image(self, frame_id, state, fixed_pose=False) -> None:
        ws, poses = self.get_generation_properties(frame_id, state)
        if fixed_pose:
            _, poses = self.get_generation_properties(0, state)
        generated_tensor = state["model"].generate(ws, poses, grad=False)
        image = tensor_to_image(generated_tensor)
        return image

    def generate_video(
        self, frames: List[Union[float, int]] = [], fixed_pose=False
    ) -> List[np.array]:
        state = self.video_model.get_dict()
        len_video = len(state["video_dict"]["frames"])
        if not frames:
            frames = range(len_video)
        images = []
        for frame_id in frames:
            image = self.generate_image(frame_id, state, fixed_pose=fixed_pose)
            images.append(image)
        return images

    def get_generation_properties(
        self, frame_id: int, state, return_seperated_ws: bool = False
    ) -> None:
        neighbours = KeyFrameSelector.get_neighbours(
            keyframes=state["keyframes"], frame_id=frame_id
        )
        person_codes = state["person_codes"]
        w_offset = KeyFrameSelector.get_parameters_from_neighbours(
            neighbours=neighbours,
            data_tensor=person_codes["w_offsets"],
        )
        w_offset = w_offset.to("cuda:0")
        ws = w_offset + person_codes["w_person"]
        ws = torch.unsqueeze(ws, dim=0)
        poses = state["additional_information"]["poses"]
        cam = KeyFrameSelector.get_parameters_from_neighbours(
            neighbours, poses
        ).unsqueeze(dim=0)
        if not return_seperated_ws:
            return ws, cam
        return person_codes["w_person"], w_offset, cam

    def add_landmarks(self, frames, selection=None):
        landmarks = self.video_model.get_dict()["video_dict"]["landmarks"]
        if selection is None:
            selection = range(68)
        new_frames = []
        for frame_idx, frame in enumerate(frames):
            image_with_point = np.copy(frame)
            for lmk_idx in selection:
                x, y = self.transform_point_from_original_size(
                    landmarks[frame_idx, lmk_idx, :]
                )
                cv2.circle(
                    image_with_point,
                    (int(x), int(y)),
                    radius=int(0.005 * frame.shape[0]),
                    color=[255, 0, 0],
                    thickness=-1,
                )
            new_frames.append(image_with_point)
        return new_frames

    def concat_videos_horizontally(
        self, video_0: List[np.ndarray], video_1: List[np.ndarray]
    ) -> List[np.ndarray]:
        concatenated = []
        shortest_frame_number = min(len(video_1), len(video_0))
        for frame_idx in range(shortest_frame_number):
            row = [video_0[frame_idx], video_1[frame_idx]]
            row = self.make_sizes_fit(row, [500, -1])
            row_image = cv2.hconcat(row)
            concatenated.append(row_image)
        return concatenated

    def generate_comparison_with_original(
        self, images: List[np.array], original_index=None
    ) -> List[np.array]:
        if original_index is None:
            state = self.video_model.get_dict()
        else:
            state = self.video_model.video_models[original_index].get_dict()
        original_images = state["video_dict"]["frames"]
        concatenated = self.concat_videos_horizontally(original_images, images)

        return concatenated

    def get_original(self, generated_images: List[np.array]) -> Tuple[torch.Tensor]:
        state = self.video_model.get_dict()
        original_images = state["video_dict"]["frames"]
        row_images = []
        shortest_frame_number = min(len(generated_images), len(original_images))

        for frame_idx in range(shortest_frame_number):
            row = [original_images[frame_idx], generated_images[frame_idx]]
            row = self.make_sizes_fit(row)
            tensor_row = [image_to_tensor(row[0]), image_to_tensor(row[1])]
            row_images.append(tensor_row)

        original_images = [pair[0] for pair in row_images]
        generated_images = [pair[1] for pair in row_images]
        original_images_array = torch.stack(original_images, dim=0)
        generated_images_array = torch.stack(generated_images, dim=0)
        return original_images_array, generated_images_array

    def get_original_numpy(
        self, generated_images: List[np.ndarray]
    ) -> Tuple[List[np.ndarray]]:
        state = self.video_model.get_dict()
        original_images = state["video_dict"]["frames"]
        row_images = []
        shortest_frame_number = min(len(generated_images), len(original_images))

        for frame_idx in range(shortest_frame_number):
            row = [original_images[frame_idx], generated_images[frame_idx]]
            row = self.make_sizes_fit(row)
            row_images.append(row)

        original_images = [pair[0] for pair in row_images]
        generated_images = [pair[1] for pair in row_images]
        return original_images, generated_images

    def highlight_keyframes(self, images: List[np.array]) -> List[np.array]:
        new_video = []
        keyframes = self.video_model.get_dict()["keyframes"]
        for i, image in enumerate(images):
            if i in keyframes:
                new_image = self.paint_margin(image)
            else:
                new_image = image
            new_video.append(new_image)
        return new_video

    def add_text_underneath(
        self,
        image: np.array,
        text: str,
        text_settings: Dict = {
            "text_color": (0, 0, 0),
            "font": cv2.FONT_HERSHEY_SIMPLEX,
        },
    ) -> np.array:
        h, w = image.shape[:2]
        white_field = np.zeros((40, w, 3))
        white_field[:, :] = (255, 255, 255)
        combined_image = np.concatenate((image, white_field), axis=0)
        position = (10, h + 30)
        cv2.putText(
            combined_image,
            text,
            position,
            text_settings["font"],
            1,
            text_settings["text_color"],
            2,
        )
        return combined_image

    def paint_margin(self, image: np.array) -> np.array:
        color = [23, 156, 125]  # green
        # color = [255,0,0] # red
        height, width, _ = image.shape
        margin_size = 10
        image_with_border = np.copy(image)
        image_with_border[0:margin_size, :] = color
        image_with_border[height - margin_size :, :] = color
        image_with_border[:, 0:margin_size] = color
        image_with_border[:, width - margin_size :] = color
        return image_with_border

    def make_sizes_fit(
        self, images: List[np.array], crop_sizes: tuple = None
    ) -> List[np.array]:
        cropped_images = []
        if crop_sizes is not None:
            for img, crop_size in zip(images, crop_sizes):
                if crop_size < 0:
                    cropped_images.append(img)
                    continue
                height, width = img.shape[:2]
                center_x, center_y = width // 2, height // 2
                x1 = center_x - crop_size // 2
                y1 = center_y - crop_size // 2
                x2 = x1 + crop_size
                y2 = y1 + crop_size
                cropped_images.append(img[y1:y2, x1:x2])
            images = cropped_images
        dim_min = min([min(img.shape[0], img.shape[1]) for img in images])
        fitted_images = []

        for img in images:
            _, w = img.shape[:2]
            scale_factor = dim_min / w
            new_img = cv2.resize(img, (0, 0), fy=scale_factor, fx=scale_factor)
            h, _ = new_img.shape[:2]
            y_0 = max(0, int(h / 2 - dim_min / 2))
            y_1 = min(h, int(h / 2 + dim_min / 2))

            new_img = new_img[y_0:y_1, :, :]
            fitted_images.append(new_img)
            if self.image_scale is None:
                self.image_scale = np.array([scale_factor, scale_factor])
                self.image_offset = np.array([-y_0, 0])
        return fitted_images

    def save_video(self, video: Union[np.array, List[np.array]]):
        if isinstance(video, np.ndarray):
            video_list = [np.array(arr) for arr in video]
            video = video_list
        os.makedirs(self.settings["save_path"], exist_ok=True)
        filename = self.settings["save_object_name"]
        num_frames = len(video)
        for i, frame_tensor in enumerate(video):
            index_str = f"{i:0{len(str(num_frames))}}"
            Visualizer.save_tensor_to_file(
                tensor=frame_tensor,
                filename=f"{filename}_{index_str}",
                out_folder=self.settings["save_path"],
            )

    def upsample_frames(
        self, start_frame: int = 0, end_frame: int = -1, scale: float = 2.0
    ) -> None:
        assert scale > 1, "function only works for upsampling"
        num_frames = len(self.video_model.get_dict()["video_dict"]["frames"])
        if end_frame == -1:
            end_frame = num_frames - 1

        frame_slice = list(range(start_frame, end_frame))
        new_frames = [frame + i / scale for frame in frame_slice for i in range(scale)]
        return new_frames

    def __call__(self):
        # k: CrossExpressionModelUser = CrossExpressionModelUser(settings)
        # upsampled_frames = k.upsample_frames(scale=4)
        # frames = [0]
        # video = self.inset_video()
        # video = k.edit("age", +10)
        # video = self.generate_video(frames=upsampled_frames)
        video = self.generate_video()
        # video = self.compensate_cropping_imbalance(video)
        video = self.generate_comparison_with_original(video)
        # video = self.video_model.get_dict()["video_dict"]["frames"]
        # video = self.add_landmarks(video, selection=[37, 41, 43, 47, 62, 66])
        # video = self.add_landmarks(video)
        # video = self.generate_comparison_with_original(video, original_index=0)
        # video = self.highlight_keyframes(video)
        # if k.video_model.has_additional_information():
        # video = k.add_debug_info(video)
        self.save_video(video)
        # self.settings["save_object_name"] = self.settings["save_object_name"]+"_orig"
        # self.save_video(original_images)

    @staticmethod
    def add_lmx(image: np.ndarray, landmarks: np.ndarray) -> np.array:
        lm_image = np.ascontiguousarray(np.copy(image), dtype=np.uint8)
        for x, y in landmarks:
            cv2.circle(
                lm_image,
                (int(x), int(y)),
                radius=int(0.005 * image.shape[0]),
                color=[255, 0, 0],
                thickness=-1,
            )
        return lm_image


class CrossPersonModelUser(KeyframeModelUser):
    def __init__(self, settings: Dict):
        super().__init__(settings)
        if "target_model" in settings.keys():
            self.swap_person_codes_combined_model(settings["target_model"])

    def swap_person_codes_combined_model(self, target_model: int):
        video_model: MultipleVideoModelObject = self.video_model
        target_person_codes = video_model.video_models[target_model].get_dict()[
            "person_codes"
        ]["w_person"]
        video_model.override_person_anchor(target_person_codes)


class NeutralExpressionModelUser(KeyframeModelUser):
    def get_generation_properties(self, frame_id: int, state) -> None:
        person_codes = state["person_codes"]
        ws = person_codes["w_person"]
        ws = torch.unsqueeze(ws, dim=0)
        poses = state["model"].get_neutral_camera()
        return ws, poses


class Editor(CrossPersonModelUser):
    def __init__(self, settings):
        sys.path.append(f"{PROJECT}/external/cgsgan/")

        super().__init__(settings)
        self.boundaries = self.load_boundaries(settings["boundary_path"])

    def load_boundaries(self, path):
        if isinstance(path, list):
            boundaries = [self.load_boundaries(p) for p in path]
            return torch.stack(boundaries, dim=0)

        if path[-3:] == "npy":
            boundary = np.load(path)
            return torch.from_numpy(boundary).to("cuda:0").unsqueeze(dim=0)
        elif path[-3:] == "pth":
            return torch.load(path)
        else:
            raise ValueError("wrong path")

    def __call__(self):
        concat_video = self.generate_edited_images(
            generate_original=self.settings["show_unedited"],
            accumulative=self.settings["accumulative_edits"],
        )
        self.save_video(concat_video)

    def generate_edited_images(self, accumulative: bool, generate_original: bool):
        orig_person_codes = self.video_model.get_dict()["person_codes"]["w_person"]
        videos = [
            self.video_model.get_dict()["video_dict"]["frames"],
        ]
        if generate_original:
            videos.append(self.generate_video())

        component = torch.zeros_like(self.boundaries[0, :])
        for ax_id, weight in self.settings["target_axis"]:
            boundary = self.boundaries[ax_id, :]
            if accumulative:
                component += boundary * weight
            else:
                component = boundary * weight
            edited_person_codes = orig_person_codes + component
            self.video_model.override_person_anchor(edited_person_codes)
            video = self.generate_video()
            videos.append(video)
        concat_video = self.concat_videos_list(videos, linebreak=8)
        return concat_video

    def concat_videos_list(self, videos, linebreak=3):
        video_len = len(videos[0])
        num_videos = len(videos)
        new_video = []

        for frame in range(video_len):
            grid_rows = []
            for i in range(0, num_videos, linebreak):
                row_videos = videos[i : i + linebreak]
                row = np.concatenate([video[frame] for video in row_videos], axis=1)
                grid_rows.append(row)

            grid_frame = np.concatenate(grid_rows, axis=0)
            new_video.append(grid_frame)

        return new_video
