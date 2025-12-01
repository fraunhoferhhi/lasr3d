from .Logger import LoggerWrapper
import wandb
from .DebugUtil import save_debug_image
from lib.SpectreUser import SpectreLoss
from .KeyframeSelector import KeyFrameSelector
from .CGSGANGenerator import CGSGANGenerator
import random
from configs.config import PROJECT
from vive3D.inset_pipeline import Space_Regularizer, lpips, BicubicDownSample # type: ignore
from vive3D.aligner import Aligner # type: ignore
from vive3D.segmenter import Segmenter # type: ignore
from vive3D.landmark_detector import LandmarkDetector # type: ignore
from vive3D.util import tensor_to_image # type: ignore
from typing import List, Dict, Tuple, Union
from torch import Tensor
import torch
from .IDLoss import IDLoss
import numpy as np
import cv2


class TuningPipeline:
    def __init__(
        self,
        video_model,
        generator,
        device="cuda",
    ):
        self.device = device
        self.segmenter: Segmenter = Segmenter(
            device=self.device, path=f"{PROJECT}/models/79999_iter.pth"
        )
        self.landmark_detector: LandmarkDetector = LandmarkDetector(device=self.device)
        self.space_regularizer: Space_Regularizer = None
        self.align: Aligner = Aligner(
            landmark_detector=self.landmark_detector,
            segmenter=self.segmenter,
            device=self.device,
        )
        self.generator: CGSGANGenerator = generator
        self.loss_L1 = torch.nn.L1Loss(reduction="sum").to(device)  #
        self.loss_L2 = torch.nn.MSELoss(reduction="sum").to(device)  #
        self.loss_percept = lpips.LPIPS(net="alex").to(device)
        self.downsampler_128 = BicubicDownSample(factor=512 // 128, device=device).to(
            device
        )

        self.total_number_of_images: int = None
        self.person_breakpoints = []
        self.last_cuda_values = {"reserved": 0, "alloc": 0}
        self.id_loss = IDLoss()
        self.video_model = video_model
        self.spectre_loss: SpectreLoss = SpectreLoss(device=device, yaw_focus=True)
        self.person_ids = []

    def __call__(
        self, input: Dict, target: Dict, hyperparameters: Dict = None
    ) -> Tensor:
        from .VideoModelObject import MultipleVideoModelObject

        self.video_model: MultipleVideoModelObject = self.video_model

        self.get_original_id_vectors(target["images"])

        optimizer, w_offsets = self.pre_tuning(input, hyperparameters)
        self.get_full_inversion_target(target)
        with LoggerWrapper(project="LaSR-3D") as logger:
            for i in range(hyperparameters["num_steps"]):
                if i % 50 == 0:
                    print(f"Tuning Iteration: {i}")
                gen_images, indices, offsets = self.get_images_random(
                    input, hyperparameters
                )
                losses = self.calculate_loss(
                    gen_images, target, indices, hyperparameters, i=i
                )
                if losses["lpips"].sum() <= 0.006:
                    return

                losses["full"] += (
                    torch.norm(offsets, p=2)
                    * hyperparameters["loss_weights"]["w_offset_regularisation"]
                )
                losses["full"].backward()
                logger.log(losses)
                if i % hyperparameters["backwards_step_distance"] == 0:
                    torch.cuda.empty_cache()
                    torch.cuda.memory.empty_cache()
                    loss, source_frames = self.calculate_spectre_loss(
                        input, hyperparameters, i
                    )
                    logger.log({"tune/spectre": loss})
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                torch.cuda.empty_cache()
                torch.cuda.memory.empty_cache()

        return w_offsets, self.generator

    def pre_tuning(
        self, input: Dict, hyperparameters: Dict
    ) -> Tuple[torch.optim.Adam, List[Tensor]]:
        self.generator.tune(force=True)
        self.space_regularizer = Space_Regularizer(
            self.generator, self.loss_percept, self.loss_L2
        )
        w_offsets = [
            person_offsets.to(self.device) for person_offsets in input["w_offsets"]
        ]
        w_offsets = torch.stack(w_offsets, dim=0).clone().detach().requires_grad_(True)
        params = self.generator.active_G.parameters()
        trainable = [{"params": params}, {"params": w_offsets}]
        optimizer: torch.optim.Adam = torch.optim.Adam(
            trainable,
            lr=hyperparameters["learning_rate"],
        )
        return optimizer, w_offsets

    def get_full_inversion_target(self, target):
        assert self.total_number_of_images is None and not self.person_breakpoints
        self.total_number_of_images = 0
        images = target["images"]
        target["images"] = []
        target["images_128"] = []

        for img in images:
            num_frames_for_person = img.shape[0]
            self.total_number_of_images += num_frames_for_person
            self.person_breakpoints.append(self.total_number_of_images)
            target["images"].append(img.to(self.device))
            target["images_128"].append(self.downsampler_128(img))

    def get_images_random(
        self, input: Dict, hyperparameters: Dict
    ) -> Tuple[Dict, List[int]]:
        indices = self.pick_random_indices(hyperparameters["batch_size"])
        w_gen, w_offsets, cam = self.generate_ws(indices, input)
        generated = self.generator.generate(w_gen, camera_params=cam, grad=True)
        return generated, indices, w_offsets

    def pick_random_indices(self, batch_size: int) -> List[Tuple[int, int]]:
        indices = random.sample(range(self.total_number_of_images), batch_size)
        return self.get_person_index_tuple(indices)

    def get_person_index_tuple(self, idx: Union[List[int], int]) -> any:
        if isinstance(idx, list):
            return [self.get_person_index_tuple(i) for i in idx]

        last_breakpoint = 0
        for person_idx, cumulative_length in enumerate(self.person_breakpoints):
            if idx < cumulative_length:
                return (person_idx, idx - last_breakpoint)
            last_breakpoint = cumulative_length

    def get_number_of_persons(self) -> int:
        return len(self.person_breakpoints)

    @staticmethod
    def get_element_from_tuple(
        source, idx: Union[List[int], int], to_tensor: bool = False
    ) -> any:
        if isinstance(idx, list):
            elements = [TuningPipeline.get_element_from_tuple(source, i) for i in idx]
            if to_tensor:
                elements = torch.stack(elements)
            return elements
        return source[idx[0]][idx[1]]

    def calculate_loss(
        self,
        gen_images,
        target: Dict,
        indices: List[int],
        hyperparameters: Dict,
        i: int = 0,
    ) -> Dict:
        weights: dict = hyperparameters["loss_weights"]
        target_image = self.get_element_from_tuple(
            target["images"], indices, to_tensor=True
        )
        loss_l1 = (
            self.loss_L1(gen_images, target_image)
            / (hyperparameters["res"] ** 2)
            * weights["l1"]
        )
        loss_lpips = (
            self.loss_percept(gen_images, target_image).sum() * weights["lpips"]
        )
        grid_img = np.concatenate(
            (tensor_to_image(target_image[0]), tensor_to_image(gen_images[0])),
            axis=1,
        )
        loss_id = self.id_loss(target_image, gen_images) * weights["id"]

        loss = loss_l1 + loss_lpips + loss_id
        save_debug_image(grid_img, name="main_tuning", step=i, from_tensor=False)
        return {"full": loss, "l1": loss_l1, "lpips": loss_lpips, "id": loss_id}

    def generate_ws(self, indices, input):
        keyframes = [input["keyframes"][person_idx] for person_idx, _ in indices]
        w_persons = [input["w_person"][person_idx] for person_idx, _ in indices]
        w_offsets = [input["w_offsets"][person_idx] for person_idx, _ in indices]
        camera_params = [input["poses"][person_idx] for person_idx, _ in indices]

        offsets = []
        w_gen = []
        cams = []
        for batch_index, idx_tuple in enumerate(indices):
            neighbours = KeyFrameSelector.get_neighbours(
                keyframes[batch_index], idx_tuple[1]
            )
            neighbour_ws = KeyFrameSelector.get_parameters_from_neighbours(
                neighbours, w_offsets[batch_index]
            ).to(self.device)
            cam = KeyFrameSelector.get_parameters_from_neighbours(
                neighbours, camera_params[batch_index]
            ).to(self.device)
            w_gen.append(neighbour_ws + w_persons[batch_index])
            offsets.append(neighbour_ws)
            cams.append(cam)

        w_gen = torch.stack(w_gen, dim=0)
        offsets = torch.stack(offsets, dim=0)
        cam = torch.stack(cams, dim=0)
        return w_gen, offsets, cam

    def get_original_id_vectors(self, images):
        for person_images in images:
            middle = int(len(person_images) / 2)
            samples = [person_images[0], person_images[middle], person_images[-1]]
            samples = torch.stack(samples, dim=0)
            feats = self.id_loss.extract_feats(samples.to(self.device), return_all=True)
            feats = torch.mean(feats, dim=0)
            self.person_ids.append(feats)

    def calculate_spectre_loss(self, input: Dict, hyperparameters: Dict, i: int):
        assert self.get_number_of_persons() == 2
        len_video = 3
        source_id = random.randint(0, 1)
        target_id = 1 - source_id

        target_dict = self.video_model.video_models[target_id].get_dict()
        len_target_video = len(target_dict["video_dict"]["frames"])

        start_id = random.randint(0, len_target_video - 1 - len_video)
        video_range = range(start_id, start_id + len_video)

        source_frames, w_offsets = self.get_cross_images(
            input, source_id, target_id, video_range
        )
        target_frame = target_dict["video_dict"]["face_tensors"][start_id].to(
            self.device
        )
        source_frame = source_frames[0]
        save_debug_image(
            torch.cat((target_frame, source_frame), dim=2), name="tunig_cross", step=i
        )
        loss_spectre = (
            self.spectre_loss(source_frames, target_dict, video_range, debug=True, i=i)
            * hyperparameters["loss_weights"]["spectre"]
        )
        reg_loss = (
            torch.norm(w_offsets, p=2)
            * hyperparameters["loss_weights"]["w_offset_regularisation"]
        )
        original_id_encoding = self.person_ids[source_id].clone().detach()
        source_id_encodings = self.id_loss.extract_feats(source_frames)
        loss_id = 1 - original_id_encoding.dot(source_id_encodings)
        loss_id *= hyperparameters["loss_weights"]["id"]
        loss = loss_spectre + reg_loss + loss_id
        return loss, source_frames

    def get_cross_images(
        self, input: dict, source_id: int, target_id: int, video_range: range
    ) -> Tuple[Dict, List[int]]:
        source_w_person = input["w_person"][source_id]
        target_w_offsets = input["w_offsets"][target_id]
        target_keyframes = input["keyframes"][target_id]
        target_camera_params = input["poses"][target_id]

        w_gen = []
        for i in video_range:
            neighbours = KeyFrameSelector.get_neighbours(target_keyframes, i)
            neighbour_ws = KeyFrameSelector.get_parameters_from_neighbours(
                neighbours, target_w_offsets
            ).to(self.device)
            cam = KeyFrameSelector.get_parameters_from_neighbours(
                neighbours, target_camera_params
            ).unsqueeze(dim=0)
            ws = neighbour_ws + source_w_person
            w_gen.append(ws.unsqueeze(dim=0))
        w_gen = torch.cat(w_gen)
        image = self.generator.generate(camera_params=cam, w=w_gen)
        return image, target_w_offsets


class InversionPipeline:

    def __init__(self, video_model, generator, device="cuda:0"):
        self.video_model = video_model
        self.generator: CGSGANGenerator = generator
        self.device = device
        self.loss_L1 = torch.nn.L1Loss(reduction="sum").to(device)  #
        self.loss_percept = lpips.LPIPS(net="alex").to(device)
        self.spectre_loss: SpectreLoss = SpectreLoss(device=device, yaw_focus=True)
        self.neutral_pose = None

    def generate_poses_from_dict(self, poses, keyframes):
        persons = []
        for person_id, person_poses in enumerate(poses):
            extr = [torch.tensor(p["extrinsics"]) for p in person_poses]
            extr = torch.stack(extr, dim=0)
            extr = extr[keyframes[person_id], ...]
            extr = extr.reshape([-1, 16])
            extr = extr.clone().detach().requires_grad_(False).to(self.device)
            persons.append(extr)
        persons = (
            torch.stack(persons, dim=0)
            .clone()
            .detach()
            .requires_grad_(False)
            .to(dtype=torch.float)
        )
        return persons

    def pre_inversion(self, input, hyperparameters):
        _, intrinsics = self.get_neutral_camera()
        keyframes = input["keyframes"]
        extrinsics = self.generate_poses_from_dict(input["poses"], keyframes)
        num_persons = len(keyframes)
        num_frames = len(keyframes[0])
        intrinsics = (
            intrinsics.reshape(1, 9)
            .repeat((num_persons, num_frames, 1))
            .clone()
            .detach()
            .requires_grad_(False)
            .to(self.device)
        )
        c = torch.cat((extrinsics, intrinsics), dim=2)
        w_person = torch.stack(input["w_person"], dim=0).unsqueeze(dim=1)
        if w_person.shape[2] != self.generator.get_num_ws():
            w_person = w_person.repeat(1, 1, self.generator.get_num_ws(), 1)
        w_person = (
            w_person.clone()
            .detach()
            .requires_grad_(True)
            .to(self.device, dtype=torch.float)
        )
        person_w_offsets = input["w_offsets"][0].clone().detach()
        avg_offsets = torch.mean(person_w_offsets, axis=0, keepdims=True)
        w_offsets = (
            avg_offsets.repeat(num_persons, num_frames, 1, 1)
            .clone()
            .to(self.device)
            .detach()
            .requires_grad_(True)
        )
        trainable = [{"params": w_offsets}] + [{"params": w_person}]
        if hyperparameters["learn_pose"]:
            trainable += [{"params": extrinsics}]
        optimizer: torch.optim.Adam = torch.optim.Adam(
            trainable,
            lr=hyperparameters["learning_rate"],
        )

        return optimizer, w_offsets, c, w_person

    def calculate_loss(
        self, gen_images, target, indices, hyperparameters, w_offsets, i=0
    ):
        losses = {"full": 0}
        person_id, start_frame, stop_frame = indices
        res = 512
        synth_image = gen_images
        target_image = target["images"][person_id][start_frame:stop_frame, ...]
        # lm = target["landmarks"][person_id][start_frame:stop_frame]
        # target_image = self.center_images(target_image, lm)
        # self.print_image_with_landmarks(target_image[0].unsqueeze(dim = 0), lm[0])
        losses["pix"] = self.loss_L1(synth_image, target_image) / (res**2)

        losses["lpips"] = self.loss_percept(synth_image, target_image).sum()
        losses["wdist"] = torch.linalg.norm(
            w_offsets[person_id][start_frame:stop_frame, ...]
        )
        generated_face = synth_image * target["face_segmentation"][person_id]
        losses["face"] = (
            self.loss_percept(
                generated_face, target["face"][person_id][start_frame:stop_frame, ...]
            ).sum()
            + self.loss_L1(
                generated_face, target["face"][person_id][start_frame:stop_frame, ...]
            ).sum()
            / target["num_face_px"][person_id]
        )
        grid_img = np.concatenate(
            (
                tensor_to_image(target_image[0]),
                tensor_to_image(target["face"][person_id][start_frame]),
                tensor_to_image(generated_face[0]),
                tensor_to_image(gen_images[0]),
            ),
            axis=1,
        )
        for key, value in losses.items():
            if key == "full":
                continue
            losses["full"] += value * hyperparameters["loss_weights"][key]
        return losses, grid_img

    def calculate_spectre_loss(self, hyperparameters, w_person, cam, i=None):
        pose = self.generator.get_neutral_camera()
        source_code = w_person[0, ...].unsqueeze(dim=0)
        target_code = w_person[1, ...].unsqueeze(dim=0)
        source_image = self.generator.generate(w=source_code, camera_params=pose)
        target_image = self.generator.generate(w=target_code, camera_params=pose)
        loss = self.spectre_loss.get_loss_only_picutres(
            source_image, target_image, debug=False
        )
        if i is not None:
            source_debug_img = tensor_to_image(source_image)
            source_debug_img = cv2.cvtColor(source_debug_img, cv2.COLOR_BGR2RGB)

            target_debug_img = tensor_to_image(target_image)
            target_debug_img = cv2.cvtColor(target_debug_img, cv2.COLOR_BGR2RGB)

            grid_img = np.concatenate((source_debug_img, target_debug_img), axis=1)
        return loss * hyperparameters["loss_weights"]["spectre"]

    def __call__(self, input, target, hyperparameters):
        with LoggerWrapper(project="LaSR-3D") as logger:
            optimizer, w_offsets, cam, w_person = self.pre_inversion(input, hyperparameters)
            num_persons, len_videos, _, _ = w_offsets.shape
            bs = 1  # thats important because synthesize doesn't work with more than 1
            for epoch in range(hyperparameters["num_steps"]):
                for person_id in range(num_persons):
                    for image_id in range(0, len_videos, bs):
                        end_index = min(len_videos, image_id + bs)
                        w_gen = (
                            w_person[person_id] + w_offsets[person_id, image_id:end_index]
                        )
                        c_gen = cam[person_id, image_id:end_index]
                        gen_images = self.generator.generate(
                            w_gen, camera_params=c_gen, output_all=False, grad=True
                        )
                        losses, grid_img = self.calculate_loss(
                            gen_images,
                            target,
                            indices=(person_id, image_id, end_index),
                            hyperparameters=hyperparameters,
                            w_offsets=w_offsets,
                        )
                        loss = losses["full"]
                        logger.log(losses)  
                        loss.backward()
                        if image_id in [0, 3, 7, 20, 35, 50] and epoch % 10 == 9:
                            # if epoch%2 == 0:
                            save_debug_image(
                                grid_img,
                                name=f"inversion_person_{person_id}_{image_id}",
                                step=epoch,
                                from_tensor=False,
                            )
                loss_spectre = self.calculate_spectre_loss(
                    hyperparameters, w_person, cam, epoch
                )
                logger.log({"inversion/spectre": loss_spectre})
                loss_spectre.backward()
                optimizer.step()
                optimizer.zero_grad()
        return w_person, w_offsets, cam

    def get_neutral_camera(self):
        if self.neutral_pose is None:
            lookat_point = torch.tensor([0, 0, 0.2]).to(self.device)
            extrinsic = (
                self.generator.sample(
                    3.14 / 2,
                    3.14 / 2,
                    lookat_point,
                    radius=2.7,
                    device=self.device,
                )
                .reshape(-1, 16)
                .to(self.device)
                .clone()
                .detach()
                .requires_grad_(False)
            )
            focal_length = 4.2647
            intrinsics = torch.tensor(
                [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
                device="cuda:0",
            ).reshape((1, 9))
            self.neutral_pose = (extrinsic, intrinsics)
        return self.neutral_pose

    def get_average_person(self, num_persons, num_images):
        neutral_extr, intrinsics = self.get_neutral_camera()
        extrinsics = neutral_extr.reshape((1, 16)).clone().detach().to(self.device)
        intrinsics = intrinsics.reshape(1, 9).clone().detach().to(self.device)
        c = torch.cat((extrinsics, intrinsics), dim=1)
        z = torch.rand((1, 512)).to("cuda:0")
        w_person = self.generator.active_G.mapping(
            z=z, c=c, truncation_psi=0.7, truncation_cutoff=14
        )
        w_person = w_person.repeat(num_persons, 1, 1)
        w_offsets = (
            torch.rand(num_persons, num_images, self.generator.get_num_ws(), 512) - 0.5
        )
        w_offsets = w_offsets * 0.1
        return w_person, w_offsets
