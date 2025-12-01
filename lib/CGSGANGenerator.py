import copy
import torch
import numpy as np
from typing import Dict
from torch import Tensor
import pickle


class CGSGANGenerator:
    def __init__(self, network_pkl, device="cuda"):
        from camera_utils import LookAtPoseSampler  # type: ignore

        self.sample = LookAtPoseSampler.sample
        self.device = device
        self.load_tuned(network_pkl)
        self.active_G = self.generator
        self.G_tune = None
        self.set_camera_parameters()
        self.evaluate_average_w()
        self.BS = 8

    def get_neutral_camera(self, return_tensor=True):
        lookat_point = torch.tensor([0, 0, 0.2]).to(self.device)
        extrinsic = (
            self.sample(
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
            [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device="cuda:0"
        ).reshape((1, 9))
        if not return_tensor:
            return (extrinsic, intrinsics)
        cam = torch.cat((extrinsic, intrinsics), dim=1)
        return cam

    def make_sizes_fit(self, camera_params: np.array, w) -> None:
        # (,512) -> (1, 512)
        if len(w.shape) == 1:
            w = w.unsqueeze(0)

        # (x, 512) -> (x, 1, 512)
        if len(w.shape) == 2:
            w = w.unsqueeze(1)

        # repeat to (1, 14, 512)
        if w.shape[1] == 1:
            w = w.repeat(1, self.get_num_ws(), 1)

        num_images = max(len(w), len(camera_params))

        # replicate if single w was passed
        if w.shape[0] < num_images:
            w = w.repeat(num_images, 1, 1)

        # replicate if single camera parameter was passed
        if camera_params.shape[0] < num_images:
            camera_params = camera_params.repeat(num_images, 1)

        assert (
            w.shape[0] == camera_params.shape[0]
        ), f"incompatible size of w ({w.shape}) and camera params ({camera_params.shape})"
        return camera_params, num_images, w

    def get_average_face_tensors(self):
        average_w = self.get_average_w()
        left = self.get_camera_parameters(yaw=0.6, pitch=0.0, focal_length=4.26)
        center = self.get_camera_parameters(yaw=0.6, pitch=0.0, focal_length=4.26)
        right = self.get_camera_parameters(yaw=0.6, pitch=0.0, focal_length=4.26)
        average_face_tensors = []
        for cam in [left, center, right]:
            face_tensor = self.generate(average_w, camera_params=cam)
            average_face_tensors.append(face_tensor)

        return torch.cat(average_face_tensors, dim=0)

    def tune(self, force=False):
        if self.G_tune == None or force:
            self.G_tune = copy.deepcopy(self.generator.eval().to(self.device)).float()
        self.G_tune.requires_grad_(True)

        self.active_G = self.G_tune.to(self.device)

    def default(self):
        self.active_G = self.G

    def set_camera_parameters(self, focal_length=4.2647, cam_pivot=[0, 0, 0.2]):
        self.focal_length = focal_length
        self.intrinsics = torch.tensor(
            [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
            device=self.device,
        )
        self.cam_pivot = torch.tensor(cam_pivot, device=self.device)
        self.cam_radius = self.active_G.rendering_kwargs.get("avg_camera_radius", 2.7)
        self.conditioning_cam2world_pose = self.sample(
            np.pi / 2,
            np.pi / 2,
            self.cam_pivot,
            radius=self.cam_radius,
            device=self.device,
        )
        self.conditioning_params = torch.cat(
            [
                self.conditioning_cam2world_pose.reshape(-1, 16),
                self.intrinsics.reshape(-1, 9),
            ],
            1,
        )

    def evaluate_average_w(self, num_samples=10000):
        truncation_psi = 1
        truncation_cutoff = 14

        with torch.no_grad():
            z_samples = np.random.RandomState(123).randn(
                num_samples, self.active_G.z_dim
            )

            w_samples = self.active_G.mapping(
                torch.from_numpy(z_samples).to(self.device),
                self.conditioning_params.repeat(num_samples, 1),
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
            )

            w_samples = (
                w_samples[:, :1, :].cpu().numpy().astype(np.float32)
            )  # [N, 1, C] #
            w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
            self.w_avg_tensor = torch.from_numpy(w_avg[0]).to(self.device)
            self.w_std = (np.sum((w_samples - w_avg) ** 2) / num_samples) ** 0.5

    def get_ws(self, z_samples, truncation_psi=1, truncation_cutoff=14):
        return self.generator.mapping(
            z_samples,
            self.conditioning_params.repeat(len(z_samples), 1),
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
        )

    def get_random_ws(
        self, num_samples, truncation_psi=1, truncation_cutoff=14, seed=0
    ):
        z_rand = torch.from_numpy(
            np.random.RandomState(seed).randn(num_samples, self.generator.z_dim)
        ).to(self.device)
        return self.get_ws(
            z_rand, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff
        )

    def get_average_w(self):
        return self.w_avg_tensor

    def get_w_std(self):
        return self.w_std

    def get_camera_parameters(self, yaw=0.0, pitch=0.0, focal_length=None):
        focal_length = self.focal_length if focal_length == None else focal_length

        def is_list(variable):
            return type(variable) in (list, np.ndarray) or (
                type(variable) is torch.Tensor and variable.dim() > 0
            )

        # check if camera parameters are a list. for torch.tensors, check also if dimensionality is larger than 0, otherwise len() fails
        if is_list(yaw) or is_list(pitch) or is_list(focal_length):
            camera_params = []
            num_yaw = 1 if type(yaw) == float else len(yaw)
            num_pitch = 1 if type(pitch) == float else len(pitch)
            num_fl = 1 if type(focal_length) == float else len(focal_length)
            create_num_parameters = max(num_yaw, num_pitch, num_fl)
            for i in range(create_num_parameters):
                # check for each parameter if single float was passed, otherwise fill from list
                y = yaw if type(yaw) == float else yaw[i]
                p = pitch if type(pitch) == float else pitch[i]
                fl = focal_length if type(focal_length) == float else focal_length[i]
                intrinsics = torch.tensor(
                    [[fl, 0, 0.5], [0, fl, 0.5], [0, 0, 1]], device=self.device
                )

                cam2world_pose = self.sample(
                    np.pi / 2 + y,
                    np.pi / 2 + p,
                    self.cam_pivot,
                    radius=self.cam_radius,
                    device=self.device,
                )
                camera_params.append(
                    torch.cat(
                        [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
                    )
                )
            camera_params = torch.cat(camera_params, axis=0)
        else:
            intrinsics = torch.tensor(
                [[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]],
                device=self.device,
            )
            cam2world_pose = self.sample(
                np.pi / 2 + yaw,
                np.pi / 2 + pitch,
                self.cam_pivot,
                radius=self.cam_radius,
                device=self.device,
            )
            camera_params = torch.cat(
                [cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1
            )

        return camera_params

    def normalize_depth(self, depth_tensor, resize=True):
        depth_image = -depth_tensor
        depth_image = (depth_image - depth_image.min()) / (
            depth_image.max() - depth_image.min()
        ) * 2 - 1

        if resize:
            depth_image = F.interpolate(
                depth_image.repeat(1, 3, 1, 1), size=(512, 512), mode="nearest"
            )
        return depth_image

    def poses_dict_to_matrix(self, poses: Dict, return_all=False) -> torch.Tensor:
        intrinsics = [
            torch.from_numpy(
                pose["intrinsics"],
            )
            .flatten()
            .unsqueeze(dim=0)
            for pose in poses
        ]
        intrinsics = torch.cat(intrinsics, dim=0).to(
            device=self.device, dtype=torch.float32
        )
        extrinsics = [
            torch.from_numpy(pose["extrinsics"]).flatten().unsqueeze(dim=0)
            for pose in poses
        ]
        extrinsics = torch.cat(extrinsics, dim=0).to(
            device=self.device, dtype=torch.float32
        )
        mat = torch.cat((extrinsics, intrinsics), dim=1)
        if return_all:
            return mat, extrinsics, intrinsics
        return mat

    def get_num_ws(self):
        return 1

    def load_tuned(self, tuned_generator_path):
        with open(tuned_generator_path, "rb") as file:
            save_file = pickle.load(file)
            weights = save_file["G_ema"]
            self.set_tuned(weights)

    def set_tuned(self, weights):
        G = weights.to(self.device)
        self.generator = copy.deepcopy(G).train().requires_grad_(True).to(self.device)

    def generate(
        self,
        w,
        camera_params: Tensor = None,
        extrinsics: Tensor = None,
        intrinsics: Tensor = None,
        noise_mode="const",
        output_all=False,
        grad=False,
    ):
        if len(w.shape) == 4 and w.shape[0] == 1:
            w = w.squeeze(dim=0)
        if camera_params is None:
            if extrinsics.shape[1] == 4 and len(extrinsics.shape) == 2:
                extrinsics = extrinsics.reshape((1, 16))
            if intrinsics.shape[1] == 3 and len(intrinsics.shape) == 2:
                intrinsics = intrinsics.reshape((1, 9))
            camera_params = torch.cat((extrinsics, intrinsics), dim=1)
        camera_params, num_images, w = self.make_sizes_fit(camera_params, w)

        if grad:
            ret_dict = self.active_G.synthesis(
                ws=w.to(torch.float32),
                c=camera_params,
                noise_mode="const",
                gs_params=None,
                random_bg=False,
            )
        else:
            with torch.no_grad():
                ret_dict = self.active_G.synthesis(
                    ws=w.to(torch.float32),
                    c=camera_params,
                    noise_mode="const",
                    gs_params=None,
                    random_bg=False,
                )
        ret_dict["image"] = torch.nn.functional.interpolate(
            ret_dict["image"], size=(512, 512), mode="bilinear", align_corners=False
        )
        if output_all:
            ret_dict["image_raw"] = scaled_tensor = torch.nn.functional.interpolate(
                ret_dict["image"], size=(128, 128), mode="bilinear", align_corners=False
            )
            return ret_dict
        else:
            return ret_dict["image"]
