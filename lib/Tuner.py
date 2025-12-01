from vive3D.landmark_detector import *
from vive3D.segmenter import *
from vive3D.inset_pipeline import *
from vive3D.aligner import *
from lib.DictOperations import DictOperations as DictOps
import sys
from lib import VideoModelObject, State
import torch
import os
from configs.config import PROJECT
from lib import CGSGANGenerator, MultipleVideoModelObject


class KeyframeModelTuner:
    def __init__(self, settings):
        self.video_state = None
        self.save_path = settings["save_path"]
        self.name = settings["save_object_name"]
        self.analyser_options = settings["analyser_options"]
        self.model_type = settings["model_type"]
        self.settings = settings
        sys.path.append(f"{PROJECT}/external/cgsgan/")
        self.load_path = settings["load_path"]
        video_models = [self.initialize_video_model_object(l) for l in self.load_path]
        self.video_model = MultipleVideoModelObject(
            video_models=video_models, name=self.name
        )
        self.final_state = State.get_max_state(multiple_videos=True)

    def initialize_video_model_object(self, load_path):
        _, file_ending = os.path.splitext(load_path)
        assert file_ending == ".mp4", "wrong file type"
        torch.cuda.empty_cache()
        video_model = VideoModelObject.initialize_from_video(
            video_path=load_path,
            analyzer_options=self.analyser_options["analyse_video"],
            name=self.name,
        )
        return video_model

    def __call__(self):
        self.video_model.upgrade_to_state(
            self.final_state, options=self.analyser_options
        )
        assert self.video_model.get_state() == self.final_state
        self.video_model.save_video_model(path=self.save_path)
        DictOps.save(self.settings, self.save_path, "settings")
