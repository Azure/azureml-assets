# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""NextQA dataset loading utilities for video question answering benchmarks.

This module provides classes and functions for loading and processing
video files and NextQA dataset entries for benchmarking video-language models.
Supports both file-based video loading and HuggingFace dataset integration.
"""

import os
import sys
from typing import List

import av
from datasets import load_dataset


def find_video_files(video_dir) -> List[str]:
    """Recursively find all video files in a directory.

    Args:
        video_dir: Directory path to search for video files

    Returns:
        List[str]: List of video file paths found
    """
    if os.path.isfile(video_dir):
        return [video_dir]

    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mov")):
                video_files.append(os.path.join(root, file))
            # if file is dir
            elif os.path.isdir(file):
                video_files.extend(find_video_files(file))
    return video_files


def video_frames(video_path, max_frames) -> int:
    """Get the number of frames to extract from a video.

    Args:
        video_path: Path to the video file
        max_frames: Maximum number of frames to extract

    Returns:
        int: Actual number of frames to extract (min of total frames and max_frames)
    """
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    return min(total_frames, max_frames)


class Video:
    """Represents a video file with frame count information.

    Attributes:
        path (str): Path to the video file
        num_frames (int): Number of frames in the video
    """

    def __init__(self, video_path, num_frames):
        """Initialize a Video object.

        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames in the video
        """
        self.path = video_path
        self.num_frames = num_frames

    def __str__(self):
        """Return string representation of the Video object."""
        return f"Video({self.path}, {self.num_frames})"

    def __iter__(self):
        """Return iterator over video path and frame count."""
        return iter((self.path, self.num_frames))


class VideoPrompt(Video):
    """Represents a video with an associated text prompt for QA tasks.

    Inherits from Video and adds prompt functionality for question-answering.

    Attributes:
        path (str): Path to the video file
        num_frames (int): Number of frames in the video
        prompt (str): Text prompt/question associated with the video
    """

    def __init__(self, video_path, num_frames, prompt):
        """Initialize a VideoPrompt object.

        Args:
            video_path (str): Path to the video file
            num_frames (int): Number of frames in the video
            prompt (str): Text prompt/question for the video
        """
        super().__init__(video_path, num_frames)
        self.prompt = prompt

    def __str__(self):
        """Return string representation of the VideoPrompt object."""
        return f"VideoPrompt({self.path}, {self.num_frames}, {self.prompt})"

    def __iter__(self):
        """Return iterator over video path, frame count, and prompt."""
        return iter((self.path, self.num_frames, self.prompt))


class VideoLoader:
    """Base class for video loading implementations.

    Provides a common interface for different video loading strategies.
    """

    pass


class VideoFileLoader(VideoLoader):
    """Load videos from filesystem directory.

    Scans a directory for video files and provides iteration interface
    for processing videos in batches.
    """

    def __init__(self, video_dir, batch_size=1, max_frames=sys.maxsize):
        """Initialize video file loader.

        Args:
            video_dir (str): Directory containing video files
            batch_size (int): Number of videos per batch
            max_frames (int): Maximum frames to extract per video
        """
        super().__init__()
        self.video_dir = video_dir
        self.video_files = find_video_files(video_dir)
        self.batch_size = batch_size
        self.max_frames = max_frames
        print(f"batch_size: {batch_size}, max_frames: {max_frames}")

    def __iter__(self):
        """Iterate over video files in the directory.

        Yields:
            Video or List[Video]: Individual videos or batches of videos
        """
        if self.batch_size == 1:
            for video_file in self.video_files:
                yield Video(video_file, video_frames(video_file, self.max_frames))
        else:
            batch = []
            for video_file in self.video_files:
                video = Video(video_file, video_frames(video_file, self.max_frames))
                batch.append(video)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []


class NExTQALoader(VideoLoader):
    """Load videos and prompts from NextQA dataset.

    Integrates with HuggingFace datasets to load NextQA dataset and
    combines it with local video files for benchmarking.
    """

    def __init__(self, video_dir, batch_size=1, max_frames=sys.maxsize, dset="test", task="OE"):
        """Initialize NextQA dataset loader.

        Args:
            video_dir (str): Directory containing NextQA video files
            batch_size (int): Number of video prompts per batch
            max_frames (int): Maximum frames to extract per video
            dset (str): Dataset split ('train', 'test', 'validation')
            task (str): Task type ('MV' for multiple choice, 'OE' for open-ended)
        """
        super().__init__()
        self.task = task
        print(f"Loading the {dset} data of {task} from lmms-lab/NExTQA")
        self.ds = load_dataset("lmms-lab/NExTQA", task)
        self.ds = self.ds[dset]

        # self.n = ds.num_rows
        self.video_dir = video_dir
        self.video_files = find_video_files(video_dir)
        self.video_to_path = dict()
        for video_file in self.video_files:
            video_id = video_file.split("/")[-1].split(".")[0]
            self.video_to_path[video_id] = video_file

        self.batch_size = batch_size
        self.max_frames = max_frames

    def get_video_prompt(self, entry, max_frames) -> VideoPrompt:
        """Create VideoPrompt object from dataset entry.

        Args:
            entry: Dataset entry containing video and question information
            max_frames (int): Maximum number of frames to extract

        Returns:
            VideoPrompt: Video object with associated question prompt
        """
        # Get video
        video_id = entry["video"]
        video_path = self.video_to_path[video_id]
        assert os.path.exists(video_path), f"Video not found: {video_path}"
        num_frames = min(entry["frame_count"], max_frames)
        # video = Video(video_path, num_frames)
        prompt = entry["question"] + "?"
        if self.task == "MC":  # add choices
            prompt += f' a0: {entry["a0"]}, a1: {entry["a1"]}, a2: {entry["a2"]}, a3: {entry["a3"]}'
        return VideoPrompt(video_path, num_frames, prompt)

    def __iter__(self):
        """Iterate over NextQA dataset entries.

        Yields:
            VideoPrompt or List[VideoPrompt]: Individual video prompts or batches
        """
        if self.batch_size == 1:
            for entry in self.ds:
                yield self.get_video_prompt(entry, self.max_frames)
        else:
            batch = []
            for entry in self.ds:
                video = self.get_video_prompt(entry, self.max_frames)
                batch.append(video)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []


# main
if __name__ == "__main__":
    video_dir = "./videos"
    # video_loader = VideoFileLoader(video_dir, batch_size=16)
    # for batch in video_loader:
    #     print(f"Number of videos in batch: {len(batch)}")
    #     for video_file, num_frames in batch:
    #         print(f"Video: {video_file} number of frames: {num_frames}")

    video_loader = NExTQALoader(video_dir, batch_size=16, dset="test", task="OE")
    for batch in video_loader:
        print(f"Number of videos in batch: {len(batch)}")
        for video_file, num_frames, prompt in batch:
            print(f"Video: {video_file} number of frames: {num_frames}, prompt: {prompt}")
        # break
        # for video_file, prompt in batch:
        #     print(f"Video: {video_file} prompt: {prompt}")
        #     break
