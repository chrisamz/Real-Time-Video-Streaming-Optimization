# video_compression.py

"""
Video Compression Module for Real-Time Video Streaming Optimization

This module contains functions for compressing video streams to optimize bandwidth usage
without compromising quality.

Techniques Used:
- H.264
- H.265
- VP9

Libraries/Tools:
- FFmpeg
- OpenCV
"""

import os
import subprocess
import cv2

class VideoCompression:
    def __init__(self, codec='h264', crf=23, preset='medium'):
        """
        Initialize the VideoCompression class.
        
        :param codec: str, video codec to use ('h264', 'h265', 'vp9')
        :param crf: int, Constant Rate Factor for controlling quality (lower is better quality)
        :param preset: str, preset for compression speed vs. compression ratio
        """
        self.codec = codec
        self.crf = crf
        self.preset = preset

    def compress_video_ffmpeg(self, input_filepath, output_filepath):
        """
        Compress video using FFmpeg.
        
        :param input_filepath: str, path to the input video file
        :param output_filepath: str, path to save the compressed video file
        """
        codec_map = {
            'h264': 'libx264',
            'h265': 'libx265',
            'vp9': 'libvpx-vp9'
        }
        codec = codec_map.get(self.codec)
        if codec is None:
            raise ValueError(f"Codec {self.codec} not supported.")
        
        cmd = [
            'ffmpeg', '-i', input_filepath, '-c:v', codec, '-crf', str(self.crf),
            '-preset', self.preset, output_filepath
        ]
        subprocess.run(cmd, check=True)
        print(f"Video compressed using {self.codec} and saved to {output_filepath}.")

    def compress_video_opencv(self, input_filepath, output_filepath, width, height, fps):
        """
        Compress video using OpenCV.
        
        :param input_filepath: str, path to the input video file
        :param output_filepath: str, path to save the compressed video file
        :param width: int, width of the output video
        :param height: int, height of the output video
        :param fps: int, frames per second of the output video
        """
        cap = cv2.VideoCapture(input_filepath)
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        out = cv2.VideoWriter(output_filepath, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height))
            out.write(frame)

        cap.release()
        out.release()
        print(f"Video compressed using OpenCV and saved to {output_filepath}.")

if __name__ == "__main__":
    input_video_path = 'data/raw/input_video.mp4'
    output_video_path_ffmpeg = 'data/compressed/output_video_ffmpeg.mp4'
    output_video_path_opencv = 'data/compressed/output_video_opencv.mp4'
    width = 1280
    height = 720
    fps = 30

    # Compress video using FFmpeg
    compressor_ffmpeg = VideoCompression(codec='h264', crf=23, preset='medium')
    compressor_ffmpeg.compress_video_ffmpeg(input_video_path, output_video_path_ffmpeg)

    # Compress video using OpenCV
    compressor_opencv = VideoCompression(codec='h264', crf=23, preset='medium')
    compressor_opencv.compress_video_opencv(input_video_path, output_video_path_opencv, width, height, fps)
