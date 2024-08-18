import cv2
import os
import numpy as np
import csv
import re
from timeit import default_timer as timer
from datetime import datetime

# Class for reading files in the dataset folders
class DirectoryReader:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    def get_participant_folders(self):
        return self._fetch_subfolders(self.root_folder)

    def get_action_folders(self, participant_folder):
        participant_path = os.path.join(self.root_folder, participant_folder)
        return self._fetch_subfolders(participant_path)

    def get_session_folders(self, participant_folder, action_folder):
        action_path = os.path.join(self.root_folder, participant_folder, action_folder)
        return self._fetch_subfolders(action_path)

    def get_angle_folders(self, participant_folder, action_folder, session_folder):
        session_path = os.path.join(self.root_folder, participant_folder, action_folder, session_folder)
        return self._fetch_subfolders(session_path)

    def _fetch_subfolders(self, folder):
        subfolders = [d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))]
        return sorted(subfolders, key=lambda x: int(re.search(r'\d+', x).group()))

# Class for loading individual frames from the dataset
class FrameReader:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

    def get_clip_folders(self):
        return [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]

    def load_frames_from_clip(self, clip_folder):
        img_folder = os.path.join(self.dataset_dir, clip_folder)
        frame_files = sorted([f for f in os.listdir(img_folder) if f.endswith('.jpg') or f.endswith('.png')])
        return [(fn[:-4], cv2.imread(os.path.join(img_folder, fn), cv2.IMREAD_GRAYSCALE)) for fn in frame_files]

# Class for normalization, blurring, background subtraction, and resize
class FlowCalculator:
    def __init__(self):
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=True)
        self.background_subtractor.setShadowValue(0)
        self.background_subtractor.setShadowThreshold(0.5)
        self.learning_rate = -1

    def calculate_flow(self, previous_frame, next_frame):
        previous_frame = cv2.normalize(src=previous_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        next_frame = cv2.normalize(src=next_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        mask_prev = self.background_subtractor.apply(previous_frame, learningRate=self.learning_rate)
        mask_next = self.background_subtractor.apply(next_frame, learningRate=self.learning_rate)
        kernel = np.ones((4, 4), np.uint8)
        mask_prev = cv2.morphologyEx(mask_prev, cv2.MORPH_CLOSE, kernel)
        mask_prev = cv2.morphologyEx(mask_prev, cv2.MORPH_OPEN, kernel)
        mask_next = cv2.morphologyEx(mask_next, cv2.MORPH_CLOSE, kernel)
        mask_next = cv2.morphologyEx(mask_next, cv2.MORPH_OPEN, kernel)
        prev_masked = cv2.bitwise_and(previous_frame, previous_frame, mask=mask_prev)
        next_masked = cv2.bitwise_and(next_frame, next_frame, mask=mask_next)
        prev_frame_blurred = cv2.medianBlur(prev_masked, 5)
        next_frame_blurred = cv2.medianBlur(next_masked, 5)
        prev_blurred = cv2.GaussianBlur(prev_frame_blurred, (5, 5), 0)
        next_blurred = cv2.GaussianBlur(next_frame_blurred, (5, 5), 0)
        flow_data = cv2.calcOpticalFlowFarneback(prev_blurred, next_blurred, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u_flow = flow_data[..., 0]
        v_flow = flow_data[..., 1]
        resized_u_flow = cv2.resize(u_flow, (51, 38))
        resized_v_flow = cv2.resize(v_flow, (51, 38))
        return resized_u_flow, resized_v_flow

# Class to write the data into NumPy arrays
class NumpySaver:
    def __init__(self, save_folder):
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

    def save_array(self, data_array, file_name):
        save_path = os.path.join(self.save_folder, f"{file_name}.npy")
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok=True)
        np.save(save_path, data_array)

# Class to process data into Optical Flow frames
class FlowProcessor:
    def __init__(self, dataset_dir, save_dir, frame_rate=18):
        self.frame_reader = FrameReader(dataset_dir)
        self.numpy_saver = NumpySaver(save_dir)
        self.frame_rate = frame_rate
        self.window_length = frame_rate
        self.overlap_length = frame_rate // 2
        self.flow_calculator = FlowCalculator()

    def calculate_total_seconds(timestamp: str) -> float:
        hrs, mins, secs = map(float, timestamp.split('T')[1].split('_'))
        total_seconds = hrs * 3600 + mins * 60 + secs
        return total_seconds

    def update_timestamp(timestamp: str) -> str:
        date_part, time_part = timestamp.split('T')
        try:
            hrs, mins, remainder = time_part.split('_')
            secs, microsecs = remainder.split('.')
        except ValueError:
            print(f"Error with timestamp: {time_part}")
            raise
        microsecs = int(microsecs)
        secs = int(secs)
        mins = int(mins)
        hrs = int(hrs)
        microsecs += 500000
        if microsecs >= 1000000:
            microsecs -= 1000000
            secs += 1
        if secs >= 60:
            secs -= 60
            mins += 1
        if mins >= 60:
            mins -= 60
            hrs += 1
        time_string = f"{hrs:02}_{mins:02}_{secs:02}.{microsecs:06}"
        return f"{date_part}T{time_string}"

    def process_clip(self, clip_folder):
        frame_data = self.frame_reader.load_frames_from_clip(clip_folder)
        index = 0
        frames_in_window = int(self.frame_rate)
        overlap_frame_count = int(self.overlap_length)
        final_frame_time = frame_data[-1][0]
        final_frame_seconds = FlowProcessor.calculate_total_seconds(final_frame_time)
        time_marker = frame_data[index][0]

        while index < len(frame_data) - frames_in_window:
            end_window = min(index + frames_in_window, len(frame_data))
            u_flows = []
            v_flows = []

            for frame_index in range(index, end_window - 1):
                try:
                    u_flow, v_flow = self.flow_calculator.calculate_flow(frame_data[frame_index][1], frame_data[frame_index + 1][1])
                    u_flows.append(u_flow)
                    v_flows.append(v_flow)
                except cv2.error as err:
                    print(f"Error processing frame {frame_data[frame_index][0]} from clip {clip_folder}. Error: {err}")
                    continue

            u_flow_stack = np.stack(u_flows, axis=0)
            v_flow_stack = np.stack(v_flows, axis=0)
            combined_flow_data = np.stack([u_flow_stack, v_flow_stack], axis=-1)
            window_label = f"{clip_folder}_{time_marker}"
            self.numpy_saver.save_array(combined_flow_data, window_label)
            time_marker = FlowProcessor.update_timestamp(time_marker)
            increment_seconds = FlowProcessor.calculate_total_seconds(time_marker)

            if final_frame_seconds - increment_seconds < 1.0:
                break
            index += (frames_in_window - overlap_frame_count)

    def execute(self):
        dir_reader = DirectoryReader(self.frame_reader.dataset_dir)
        for participant_folder in dir_reader.get_participant_folders():
            for action_folder in dir_reader.get_action_folders(participant_folder):
                for session_folder in dir_reader.get_session_folders(participant_folder, action_folder):
                    for angle_folder in dir_reader.get_angle_folders(participant_folder, action_folder, session_folder):
                        print(f"Processing clip: {angle_folder}")
                        self.process_clip(os.path.join(participant_folder, action_folder, session_folder, angle_folder))

if __name__ == "__main__":
    model_type=f"model_name_here"   
    try:
        dataset_dir = "E:/MscProject/mini_dataset"
        save_dir = f"../Outputs/2StreamConv_Seq/Unbalanced"

        processor = FlowProcessor(dataset_dir, save_dir)
        processor.run()
        print("------------Prep Complete------------")
        
    except:
        print("Error")
