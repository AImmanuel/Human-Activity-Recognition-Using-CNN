import cv2
import os
import numpy as np
import csv
import re
from timeit import default_timer as timer
from datetime import datetime

class DatasetDirectoryHandler:
    def __init__(self, base_folder):
        self.base_folder = base_folder

    def get_subject_folders(self):
        return self._get_subfolders(self.base_folder)

    def get_activity_folders(self, subject_folder):
        subject_path = os.path.join(self.base_folder, subject_folder)
        return self._get_subfolders(subject_path)

    def get_trial_folders(self, subject_folder, activity_folder):
        activity_path = os.path.join(self.base_folder, subject_folder, activity_folder)
        return self._get_subfolders(activity_path)

    def get_camera_folders(self, subject_folder, activity_folder, trial_folder):
        trial_path = os.path.join(
            self.base_folder, subject_folder, activity_folder, trial_folder
        )
        return self._get_subfolders(trial_path)

    def _get_subfolders(self, folder):
        folders = [
            d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))
        ]
        return sorted(folders, key=lambda x: int(re.search(r"\d+", x).group()))

class FrameLoader:
    def __init__(self, dataset_folder):
        self.dataset_folder = dataset_folder

    def get_video_folders(self):
        return [
            d
            for d in os.listdir(self.dataset_folder)
            if os.path.isdir(os.path.join(self.dataset_folder, d))
        ]

    def load_frames_from_video(self, video_folder):
        image_folder = os.path.join(self.dataset_folder, video_folder)
        file_names = sorted(
            [
                f
                for f in os.listdir(image_folder)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        )

        return [
            (fn[:-4], cv2.imread(os.path.join(image_folder, fn), cv2.IMREAD_GRAYSCALE))
            for fn in file_names
        ]
 
class OpticalFlowComputer:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=True)
        self.fgbg.setShadowValue(0)
        self.fgbg.setShadowThreshold(0.5)
        self.learning_rate = -1
        
    def compute_optical_flow(self, prev_frame, current_frame):
        prev_frame = cv2.normalize(src=prev_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        current_frame = cv2.normalize(src=current_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        fgmask_prev = self.fgbg.apply(prev_frame, learningRate = self.learning_rate)
        fgmask_curr = self.fgbg.apply(current_frame, learningRate = self.learning_rate)
        
        kernel = np.ones((4,4), np.uint8)
        fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_CLOSE, kernel)
        fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_OPEN, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        
        prev_frame_masked = cv2.bitwise_and(prev_frame, prev_frame, mask=fgmask_prev)
        current_frame_masked = cv2.bitwise_and(current_frame, current_frame, mask=fgmask_curr)
        
        # concatenated_frames = cv2.hconcat([current_frame, current_frame_masked])
        # cv2.imshow('Original vs Preprocessed Frame', concatenated_frames)
        # cv2.waitKey(1)
        
        prev_frame = cv2.medianBlur(prev_frame_masked, 5)
        current_frame = cv2.medianBlur(current_frame_masked, 5)
        
        prev_blurred = cv2.GaussianBlur(prev_frame, (5, 5), 0)
        curr_blurred = cv2.GaussianBlur(current_frame, (5, 5), 0)
        
        flow = cv2.calcOpticalFlowFarneback(prev_blurred, curr_blurred, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u_component = flow[..., 0]
        v_component = flow[..., 1]

        # Uncomment below to view optical flow as it runs
        # magnitude, angle = cv2.cartToPolar(u_component, v_component, angleInDegrees=True)
        # hsv = np.zeros((prev_frame.shape[0], prev_frame.shape[1], 3), dtype=np.uint8)
        # hsv[..., 1] = 255
        # hsv[..., 0] = angle * 180 / np.pi / 2
        # hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # cv2.imshow('Optical Flow', rgb)
        # cv2.waitKey(1)

        #New Resize: 320x240
        resized_u = cv2.resize(u_component, (320, 240))
        resized_v = cv2.resize(v_component, (320, 240))
        
        return resized_u, resized_v
    
    def compute_resize(self, current_frame):
        current_frame = cv2.normalize(
            src=current_frame,
            dst=None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_8U,
        )
        fgmask_curr = self.fgbg.apply(current_frame)
        kernel = np.ones((5, 5), np.uint8)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        current_frame_masked = cv2.bitwise_and(
            current_frame, current_frame, mask=fgmask_curr
        )
        equalized_curr= cv2.equalizeHist(current_frame)
        blurred_img = cv2.blur(equalized_curr,ksize=(5,5))
        med_val = np.median(blurred_img) 
        lower = int(max(0 ,0.5*med_val))
        upper = int(min(255,1.5*med_val))
        current_frame_masked = cv2.Canny(current_frame_masked, lower, upper)
        
        #New Resize: 320x240
        resized_frame = cv2.resize(current_frame_masked, (320, 240))
        
        return resized_frame

class NumpyWriter:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def write_array(self, array, name):
        file_path = os.path.join(self.output_folder, f"{name}.npy")
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        np.save(file_path, array)
        
class OpticalFlowProcessor:
    def __init__(self, dataset_folder, output_folder, fps = 18):
        self.frame_loader = FrameLoader(dataset_folder)
        self.numpy_writer = NumpyWriter(output_folder)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.optical_flow_computer = OpticalFlowComputer()
        
    def total_seconds_from_timestamp(timestamp: str) -> float:
        hours, minutes, seconds = map(float, timestamp.split('T')[1].split('_'))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds
       
    def increment_timestamp(timestamp: str) -> str:
        date, time = timestamp.split('T')
        try:
            hours, minutes, remainder = time.split('_')
            seconds, ms = remainder.split('.')
        except ValueError:
            print(f"Error with timestamp: {time}")
            raise
        
        ms = int(ms)
        seconds = int(seconds)
        minutes = int(minutes)
        hours = int(hours)
        
        ms += 500000
        if ms >= 1000000:
            ms -= 1000000
            seconds += 1
        
        if seconds >= 60:
            seconds -= 60
            minutes += 1
        
        if minutes >= 60:
            minutes -= 60
            hours += 1

        time_str = f"{hours:02}_{minutes:02}_{seconds:02}.{ms:06}"
        return f"{date}T{time_str}"
    
    def process_video(self, video_folder):

#OF        
        frames = self.frame_loader.load_frames_from_video(video_folder)
        #print(f"First frame timestamp for {video_folder}: {frames[0][0]}")
        #print(f"Last frame timestamp for {video_folder}: {frames[-1][0]}")

        i = 0
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)

        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(last_frame_time)
        timestamp = frames[i][0]

        while i < len(frames) - num_frames_in_window:
            window_end = min(i + num_frames_in_window, len(frames))
            
            optical_flows_u= []
            optical_flows_v = []

            #canny_frames = []

            for j in range(i, window_end - 1):
                try:
                    u_component, v_component = self.optical_flow_computer.compute_optical_flow(frames[j][1], frames[j + 1][1])
                    optical_flows_u.append(u_component)
                    optical_flows_v.append(v_component)

                #    final_components = self.optical_flow_computer.compute_resize(frames[k + 1][1])
                #    canny_frames.append(final_components)

                except cv2.error as e:
                    print(f"[OF] Error processing frame {frames[j][0]} from video {video_folder}. Error: {e}")
                    #print(f"[RAW] Error processing frame {frames[k][0]} from video {video_folder}. Error: {e}")
                    continue
            # Commented out to test preprocess
            optical_flows_u_array = np.stack(optical_flows_u, axis = 0)
            optical_flows_v_array = np.stack(optical_flows_v, axis = 0)
            
            combined_optical_flow = np.stack([optical_flows_u_array, optical_flows_v_array], axis=-1)
            #components_stacked = np.stack(canny_frames, axis=0)
            
            window_name_of = f"{video_folder}_{timestamp}"
            #window_name_raw = f"{video_folder}_{timestamp}"

            self.numpy_writer.write_array(combined_optical_flow, window_name_of)
            #self.numpy_writer.write_array(components_stacked, window_name_raw)

            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            #print(f"Incremented timestamp for {video_folder}: {timestamp}")
            next_increment_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(timestamp)

            if last_frame_seconds - next_increment_seconds < 1.0:
                break

            i += (num_frames_in_window - overlap_frames)

#RAW[resize]

        frames = self.frame_loader.load_frames_from_video(video_folder)
        #print(f"First frame timestamp for {video_folder}: {frames[0][0]}")
        #print(f"Last frame timestamp for {video_folder}: {frames[-1][0]}")

        l = 0
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)

        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(last_frame_time)
        timestamp = frames[i][0]

        while l < len(frames) - num_frames_in_window:
            window_end = min(l + num_frames_in_window, len(frames))
            
        #    optical_flows_u= []
        #    optical_flows_v = []

            canny_frames = []

            for k in range(l, window_end - 1):
                try:
                    #u_component, v_component = self.optical_flow_computer.compute_optical_flow(frames[j][1], frames[j + 1][1])
                    #optical_flows_u.append(u_component)
                    #optical_flows_v.append(v_component)

                    final_components = self.optical_flow_computer.compute_resize(frames[k + 1][1])
                    canny_frames.append(final_components)

                except cv2.error as e:
                    #print(f"[OF] Error processing frame {frames[j][0]} from video {video_folder}. Error: {e}")
                    print(f"[RAW] Error processing frame {frames[k][0]} from video {video_folder}. Error: {e}")
                    continue
            # Commented out to test preprocess
            #optical_flows_u_array = np.stack(optical_flows_u, axis = 0)
            #optical_flows_v_array = np.stack(optical_flows_v, axis = 0)
            
            #combined_optical_flow = np.stack([optical_flows_u_array, optical_flows_v_array], axis=-1)
            components_stacked = np.stack(canny_frames, axis=0)
            
            window_name_of = f"{video_folder}_{timestamp}"
            window_name_raw = f"{video_folder}_{timestamp}"

            self.numpy_writer.write_array(combined_optical_flow, window_name_of)
            #self.numpy_writer.write_array(components_stacked, window_name_raw)

            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            #print(f"Incremented timestamp for {video_folder}: {timestamp}")
            next_increment_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(timestamp)

            if last_frame_seconds - next_increment_seconds < 1.0:
                break

            l += (num_frames_in_window - overlap_frames)

    def run(self):
        dir_handler = DatasetDirectoryHandler(self.frame_loader.dataset_folder)
        
        for subject_folder in dir_handler.get_subject_folders():
            for activity_folder in dir_handler.get_activity_folders(subject_folder):
                for trial_folder in dir_handler.get_trial_folders(subject_folder, activity_folder):
                    for camera_folder in dir_handler.get_camera_folders(subject_folder, activity_folder, trial_folder):
                        print(f"Processing video: {camera_folder}")
                        start = timer()
                        self.process_video(os.path.join(subject_folder, activity_folder, trial_folder, camera_folder))
                        print("Time Taken: ", timer() - start)  
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print("Process Completed at : ", current_time)                      


if __name__ == "__main__":
    model_type=f"2StreamConv_Seq"   
    try:
        dataset_folder = "E:/MscProject/UP-Fall"
        output_folder = f"../Outputs/2StreamConv_Seq/Unbalanced"

        processor = OpticalFlowProcessor(dataset_folder, output_folder)
        processor.run()
        print("------------Processing Complete------------")
        
    except:
        print("~~~~~~~~Error~~~~~~~~")
