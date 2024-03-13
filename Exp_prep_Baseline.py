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
        self.fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.fgbg.setShadowValue(0)
        self.fgbg.setShadowThreshold(0.5)

    # def concat_tile_resize(list_2d,  interpolation = cv2.INTER_CUBIC):
    #    img_list_v = [hconcat_resize(list_h,
    #                             interpolation = cv2.INTER_CUBIC)
    #              for list_h in list_2d]

    #    return vconcat_resize(img_list_v, interpolation=cv2.INTER_CUBIC)

    def compute_edge_detector(self, current_frame):
        resized_frame = cv2.resize(current_frame, (51, 38))
        return resized_frame

################################down#################################
    def compute_optical_flow(self, prev_frame, current_frame):
        prev_frame = cv2.normalize(src=prev_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        current_frame = cv2.normalize(src=current_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
         # Apply Gaussian blur to reduce noise
        equalized_curr= cv2.equalizeHist(current_frame)
        equalized_prev= cv2.equalizeHist(prev_frame)
        
        blurred = cv2.GaussianBlur(equalized_curr , (5, 5), 0)
        blurred_prev = cv2.GaussianBlur(equalized_prev , (5, 5), 0)

        med_val = np.median(blurred) 
        lower = int(max(0 ,0.5*med_val))
        upper = int(min(255,1.5*med_val))


        edges_new=cv2.Canny(blurred,lower, upper)
        edges_new_prev=cv2.Canny(blurred_prev,lower, upper)
        
        ret, thresh = cv2.threshold(edges_new, 150, 255, cv2.THRESH_BINARY)
        ret_prev, thresh_prev = cv2.threshold(edges_new_prev, 150, 255, cv2.THRESH_BINARY)

        contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1)
        contours_prev , hierarchy_prev = cv2.findContours(image=thresh_prev, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_TC89_L1)
        
        image = current_frame.copy()
        image_prev = prev_frame.copy()

        cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.drawContours(image=image_prev, contours=contours_prev, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

        fgmask_prev = self.fgbg.apply(image)
        fgmask_curr = self.fgbg.apply(image_prev)
        
        kernel = np.ones((5,5), np.uint8)
        fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_OPEN, kernel)
        fgmask_prev = cv2.morphologyEx(fgmask_prev, cv2.MORPH_CLOSE, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_OPEN, kernel)
        fgmask_curr = cv2.morphologyEx(fgmask_curr, cv2.MORPH_CLOSE, kernel)
        
        prev_frame_masked = cv2.bitwise_and(prev_frame, prev_frame, mask=fgmask_prev)
        current_frame_masked = cv2.bitwise_and(current_frame, current_frame, mask=fgmask_curr)
        
        prev_frame = cv2.medianBlur(prev_frame_masked, 5)
        current_frame = cv2.medianBlur(current_frame_masked, 5)
        
        prev_blurred = cv2.GaussianBlur(prev_frame, (5, 5), 0)
        curr_blurred = cv2.GaussianBlur(current_frame, (5, 5), 0)
        
        flow = cv2.calcOpticalFlowFarneback(prev_blurred, curr_blurred, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        u_component = flow[..., 0]
        v_component = flow[..., 1]
        resized_u = cv2.resize(u_component, (51, 38))
        resized_v = cv2.resize(v_component, (51, 38))
        
        return resized_u, resized_v

###################################up#################################3


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
    def __init__(self, dataset_folder, output_folder, fps=18):
        self.frame_loader = FrameLoader(dataset_folder)
        self.numpy_writer = NumpyWriter(output_folder)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.optical_flow_computer = OpticalFlowComputer()

    def total_seconds_from_timestamp(timestamp: str) -> float:
        hours, minutes, seconds = map(float, timestamp.split("T")[1].split("_"))
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds

    def increment_timestamp(timestamp: str) -> str:
        date, time = timestamp.split("T")
        try:
            hours, minutes, remainder = time.split("_")
            seconds, ms = remainder.split(".")
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
        frames = self.frame_loader.load_frames_from_video(video_folder)
        # print(f"First frame timestamp for {video_folder}: {frames[0][0]}")
        # print(f"Last frame timestamp for {video_folder}: {frames[-1][0]}")

        i = 0
        num_frames_in_window = int(self.fps)
        overlap_frames = int(self.overlap)

        last_frame_time = frames[-1][0]
        last_frame_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(
            last_frame_time
        )
        timestamp = frames[i][0]

        while i < len(frames) - num_frames_in_window:
            window_end = min(i + num_frames_in_window, len(frames))
            optical_flows_u= []
            optical_flows_v = []
            canny_frames = []
            for j in range(i, window_end - 1):
                try:
                    final_components = self.optical_flow_computer.compute_edge_detector(
                        frames[j + 1][1]
                    )
                    canny_frames.append(final_components)
                    u_component, v_component = self.optical_flow_computer.compute_optical_flow(frames[j][1], frames[j + 1][1])
                    optical_flows_u.append(u_component)
                    optical_flows_v.append(v_component)
                except cv2.error as e:
                    print(
                        f"Error processing frame {frames[j][0]} from video {video_folder}. Error: {e}"
                    )
                    continue
            # Commented out to test preprocess
            components_stacked = np.stack(canny_frames, axis=0)
            optical_flows_u_array = np.stack(optical_flows_u, axis = 0)
            optical_flows_v_array = np.stack(optical_flows_v, axis = 0)

            combined_optical_flow = np.stack([optical_flows_u_array, optical_flows_v_array], axis=-1)
            window_name = f"{video_folder}_{timestamp}"
            #######################################
            ##########check this one###############
            #######################################
            self.numpy_writer.write_array(components_stacked+combined_optical_flow, window_name)

            timestamp = OpticalFlowProcessor.increment_timestamp(timestamp)
            # print(f"Incremented timestamp for {video_folder}: {timestamp}")
            next_increment_seconds = OpticalFlowProcessor.total_seconds_from_timestamp(timestamp)

            if last_frame_seconds - next_increment_seconds < 1.0:
                break

            i += num_frames_in_window - overlap_frames

    def run(self):
        dir_handler = DatasetDirectoryHandler(self.frame_loader.dataset_folder)

        logging_output(f"# Data Process starts for {output_folder} ##",log_path)
        for subject_folder in dir_handler.get_subject_folders():
            for activity_folder in dir_handler.get_activity_folders(subject_folder):
                for trial_folder in dir_handler.get_trial_folders(subject_folder, activity_folder):
                    for camera_folder in dir_handler.get_camera_folders(subject_folder, activity_folder, trial_folder):
                        print(f"Processing video: {camera_folder}")
                        logging_output(f"Processing video: {camera_folder},log_path")
                        start = timer()
                        self.process_video(os.path.join(subject_folder,activity_folder,trial_folder,camera_folder,))
                        print("Time Taken: ", timer() - start)
                        logging_output(f"Time Taken: {timer() - start}",log_path)
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print("Process Completed at : ", current_time)

def logging_output(message, file_path=f'../Outputs/prep_log.txt'):
    try:
    # Write to file
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        with open(file_path, 'a') as file:
            file.write(f'{current_time} : {message} \n')
    except Exception as e:
        print(f"Error logging: {e}")    

if __name__ == "__main__":                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
    try:
        script_path = os.path.abspath(__file__)
         # Extract the file name from the path
        script_name = os.path.basename(script_path)
        name_only=os.path.splitext(script_name)
        model_folder=name_only[0]
        model_type =model_folder

        log_path=f"../Outputs/{model_type}/prep_log.log"
        dataset_folder = "E:/MscProject/mini_dataset"
        output_folder = f"../Outputs/{model_type}/Unbalanced"

        processor = OpticalFlowProcessor(dataset_folder, output_folder)
        processor.run()
        print("###############Data Process Complete############")
        logging_output(f"# Data Process Complete for {output_folder} ##",log_path)

        print("~~~~~~~~Hibernating~~~~~~~~~~~~~")
        # os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")
    except Exception as e:
        print(f"~~~~~~~~Error Occurred~~~~~~~~{e}")
