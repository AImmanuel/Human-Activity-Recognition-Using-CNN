
# Class for reading files in the dataset
class Dataset_Reader:
    def __init__(self, source_folder):
        self.source_folder = source_folder

    def return_subject_folders(self):
        return self.return_subfolders(self.source_folder)

    def return_activity_folders(self, subj_folder):
        sub_path = os.path.join(self.source_folder, subj_folder)
        return self.return_subfolders(sub_path)

    def return_trial_folders(self, subj_folder, act_folder):
        act_path = os.path.join(self.source_folder, subj_folder, act_folder)
        return self.return_subfolders(act_path)

    def return_camera_folders(self, subj_folder, act_folder, tr_folder):
        tr_path = os.path.join(
            self.source_folder, subj_folder, act_folder, tr_folder
        )
        return self.return_subfolders(tr_path)

    def return_subfolders(self, folder):
        folders = [
            d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder, d))
        ]
        return sorted(folders, key=lambda x: int(re.search(r"\d+", x).group()))



# Class for loading individual (video) frames from dataset
class Load_Frames:
    def __init__(self, dataset):
        self.dataset = dataset

    def return_video_folders(self):
        return [
            d
            for d in os.listdir(self.dataset)
            if os.path.isdir(os.path.join(self.dataset, d))
        ]

    def load_video_frames(self, video_folder):
        img_folder = os.path.join(self.dataset, video_folder)
        file_names = sorted(
            [
                f
                for f in os.listdir(img_folder)
                if f.endswith(".jpg") or f.endswith(".png")
            ]
        )

        return [
            (fn[:-4], cv2.imread(os.path.join(img_folder, fn), cv2.IMREAD_GRAYSCALE))
            for fn in file_names
        ]
 

# Preprocessing (Background subtraction, Blurring, Optical Flow estimation and resizing)
class Optical_Flow_BGS:
    def __init__(self):
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=30, detectShadows=True)
        self.fgbg.setShadowValue(0)
        self.fgbg.setShadowThreshold(0.5)
        self.learning_rate = -1
        
    def optical_flow(self, pre_frame, cur_frame):
        pre_frame = cv2.normalize(src=pre_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cur_frame = cv2.normalize(src=cur_frame, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        fg_mask_pre = self.fgbg.apply(pre_frame, learningRate = self.learning_rate)
        fg_mask_cur = self.fgbg.apply(cur_frame, learningRate = self.learning_rate)
        
        kernel = np.ones((4,4), np.uint8)

        fg_mask_pre = cv2.morphologyEx(fg_mask_pre, cv2.MORPH_CLOSE, kernel)
        fg_mask_pre = cv2.morphologyEx(fg_mask_pre, cv2.MORPH_OPEN, kernel)
        fg_mask_cur = cv2.morphologyEx(fg_mask_cur, cv2.MORPH_CLOSE, kernel)
        fg_mask_cur = cv2.morphologyEx(fg_mask_cur, cv2.MORPH_OPEN, kernel)
        
        pre_frame_mask = cv2.bitwise_and(pre_frame, pre_frame, mask=fg_mask_pre)
        cur_frame_mask = cv2.bitwise_and(cur_frame, cur_frame, mask=fg_mask_cur)
          
        # Add Median Blur
        pre_frame = cv2.medianBlur(pre_frame_mask, 5)
        cur_frame = cv2.medianBlur(cur_frame_mask, 5)
        
        # Add Gaussian Blur
        pre_blur = cv2.GaussianBlur(pre_frame, (5, 5), 0)
        cur_blur = cv2.GaussianBlur(cur_frame, (5, 5), 0)
        
        # Farenback Optical Flow estimation 
        opt_flow_FB = cv2.calcOpticalFlowFarneback(pre_blur, cur_blur, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # U and V components 
        u_comp = opt_flow_FB[..., 0]
        v_comp = opt_flow_FB[..., 1]

        # Resize to 51x38 
        resize_51x38_u = cv2.resize(u_comp, (51, 38))
        resize_51x38_v = cv2.resize(v_comp, (51, 38))
        
        return resize_51x38_u, resize_51x38_v
    


# Class to write data into NumPy files
class Write_NumPy:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)
        
    def write_npy_array(self, npy_array, name, folder):
        file_path = os.path.join(self.output_folder, folder, f"{name}.npy")
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        np.save(file_path, npy_array)
        


#Optical Flow processing and timestamp calculations         
class OpticalFlow:
    def __init__(self, dataset, output, fps = 18):
        self.load_frames = Load_Frames(dataset)
        self.write_npy = Write_NumPy(output)
        self.fps = fps
        self.window_size = fps
        self.overlap = fps // 2
        self.optical_flow_bgs = Optical_Flow_BGS()
        
    def secs_from_timestamp(time_stamp: str) -> float:
        hr, min, sec = map(float, time_stamp.split('T')[1].split('_'))
        total_secs = hr * 3600 + min * 60 + sec
        return total_secs
       
    def inc_timestamp(time_stamp: str) -> str:
        date, time = time_stamp.split('T')
        try:
            hr, min, rem = time.split('_')
            sec, ms = rem.split('.')
        except ValueError:
            print(f"Timestamp error at: {time}")
            raise
        
        ms = int(ms)
        sec = int(sec)
        min = int(min)
        hr = int(hr)

        #Count time (hours, minutes and seconds) from timestamps
        ms += 500000    #milliseconds to seconds
        if ms >= 1000000:
            ms -= 1000000
            sec += 1
        
        if sec >= 60:   #seconds to minutes
            sec -= 60
            min += 1
        
        if min >= 60:   #minutes to hours
            min -= 60
            hrs += 1

        time_str = f"{hr:02}_{min:02}_{sec:02}.{ms:06}"
        return f"{date}T{time_str}"
    

    def process_video_opt_flow(self, video, output):
        video_frames = self.load_frames.load_video_frames(video)
        i = 0
        count_frames_in_window = int(self.fps)
        frame_overlap = int(self.overlap)

        last_frame_time = video_frames[-1][0]
        last_frame_secs = OpticalFlow.secs_from_timestamp(last_frame_time)
        timestamp = video_frames[i][0]

        while i < len(video_frames) - count_frames_in_window:
            window_end = min(i + count_frames_in_window, len(video_frames))
            
            OF_u= []
            OF_v = []

            for j in range(i, window_end - 1):
                try:
                    u_comp, v_comp = self.optical_flow_bgs.optical_flow(video_frames[j][1], video_frames[j + 1][1])
                    OF_u.append(u_comp)
                    OF_v.append(v_comp)

                except cv2.error as e:
                    print(f"[OF] Error processing frame: {video_frames[j][0]} in video: {video}. Error: {e}")
                    continue

            OF_u_array = np.stack(OF_u, axis = 0)
            OF_v_array = np.stack(OF_v, axis = 0)
            
            combined_OF_u_v = np.stack([OF_u_array, OF_v_array], axis=-1)
            
            window_name_of = f"{video}_{timestamp}"
            self.write_npy.write_npy_array(combined_OF_u_v, window_name_of, output)

            timestamp = OpticalFlow.inc_timestamp(timestamp)
            next_inc_secs = OpticalFlow.secs_from_timestamp(timestamp)

            if last_frame_secs - next_inc_secs < 1.0:
                break

            i += (count_frames_in_window - frame_overlap)


    def run(self):
        data_reader = Dataset_Reader(self.load_frames.dataset)
        
        for subj_folder in data_reader.return_subject_folders():
            for act_folder in data_reader.return_activity_folders(subj_folder):
                for tr_folder in data_reader.return_trial_folders(subj_folder, act_folder):
                    for camera_folder in data_reader.return_camera_folders(subj_folder, act_folder, tr_folder):
                        print(f"Processing video [OF]: {camera_folder}")
                        start = timer()
                        self.process_video_opt_flow(os.path.join(subj_folder, act_folder, tr_folder, camera_folder), 'OF')
                        print("Time Taken: ", timer() - start)  
                        now = datetime.now()
                        current_time = now.strftime("%H:%M:%S")
                        print("Process Completed at : ", current_time)                      

