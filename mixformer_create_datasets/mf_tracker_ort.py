import os
import sys
import argparse
import numpy as np
import time
import cv2
import glob
import onnxruntime
import numpy as np
import math
import random

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

def get_total_frames(video_name):
    """获取视频总帧数

    Args:
        video_name (_type_): 视频路径

    Returns:
        int: 总帧数，如果无法获取则返回-1
    """
    if not video_name:
        return -1  # RTSP流无法获取总帧数
    elif video_name.endswith('avi') or video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total_frames
    else:
        # 图像序列
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        return len(images)

def get_frames(video_name):
    """获取视频帧

    Args:
        video_name (_type_): _description_

    Yields:
        _type_: _description_
    """
    if not video_name:
        rtsp = "rtsp://%s:%s@%s:554/cam/realmonitor?channel=1&subtype=1" % ("admin", "123456", "192.168.1.108")
        cap = cv2.VideoCapture(rtsp) if rtsp else cv2.VideoCapture()
        
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                # print('读取成功===>>>', frame.shape)
                yield cv2.resize(frame,(800, 600))
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame

class Preprocessor_wo_mask(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1)).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1)).astype(np.float32)

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = np.ascontiguousarray(img_arr.transpose((2, 0, 1))[np.newaxis].astype(np.float32))
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return np.ascontiguousarray(img_tensor_norm)

class MFTrackerORT:
    def __init__(self, model_path, fp16=False) -> None:
        self.debug = True
        self.gpu_id = 0
        self.providers = ["CUDAExecutionProvider"]
        self.provider_options = [{"device_id": str(self.gpu_id)}]
        self.model_path = model_path
        self.fp16 = fp16
        
        self.init_track_net()
        self.preprocessor = Preprocessor_wo_mask()
        self.max_score_decay = 1.0
        self.search_factor = 2.5
        self.search_size = 224
        self.template_factor = 2.0
        self.template_size = 112
        self.update_interval = 25
        self.online_size = 1

    def init_track_net(self):
        """使用设置的参数初始化tracker网络
        """
        self.ort_session = onnxruntime.InferenceSession(self.model_path, providers=self.providers, provider_options=self.provider_options)

    def track_init(self, frame, target_pos=None, target_sz = None):
        """使用第一帧进行初始化

        Args:
            frame (_type_): _description_
            target_pos (_type_, optional): _description_. Defaults to None.
            target_sz (_type_, optional): _description_. Defaults to None.
        """
        self.trace_list = []
        try:
            # [x, y, w, h]
            init_state = [target_pos[0], target_pos[1], target_sz[0], target_sz[1]]
            z_patch_arr, _, z_amask_arr = self.sample_target(frame, init_state, self.template_factor, output_sz=self.template_size)
            template = self.preprocessor.process(z_patch_arr)
            self.template = template
            self.online_template = template

            self.online_state = init_state
            self.online_image = frame
            self.max_pred_score = -1.0
            self.online_max_template = template
            self.online_forget_id = 0

            # save states
            self.state = init_state
            self.frame_id = 0
            print(f"第一帧初始化完毕！")
        except:
            print(f"第一帧初始化异常！")
            exit()

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = self.sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        # compute ONNX Runtime output prediction
        ort_inputs = {'img_t': self.to_numpy(self.template), 'img_ot': self.to_numpy(self.online_template), 'img_search': self.to_numpy(search)}

        ort_outs = self.ort_session.run(None, ort_inputs)

        # print(f">>> lenght trt_outputs: {ort_outs}")
        pred_boxes = ort_outs[0]
        pred_score = ort_outs[1]
        # print(f">>> box and score: {pred_boxes}  {pred_score}")
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(axis=0) * self.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = self.clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.max_pred_score = self.max_pred_score * self.max_score_decay
        # update template
        if pred_score > 0.5 and pred_score > self.max_pred_score:
            z_patch_arr, _, z_amask_arr = self.sample_target(image, self.state,
                                                        self.template_factor,
                                                        output_sz=self.template_size)  # (x1, y1, w, h)
            self.online_max_template = self.preprocessor.process(z_patch_arr)
            self.max_pred_score = pred_score

        
        if self.frame_id % self.update_interval == 0:
            if self.online_size == 1:
                self.online_template = self.online_max_template
            else:
                self.online_template[self.online_forget_id:self.online_forget_id+1] = self.online_max_template
                self.online_forget_id = (self.online_forget_id + 1) % self.online_size

            self.max_pred_score = -1
            self.online_max_template = self.template

        # 跟踪框绘制逻辑已移到主循环中

        return {"target_bbox": self.state, "conf_score": pred_score}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: np.ndarray, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box[..., 0], pred_box[..., 1], pred_box[..., 2], pred_box[..., 3] # (N,4) --> (N,)
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return np.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], axis=-1)
    
    def to_numpy(self, tensor):
        if self.fp16:
            return tensor.astype(np.float16) if isinstance(tensor, np.ndarray) else np.array(tensor, dtype=np.float16)
        return tensor if isinstance(tensor, np.ndarray) else np.array(tensor)
    
    def sample_target(self, im, target_bb, search_area_factor, output_sz=None, mask=None):
        """ Extracts a square crop centered at target_bb box, of area search_area_factor^2 times target_bb area

        args:
            im - cv image
            target_bb - target box [x, y, w, h]
            search_area_factor - Ratio of crop size to target size
            output_sz - (float) Size to which the extracted crop is resized (always square). If None, no resizing is done.

        returns:
            cv image - extracted crop
            float - the factor by which the crop has been resized to make the crop size equal output_size
        """
        if not isinstance(target_bb, list):
            x, y, w, h = target_bb.tolist()
        else:
            x, y, w, h = target_bb
        # Crop image
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

        if crop_sz < 1:
            raise Exception('Too small bounding box.')

        x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
        x2 = int(x1 + crop_sz)

        y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
        y2 = int(y1 + crop_sz)

        x1_pad = int(max(0, -x1))
        x2_pad = int(max(x2 - im.shape[1] + 1, 0))

        y1_pad = int(max(0, -y1))
        y2_pad = int(max(y2 - im.shape[0] + 1, 0))

        # Crop target
        im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]
        if mask is not None:
            mask_crop = mask[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad]

        # Pad
        im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)
        # deal with attention mask
        H, W, _ = im_crop_padded.shape
        att_mask = np.ones((H,W))
        end_x, end_y = -x2_pad, -y2_pad
        if y2_pad == 0:
            end_y = None
        if x2_pad == 0:
            end_x = None
        att_mask[y1_pad:end_y, x1_pad:end_x] = 0
        if mask is not None:
            mask_crop_padded = np.pad(mask_crop, pad=((y1_pad, y2_pad), (x1_pad, x2_pad)), mode='constant', constant_values=0)


        if output_sz is not None:
            resize_factor = output_sz / crop_sz
            im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
            att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
            if mask is None:
                return im_crop_padded, resize_factor, att_mask
            mask_crop_padded = cv2.resize(mask_crop_padded, (output_sz, output_sz), interpolation=cv2.INTER_LINEAR)
            return im_crop_padded, resize_factor, att_mask, mask_crop_padded

        else:
            if mask is None:
                return im_crop_padded, att_mask.astype(np.bool_), 1.0
            return im_crop_padded, 1.0, att_mask.astype(np.bool_), mask_crop_padded
        
    def clip_box(self, box: list, H, W, margin=0):
        x1, y1, w, h = box
        x2, y2 = x1 + w, y1 + h
        x1 = min(max(0, x1), W-margin)
        x2 = min(max(margin, x2), W)
        y1 = min(max(0, y1), H-margin)
        y2 = min(max(margin, y2), H)
        w = max(margin, x2-x1)
        h = max(margin, y2-y1)
        return [x1, y1, w, h]
        
if __name__ == '__main__':
    print("测试")
    model_path = "mixformer_v2_sim_fp16.onnx"
    Tracker = MFTrackerORT(model_path = model_path, fp16=False)
    first_frame = True
    display_only_mode = False  # 仅显示模式，跳过跟踪
    paused_mode = False  # 暂停模式，结束跟踪
    video_path = r"e:\20251205-2.mp4"
    win_name = "demo_mf_tracker_ort"
    
    # 创建输出目录
    output_dir = "20251129-6666-player2"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)  # 允许窗口大小可调

    # Market1501格式参数
    person_id = "0006"  # 人员ID
    camera_id = "c1"    # 摄像头ID
    frame_id = 0
    total_time = 0
    
    # 生成随机数用于文件名，防止重名（6位随机数）
    random_id = random.randint(100000, 999999)
    print(f"本次运行的文件名随机数: {random_id}")
    
    # 获取总帧数
    total_frames = get_total_frames(video_path)
    if total_frames > 0:
        print(f"视频总帧数: {total_frames}")
    else:
        print("无法获取总帧数（可能是RTSP流或无法读取）")
    
    for frame in get_frames(video_path):
        print(f"frame shape {frame.shape}")
        tic = cv2.getTickCount()
        
        # 显示当前帧并检查键盘事件（在跟踪前先显示，用于第一帧选择）
        if first_frame:
            # 在图像上绘制帧数信息
            display_frame = frame.copy()
            if total_frames > 0:
                frame_text = f"Frame: {frame_id}/{total_frames}"
            else:
                frame_text = f"Frame: {frame_id}"
            
            # 在左上角绘制帧数信息（白色文字，黑色背景）
            text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x, text_y = 10, 30
            # 绘制背景矩形
            cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            # 绘制文字
            cv2.putText(display_frame, frame_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(win_name, display_frame)
        
        # 处理键盘事件
        key = cv2.waitKey(1) & 0xFF
        
        # 按 'q' 暂停跟踪（可以多次按q进入暂停模式）
        if key == ord('q') and not first_frame:
            if not paused_mode:
                print("暂停跟踪，已结束跟踪。按'w'显示下一帧，按'r'重新画框恢复跟踪，按'q'继续暂停")
                paused_mode = True
                display_only_mode = True  # 进入仅显示模式，结束跟踪
        
        # 如果处于暂停模式，进入等待循环（支持多次交互）
        if paused_mode and not first_frame:
            # 在图像上绘制帧数信息
            display_frame = frame.copy()
            if total_frames > 0:
                frame_text = f"Frame: {frame_id}/{total_frames}"
            else:
                frame_text = f"Frame: {frame_id}"
            
            # 在左上角绘制帧数信息（白色文字，黑色背景）
            text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x, text_y = 10, 30
            # 绘制背景矩形
            cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5), 
                         (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
            # 绘制文字
            cv2.putText(display_frame, frame_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 显示当前帧
            cv2.imshow(win_name, display_frame)
            # 在暂停模式下循环等待用户按键（支持多次交互）
            while paused_mode:
                wait_key = cv2.waitKey(0) & 0xFF
                if wait_key == ord('w'):
                    # 按 'w' 显示下一帧（可以多次按a）
                    frame_id += 1
                    print(f"暂停状态：按'w'键显示第 {frame_id} 帧（不跟踪）")
                    break  # 退出等待循环，进入下一帧
                elif wait_key == ord('r'):
                    # 按 'r' 重新画框（可以多次按r重新选择）
                    print("重新画框，请选择目标（框选时按ESC可取消）...")
                    x, y, w, h = cv2.selectROI(win_name, frame, fromCenter=False)
                    if w > 0 and h > 0:  # 确保用户选择了有效的区域
                        target_pos = [x, y]
                        target_sz = [w, h]
                        
                        # 保存重新选择的目标图像（在绘制之前）
                        H, W, _ = frame.shape
                        x1, y1 = max(0, int(x)), max(0, int(y))
                        x2, y2 = min(W, int(x + w)), min(H, int(y + h))
                        
                        if x2 > x1 and y2 > y1:  # 确保裁剪区域有效
                            reinit_target = frame[y1:y2, x1:x2]
                            # Market1501格式: {person_id}_{camera_id}_{frame_id}_{random_id}_{bbox_id}.jpg
                            # bbox_id=01 表示重新选择的目标
                            save_path = os.path.join(output_dir, f"{person_id}_{camera_id}_{frame_id:06d}_{random_id}_01.jpg")
                            cv2.imwrite(save_path, reinit_target)
                            print(f"保存重新选择的目标图像: {save_path}")
                        
                        # 在保存后再绘制重新选择的框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                        
                        # 重新初始化跟踪器
                        Tracker.track_init(frame, target_pos, target_sz)
                        paused_mode = False  # 退出暂停模式
                        display_only_mode = False  # 恢复正常跟踪
                        first_frame = False
                        print("重新选择目标完成，恢复跟踪")
                        # 在图像上绘制帧数信息
                        display_frame = frame.copy()
                        if total_frames > 0:
                            frame_text = f"Frame: {frame_id}/{total_frames}"
                        else:
                            frame_text = f"Frame: {frame_id}"
                        
                        # 在左上角绘制帧数信息（白色文字，黑色背景）
                        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        text_x, text_y = 10, 30
                        # 绘制背景矩形
                        cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5), 
                                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
                        # 绘制文字
                        cv2.putText(display_frame, frame_text, (text_x, text_y), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # 显示绘制了框的帧
                        cv2.imshow(win_name, display_frame)
                        # 退出等待循环，继续执行后续跟踪逻辑
                        break
                    else:
                        print("取消重新选择，继续暂停状态")
                elif wait_key == ord('q'):
                    # 再次按'q'，继续暂停状态（可以多次按q）
                    print("继续暂停状态，按'w'显示下一帧，按'r'重新画框恢复跟踪")
                    # 继续while循环，等待下一个按键
                    continue
                elif wait_key == 27:  # ESC键
                    print("退出程序")
                    cv2.destroyAllWindows()
                    exit()
            # 如果按了'w'，continue跳过当前帧处理；如果按了'r'恢复跟踪，继续执行后续逻辑
            if paused_mode:
                continue
        
        # 按 'w' 跳帧（非暂停模式下）
        if key == ord('w') and not paused_mode:
            frame_id += 1
            print(f"按'w'键跳过第 {frame_id} 帧")
            continue  # 跳过当前帧的处理
        
        if first_frame:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
            print("第一帧：按'w'键跳过当前帧，或开始框选目标区域（框选时按ESC可取消并跳过）")
            
            # 在框选前先检查是否有按键（给用户机会跳过）
            key = cv2.waitKey(100) & 0xFF  # 100ms超时
            
            if key == ord('w'):
                frame_id += 1
                print(f"按'w'键跳过第 {frame_id} 帧（第一帧），进入仅显示模式")
                display_only_mode = True  # 设置为仅显示模式
                first_frame = False  # 标记第一帧已处理
                continue
            elif key == 27:  # ESC键
                print("退出程序")
                break
            
            # 调用selectROI让用户框选（用户可以在框选时按ESC取消）
            x, y, w, h = cv2.selectROI(win_name, frame)

            # 验证ROI选择是否有效（用户按ESC或取消框选会返回(0,0,0,0)）
            if w <= 0 or h <= 0:
                print("未选择有效目标，跳过当前帧")
                display_only_mode = True  # 设置为仅显示模式
                first_frame = False  # 标记第一帧已处理
                continue

            target_pos = [x, y]
            target_sz = [w, h]
            print('====================type=================', target_pos, type(target_pos), type(target_sz))
            
            # 保存初始目标图像（在绘制之前）
            H, W, _ = frame.shape
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            
            if x2 > x1 and y2 > y1:  # 确保裁剪区域有效
                initial_target = frame[y1:y2, x1:x2]
                # Market1501格式: {person_id}_{camera_id}_{frame_id}_{random_id}_{bbox_id}.jpg
                # frame_id=000000, bbox_id=00 表示初始目标
                save_path = os.path.join(output_dir, f"{person_id}_{camera_id}_000000_{random_id}_00.jpg")
                cv2.imwrite(save_path, initial_target)
                print(f"保存初始目标图像: {save_path}")
            
            # 在保存后再绘制初始框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            
            Tracker.track_init(frame, target_pos, target_sz)
            first_frame = False
        else:
            if paused_mode:
                # 暂停模式：已经在上面处理了，这里不应该执行到
                pass
            elif display_only_mode:
                # 仅显示模式，不进行跟踪
                frame_id += 1
                print(f"显示模式：跳过第 {frame_id} 帧的跟踪")
            else:
                # 正常跟踪模式
                state = Tracker.track(frame)
                frame_id += 1
                
                # 获取跟踪框坐标
                x1, y1, w, h = state["target_bbox"]
                x1, y1, w, h = int(x1), int(y1), int(w), int(h)
                
                # 确保裁剪区域在图像范围内
                H, W, _ = frame.shape
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x1 + w)
                y2 = min(H, y1 + h)
                
                # 先保存ROI区域（在绘制框之前，避免将框也截取下来）
                if x2 > x1 and y2 > y1:  # 确保裁剪区域有效
                    target_crop = frame[y1:y2, x1:x2]
                    # Market1501格式: {person_id}_{camera_id}_{frame_id}_{random_id}_{bbox_id}.jpg
                    # bbox_id=00 表示正常跟踪的目标
                    save_path = os.path.join(output_dir, f"{person_id}_{camera_id}_{frame_id:06d}_{random_id}_00.jpg")
                    cv2.imwrite(save_path, target_crop)
                
                # 在保存跟踪区域后再绘制跟踪框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
        
        # 在图像上绘制帧数信息
        display_frame = frame.copy()
        if total_frames > 0:
            frame_text = f"Frame: {frame_id}/{total_frames}"
        else:
            frame_text = f"Frame: {frame_id}"
        
        # 在左上角绘制帧数信息（白色文字，黑色背景）
        text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x, text_y = 10, 30
        # 绘制背景矩形
        cv2.rectangle(display_frame, (text_x - 5, text_y - text_size[1] - 5), 
                     (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
        # 绘制文字
        cv2.putText(display_frame, frame_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 显示当前帧（包含绘制的跟踪框和帧数信息）
        if not first_frame:
            cv2.imshow(win_name, display_frame)

        toc = cv2.getTickCount() - tic
        toc = int(1 / (toc / cv2.getTickFrequency()))
        total_time += toc
        print('Video: {:12s} {:3.1f}fps'.format('tracking', toc))
    
    print('video: average {:12s} {:3.1f} fps'.format('finale average tracking fps', total_time/(frame_id - 1)))
    cv2.destroyAllWindows()


# 按 'q' 暂停跟踪
# 暂停状态下按 'w'：跳过当前帧，进入下一帧
# 按 'r' 恢复跟踪：需要重新选择目标，然后恢复跟踪

# 按r进入画框时，按esc退出画框