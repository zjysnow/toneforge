import subprocess
import threading
import queue
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from typing import Optional
from tqdm import tqdm

import toneforge.color_space as color

@dataclass
class ColorParams:
    range:str = "limited" # limited 
    hdr10_opt:Optional[str] = None
    transfer:Optional[str] = None
    color_matrix:Optional[str] = None
    color_prim:Optional[str] = None

    def __str__(self) -> str:
        return f"log-level=error:range={self.range}:hdr10-opt={self.hdr10_opt}:transfer={self.transfer}:colorprim={self.color_prim}:colormatrix={self.color_matrix}"

    def get(self)->tuple:
        GAMUT_MAP = {
            "bt709": color.BT709,
            "bt2020": color.BT2020,
            "bt2020nc": color.BT2020,
            "bt2020c": color.BT2020,
            "smpte170m": color.BT601_525,
            "bt470bg": color.BT601_625,
        }
        gamut = GAMUT_MAP.get(self.color_matrix, color.BT709)
        white_point = color.WhitePoint[color.D65]
        is_narrow = self.range == "limited"
        return gamut, white_point, is_narrow

    @classmethod
    def SDR(cls)->ColorParams:
        return cls(
            range="limited",
            hdr10_opt="false",
            transfer="bt709",
            color_matrix="bt709",
            color_prim="bt709"
        )

    @classmethod
    def PQ(cls)->ColorParams:
        return cls(
            range="limited",
            hdr10_opt="true",
            transfer="smpte2084",
            color_matrix="bt2020nc",
            color_prim="bt2020"
        )

    @classmethod
    def HLG(cls)->ColorParams:
        return cls(
            range="limited",
            hdr10_opt="true",
            transfer="arib-std-b67",
            color_matrix="bt2020nc",
            color_prim="bt2020"
        )

class BaseVideoProcess(ABC):
    def __init__(self, input_file:str, width:int, height:int, input_color: ColorParams, 
                 output_file:str, rate:int = 30, output_color: Optional[ColorParams] = None, max_queue_size:int = 8):
        self.input_file = input_file
        self.output_file = output_file

        self.width = width
        self.height = height

        self.input_color = input_color
        self.output_color = input_color if output_color is None else output_color

        self.rate = rate

        self.pix_fmt = "yuv444p10le"
        self.dtype = np.uint16

        self.bufsize = width * height * 3 * 2
        # self.y_size = width * height
        # self.uv_size = self.y_size // 4

        self.M_rgb2yuv, self.O_rgb2yuv = color.getMatrixRGB2YUV(*self.output_color.get(), weight_bits=12, offset_bits=10)
        self.M_yuv2rgb, self.O_yuv2rgb = color.getMatrixYUV2RGB(*self.input_color.get(), weight_bits=12, offset_bits=10)

        self.input_queue = queue.Queue(maxsize=max_queue_size)
        self.output_queue = queue.Queue(maxsize=max_queue_size)

    @abstractmethod
    def process_frame(self, rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError


    
    def _start_subprocess(self):
        decoder_cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-i", self.input_file,
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", self.pix_fmt,
            "-f", "rawvideo",
            "pipe:1"
        ]
        self.decoder_process = subprocess.Popen(decoder_cmd, stdout=subprocess.PIPE)

        encoder_cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-s", f"{self.width}x{self.height}",
            "-f", "rawvideo",
            "-pix_fmt", self.pix_fmt,
            "-i", "pipe:0",
            "-pix_fmt", "yuv420p10le",
            "-c:v", "libx265",
            "-tag:v", "hvc1",
            "-x265-params", f"{self.output_color}",
            "-r", f"{self.rate}",
            self.output_file
        ]
        self.encoder_process = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE)

    def _read_thread(self):
        try:
            while True:
                buffer = self.decoder_process.stdout.read(self.bufsize)
                if not buffer or len(buffer) != self.bufsize:
                    break

                data = np.frombuffer(buffer, dtype=self.dtype)
                yuv = data.reshape(3, self.height, self.width).transpose(1,2,0)

                # Y = data[:self.y_size].reshape(self.height, self.width)
                # U = cv2.resize(data[self.y_size : self.y_size + self.uv_size].reshape(self.height // 2, self.width // 2), 
                #             (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                # V = cv2.resize(data[self.y_size + self.uv_size:].reshape(self.height // 2, self.width // 2), 
                #             (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                # yuv = np.stack([Y,U,V], axis = -1)
                rgb = np.clip(((yuv @ self.M_yuv2rgb.T + 2048) >> 12) + self.O_yuv2rgb, 0, 1023).astype(self.dtype)

                self.input_queue.put(rgb)
        finally:
            self.input_queue.put(None)
            self.decoder_process.stdout.close()

    def _write_thread(self):
        try:
            while True:
                rgb = self.output_queue.get()
                if rgb is None:
                    break

                yuv = np.clip(((rgb @ self.M_rgb2yuv.T + 2048) >> 12) + self.O_rgb2yuv, 0, 1023).astype(self.dtype)
                data = yuv.transpose(2,0,1).ravel()
                
                # Y = yuv[:,:,0].ravel()
                # # U = cv2.resize(yuv[:,:,1], (self.width//2, self.height//2), interpolation=cv2.INTER_LINEAR).ravel()
                # # V = cv2.resize(yuv[:,:,2], (self.width//2, self.height//2), interpolation=cv2.INTER_LINEAR).ravel()
                # U = yuv[::2,::2,1].ravel()
                # V = yuv[::2,::2,2].ravel()

                # data = np.concatenate([Y,U,V])
                
                self.encoder_process.stdin.write(data.tobytes())
                self.output_queue.task_done()
        finally:
            self.encoder_process.stdin.close()

    def run(self):
        self._start_subprocess()

        reader = threading.Thread(target=self._read_thread, daemon=True)
        writer = threading.Thread(target=self._write_thread, daemon=True)

        reader.start()
        writer.start()

        try:
            with tqdm(desc="Processing Stream", unit=" frames") as pbar:
                while True:
                    rgb = self.input_queue.get()
                    if rgb is None:
                        break
                    rgb = self.process_frame(rgb)
                    self.output_queue.put(rgb)

                    pbar.update(1)
        finally:
            self.output_queue.put(None)

            reader.join()
            writer.join()

            self.decoder_process.wait()
            self.encoder_process.wait()
            # if self.decoder_process.poll() is None:
            #     self.decoder_process.kill()
            # if self.encoder_process.poll() is None:
            #     self.encoder_process.kill()


if __name__ == "__main__":
    import toneforge.color_space as color
    import toneforge.transfer_funcs.eotf as EOTF
    import toneforge.transfer_funcs.oetf as OETF

    M2020 = color.getMatrixRGB2XYZ(color.BT2020, color.WhitePoint[color.D65])
    M709 = color.getMatrixRGB2XYZ(color.BT709, color.WhitePoint[color.D65])
    M = np.linalg.inv(M709) @ M2020

    M_yuv2rgb, O_yuv2rgb = color.getMatrixYUV2RGB(color.BT2020, color.WhitePoint[color.D65], is_narrow=True, weight_bits=12, offset_bits=10)
    M_rgb2yuv, O_rgb2yuv = color.getMatrixRGB2YUV(color.BT709, color.WhitePoint[color.D65], is_narrow=True, weight_bits=12, offset_bits=10)

    class VideoProcess(BaseVideoProcess):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def process_frame(self, rgb):
            linear = EOTF.HLG(rgb / 1023.0)
            linear = np.clip(linear @ M.T, 0, 1)
            rgb = OETF.sRGB(linear) * 1023
            return np.clip(rgb, 0, 1023).astype(np.uint16)

        
    params = {
        "input_file": "./data/IMG_0161.MOV",
        "output_file": "./data/test.mp4",
        "width": 1080,
        "height": 1920,
        "input_color": ColorParams.HLG(),
        "output_color": ColorParams.SDR()
    }

    processor = VideoProcess(**params)
    processor.run()

