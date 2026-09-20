import numpy as np
from toneforge.texture.shear import ShearMethod
from toneforge.texture.rotate import RotateMethod

from dataclasses import dataclass

CoordinateMappingMethod = RotateMethod | ShearMethod

@dataclass(frozen=True)
class PipelineConfig:
    method: CoordinateMappingMethod
    scale: tuple[int, int]
    translation: tuple[int, int]
    threshold: int

@dataclass(frozen=True)
class ResampleConfig:
    n: int
    flip: bool
    jitter: bool
    resample: tuple[tuple[int, int],...]

    def __post_init__(self):
        if self.n < 0 or self.n > 2:
            raise ValueError("n must be a positive integer and at most 2")
        expected_length = 1 << self.n
        if len(self.resample) != expected_length:
            raise ValueError(f"resample must have length {expected_length}")
        for coor in self.resample:
            if len(coor) != 2:
                raise ValueError("each resample coordinate must be a tuple of length 2")
            if not all(isinstance(x, int) for x in coor):
                raise ValueError("each resample coordinate must contain integers")
        

@dataclass(frozen=True)
class RandomTextureConfig:
    texture_table: np.ndarray
    pipeline: tuple[PipelineConfig,...]
    pipeline_count: int
    resample: ResampleConfig

    def __post_init__(self):
        if self.pipeline_count <0 or self.pipeline_count > 4:
            raise ValueError("pipeline_count must be a positive integer and at most 4")
        expected_length = 1 << self.pipeline_count
        if len(self.pipeline) != expected_length:
            raise ValueError(f"pipeline must have length {expected_length}")


class RandomTexture:
    def __init__(self, config: RandomTextureConfig):
        self.config = config

    def generate(self, height, width):
        mesh_grid = np.mgrid[0:height, 0:width].transpose(1,2,0).astype(np.uint16)

        random_texture = np.zeros((height, width), dtype=np.int16)
        for pipeline_cfg in self.config.pipeline:
            # Apply each pipeline configuration to the mesh grid
            random_texture += (RandomTexture._generate_layer(mesh_grid, self.config.texture_table, pipeline_cfg, self.config.resample) >> self.config.pipeline_count)
        return random_texture

    @staticmethod
    def _rand_coor_2d_to_16bit(x_16, y_16):
        h = (np.uint32(x_16)) ^ (np.uint32(y_16) << 16)

        with np.errstate(over='ignore'):
            h ^= h << 13
            h ^= h >> 17
            h ^= h << 5
            h = h + (h >> 3) + (h << 7)
        return np.uint16(h & 0xFFFF)

    @staticmethod 
    def _generate_layer(mesh_grid, texture_table, pipeline_config: PipelineConfig, resample_config: ResampleConfig):
        proj_x, proj_y = RandomTexture._transform_spatial(mesh_grid, pipeline_config.method, pipeline_config.scale, pipeline_config.translation)
        grid_x, grid_y = proj_x >> 4, proj_y >> 4

        rand_val = RandomTexture._rand_coor_2d_to_16bit(grid_x, grid_y)

        mask = (rand_val & 0x3FF) >= pipeline_config.threshold
        jitter_x, jitter_y = (rand_val >> 10) & 0x3, (rand_val >> 12) & 0x3
        flip_x, flip_y = (rand_val >> 14) & 0x1, (rand_val >> 15) & 0x1
    
        layer_texture = np.zeros(mesh_grid.shape[:2], dtype = np.int32)
        for resample in resample_config.resample:
            sub_sample_x, sub_sample_y = proj_x + resample[0], proj_y + resample[1]
            if resample_config.jitter:
                sub_sample_x += jitter_x
                sub_sample_y += jitter_y
            sub_sample_x &= 0xF
            sub_sample_y &= 0xF

            # fold
            sub_sample_x= np.where(sub_sample_x >= 8, 15 - sub_sample_x, sub_sample_x)
            sub_sample_y= np.where(sub_sample_y >= 8, 15 - sub_sample_y, sub_sample_y)

            if resample_config.flip:
                sub_sample_x = np.where(flip_x, 7 - sub_sample_x, sub_sample_x)
                sub_sample_y = np.where(flip_y, 7 - sub_sample_y, sub_sample_y)

            layer_texture[mask] += ((texture_table[sub_sample_y, sub_sample_x][mask]) >> resample_config.n)

        
        return layer_texture

    @staticmethod
    def _transform_spatial(mesh_grid, method: CoordinateMappingMethod, scale: tuple[int, int], translation: tuple[int, int]):
        x, y = method.apply(mesh_grid[:,:,0], mesh_grid[:,:,1])

        if scale[0] < 0:
            x <<= -scale[0]
        else:
            x >>= scale[0]
        
        if scale[1] < 0:
            y <<= -scale[1]
        else:
            y >>= scale[1]

        x += translation[0]
        y += translation[1]
        return x, y



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cv2
    
    config = RandomTextureConfig(
        texture_table = np.array([
            [-13,  51, -13, -13, -13, -13,  51, -13],
            [-13, -13,  51, -13, -13,  51, -13, -13],
            [-13, -13, -13,  51,  51, -13, -13, -13],
            [-13, -13, -13, -13,  51,  51, -13, -13],
            [-13, -13, -13,  51, -13, -13,  51, -13],
            [-13, -13,  51, -13, -13,  51, -13, -13],
            [-13,  51, -13, -13, -13, -13, -13, -13],
            [-13, -13, -13, -13, -13, -13,  51, -13],
        ], dtype=np.int16) * 20,
        pipeline_count=2,
        # pipeline = (
        #     # 两层连续的主结构
        #     PipelineConfig(ShearMethod.X18, scale=(2, 0), translation=(31, 27), threshold=0),
        #     PipelineConfig(ShearMethod.Y18, scale=(0, 2), translation=(231, 87), threshold=0),
        #     # 两层稀疏的交叉纤维
        #     PipelineConfig(ShearMethod.X31, scale=(1, 1), translation=(63, 765), threshold=128),
        #     PipelineConfig(ShearMethod.Y31, scale=(1, 1), translation=(98, 112), threshold=128),
        # ),
        # resample=ResampleConfig(resample=((0,0),(3,11)), jitter=True, flip=True, n=1)
        # pipeline = (
        #     # 主方向：两层相同 shear、不同相位，贡献约 50%
        #     PipelineConfig(ShearMethod.X8,  scale=(1, 0), translation=(31, 27),  threshold=0),
        #     PipelineConfig(ShearMethod.X8,  scale=(1, 0), translation=(231, 87), threshold=0),

        #     # 次方向：增加不规则的斜向短纤维
        #     PipelineConfig(ShearMethod.X18, scale=(1, 0), translation=(63, 765), threshold=0),

        #     # 细节层：打散而不形成明显横纹
        #     PipelineConfig(ShearMethod.X31, scale=(0, 0), translation=(98, 112), threshold=0),
        # ),
        # resample = ResampleConfig(
        #     n=1,
        #     resample=((0, 0), (3, 11)),
        #     jitter=True,
        #     flip=True,
        # )
        pipeline = (
            # 主方向：两层相同 shear、不同相位，贡献约 50%
            PipelineConfig(ShearMethod.X8,  scale=(-1, -1), translation=(31, 27),  threshold=768),
            PipelineConfig(ShearMethod.X45,  scale=(-1, -1), translation=(231, 87), threshold=768),

            # 次方向：增加不规则的斜向短纤维
            PipelineConfig(ShearMethod.X18, scale=(-1, -1), translation=(63, 765), threshold=768),

            # 细节层：打散而不形成明显横纹
            PipelineConfig(ShearMethod.X31, scale=(-1, -1), translation=(98, 112), threshold=768),
        ),
        resample = ResampleConfig(
            n=1,
            resample=((0, 0), (3, 11)),
            jitter=True,
            flip=True,
        )
    )

    random_texture = RandomTexture(config=config)

    texture = random_texture.generate(1920, 1080)
    print(config.texture_table.std())
    print(texture.mean())

    image = cv2.imread("data/desktop.jpg")[:,:,::-1] / 255.0
    h, w, _ = image.shape
    texture = random_texture.generate(h, w)

    print(texture.std())

    ret = image + 0.3*image * texture[:,:,np.newaxis]/1023 + 0.1*texture[:,:,np.newaxis] / 1023

    cv2.imwrite("data/desktop_2.jpg", ret[:,:,::-1] * 255)
    plt.title("Random Texture")
    plt.imshow(texture)
    plt.xticks([])
    plt.yticks([])

    plt.figure()
    plt.imshow(ret)
    plt.xticks([])
    plt.yticks([])
    plt.show()