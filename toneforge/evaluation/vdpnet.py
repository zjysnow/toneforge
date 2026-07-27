import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.hub import load_state_dict_from_url
import numpy as np
import urllib.parse
import os

class Regressor(nn.Module):
    def __init__(self, in_size=1, out_size=1, params_size = None, bSigmoid = True):
        super(Regressor, self).__init__()

        if params_size == None:
            params_size = 0

        self.params_size = params_size

        if bSigmoid:
            self.regressor = nn.Sequential(
                    nn.Linear(in_size + params_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, out_size),
                    nn.Sigmoid()
                )
        else:
            self.regressor = nn.Sequential(
                    nn.Linear(in_size + params_size, 256),
                    nn.ReLU(),
                    nn.Linear(256, out_size)
                )

    def forward(self, features, params = None):
        if len(features.shape) == 4:
            features = features.mean(-1).mean(-1)
                
        if (self.params_size != 0) and (params != None):
            features = torch.cat((features, params), dim = 1)
            
        q = self.regressor(features)
        
        if not self.training:
            q = q.clamp(0,1)
            
        return q


class BlockQ(nn.Module):
    def __init__(self, in_size, out_size, std = 1):
        super(BlockQ, self).__init__()
    
        self.conv = nn.Sequential(
                    nn.Conv2d(in_size, out_size, 3, stride = std, padding=1),
                    nn.ReLU())

    def forward(self, input):
        return self.conv(input)


class QNet(nn.Module):
    def __init__(self, in_size=1, out_size=1, params_size = None):
        super(QNet, self).__init__()

        self.conv = nn.Sequential(
                    BlockQ(in_size, 32),
                    BlockQ(32, 32),
                    nn.MaxPool2d(2),
                                  
                    BlockQ(32, 64),
                    BlockQ(64, 64),
                    nn.MaxPool2d(2),
                                  
                    BlockQ(64, 128),
                    BlockQ(128, 128),
                    nn.MaxPool2d(2),
                                  
                    BlockQ(128, 256),
                    BlockQ(256, 256),
                    nn.MaxPool2d(2),

                    BlockQ(256, 512),
                    BlockQ(512, 512, 2),
                    nn.MaxPool2d(2),
                    )
 
        self.regressor = Regressor(512, out_size, params_size, bSigmoid = False)


    def forward(self, stim, lmax = None):
        features = self.conv(stim)
        q = self.regressor(features, lmax)

        if not self.training:
            q = q.clamp(0,1)

        return q


class QModel:
    def __init__(self, ckpt_file = os.path.expanduser("~/.cache/torch/hub/checkpoints/norvdpnet_tmo.pth")):
        checkpoint = self._load_checkpoint(ckpt_file)

        self.model = QNet()
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.get_device())

        self.model.eval()

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _is_url(self, path: str) -> bool:
        parsed = urllib.parse.urlparse(path)
        return parsed.scheme in ('http', 'https')

    def _load_checkpoint(self, path_or_url: str) -> dict:
        if self._is_url(path_or_url):
            checkpoint = load_state_dict_from_url(
                path_or_url, 
                map_location="cpu", 
                progress=True,
                weights_only=True
            )
        else:
            if not os.path.exists(path_or_url):
                checkpoint = load_state_dict_from_url(
                                "http://www.banterle.com/francesco/projects/norvdpnet/norvdpnet_tmo.pth", 
                                map_location="cpu", 
                                progress=True,
                                weights_only=True
                            )
            else:
                checkpoint = torch.load(path_or_url, weights_only=True, map_location=torch.device("cpu"))

        return checkpoint


    def predict(self, image:torch.Tensor):
        image = image.to(self.get_device())

        with torch.no_grad():
            qval = self.model(image)

        return qval.data.cpu().numpy().squeeze()


if __name__ == '__main__':
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import cv2
    from toneforge import EOTF
    import torchvision.transforms.functional as F

    model = QModel()
    
    # image = torch.from_numpy((np.random.rand(1,1,512,512) ** 2.2).astype(np.float32))

    # image = torch.from_numpy(np.ones((1,1,512,512)).astype(np.float32))

    image = cv2.imread("./data/a0005-jn_2007_05_10__564.jpg")[:,:,::-1] / 255.0
    image = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    image = torch.from_numpy(EOTF.sRGB(image).astype(np.float32)).unsqueeze(0).unsqueeze(0)

    qval = model.predict(image)
    print(round(qval * 10000)/100)

    blurred_image = F.gaussian_blur(image, kernel_size=[5, 5], sigma=[1.5, 1.5])

    qval = model.predict(blurred_image)
    print(round(qval * 10000)/100)
