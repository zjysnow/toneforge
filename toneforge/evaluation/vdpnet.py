from toneforge.evaluation.vdpnet.model import QNet

import torch
from torch.hub import load_state_dict_from_url
import numpy as np
import urllib.parse
import os

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


if __name__ == "__main__":
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import cv2
    from toneforge import EOTF
    import torchvision.transforms.functional as F

    # model = QModel("./toneforge/evaluation/vdpnet/norvdpnet_tmo.pth")
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

    