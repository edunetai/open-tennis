import numpy as np

try:
    import torch
    import torch.nn as nn
    from torchvision import models
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class ResNet50KeypointRegressor(nn.Module):
        def __init__(self, num_keypoints=12, pretrained=True):
            super().__init__()
            resnet = models.resnet50(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(2048, num_keypoints * 2)

        def forward(self, x):
            x = self.backbone(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x.view(-1, 12, 2)

        @torch.no_grad()
        def predict(self, image_batch):
            self.eval()
            device = next(self.parameters()).device
            image_batch = torch.from_numpy(image_batch).to(device)
            heatmaps = self.forward(image_batch)
            return heatmaps.cpu().numpy()


class CourtKeypointDetector:
    def __init__(self, model_path, device="cuda"):
        if not _TORCH_AVAILABLE:
            raise ImportError("torch and torchvision are required for CourtKeypointDetector. Install requirements.txt first.")
        self.device = device
        self.model = ResNet50KeypointRegressor(num_keypoints=12, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    def detect(self, frame):
        input_tensor = self._preprocess(frame)
        preds = self.model.predict(input_tensor)
        return preds[0].astype(np.float32)

    def _preprocess(self, frame):
        img = frame.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img
