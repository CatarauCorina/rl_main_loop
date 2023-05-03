import torch
import torchvision.models as models
import torchvision.transforms as transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
import numpy as np, cv2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SegmentAnythingObjectExtractor(object):

    def __init__(self, checkpoint="sam_vit_b_01ec64.pth", model_type="vit_b", no_objects=8):
        self.checkpoint_path = "checkpoints"
        self.checkpoint_file = f'{self.checkpoint_path}/{checkpoint}'
        self.sam_model = sam_model_registry[model_type](checkpoint=self.checkpoint_file).to(device)
        self.sam_encoder = SamPredictor(self.sam_model)
        self.no_objects = no_objects
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam_model,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.97,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=100,  # Requires open-cv to run post-processin
        )
        self.transform = transforms.ToTensor()
        self.pil_transform = transforms.ToPILImage()
        self.encoder = models.vgg16(pretrained=True).to(device)

    def extract_objects(self, frame):
        objects = []
        masks = self.mask_generator.generate(frame)
        masks = masks[:self.no_objects]
        for mask in masks:
            mask_inverted = np.invert(mask['segmentation']).astype(int)
            mask_arr = np.stack((mask_inverted,) * 3, axis=-1)
            masked = np.where(mask_arr == 0, frame, 0)
            tensor = self.transform(masked).float()
            objects.append(tensor)
        obj_tensor = torch.stack(objects).to(device)
        encoded_objs = self.encoder(obj_tensor)
        return encoded_objs.unsqueeze(0).unsqueeze(0).to(device)



