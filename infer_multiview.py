import argparse
import os
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional,  List
from omegaconf import OmegaConf, DictConfig
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import Dict
import torch
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from einops import rearrange, repeat
from multiview.pipeline_multiclass import StableUnCLIPImg2ImgPipeline

weight_dtype = torch.float16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def tensor_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def nonzero_normalize_depth(depth, mask=None):
    if mask.max() > 0:  # not all transparent
        nonzero_depth_min = depth[mask > 0].min()
    else:
        nonzero_depth_min = 0
    depth = (depth - nonzero_depth_min) / depth.max()
    return np.clip(depth, 0, 1)


class SingleImageData(Dataset):
    def __init__(self,
                 input_dir,
                 prompt_embeds_path='./multiview/fixed_prompt_embeds_6view',
                 image_transforms=[],
                 total_views=6,
                 ext="png",
                 return_paths=True,
                 ) -> None:
        """Create a dataset from a folder of images.
        If you pass in a root directory it will be searched for images
        ending in ext (ext can be a list)
        """
        self.input_dir = Path(input_dir)
        self.return_paths = return_paths
        self.total_views = total_views

        self.paths = glob.glob(str(self.input_dir / f'*.{ext}'))

        print('============= length of dataset %d =============' % len(self.paths))
        self.tform = image_transforms
        self.normal_text_embeds = torch.load(f'{prompt_embeds_path}/normal_embeds.pt')
        self.color_text_embeds = torch.load(f'{prompt_embeds_path}/clr_embeds.pt')


    def __len__(self):
        return len(self.paths)


    def load_rgb(self, path, color):
        img = plt.imread(path)
        img = Image.fromarray(np.uint8(img * 255.))
        new_img = Image.new("RGB", (1024, 1024))
        # white background
        width, height = img.size
        new_width = int(width / height * 1024)
        img = img.resize((new_width, 1024))
        new_img.paste((255, 255, 255), (0, 0, 1024, 1024))
        offset = (1024 - new_width) // 2
        new_img.paste(img, (offset, 0))
        return new_img

    def __getitem__(self, index):
        data = {}
        filename = self.paths[index]

        if self.return_paths:
            data["path"] = str(filename)
        color = 1.0
        cond_im_rgb = self.process_im(self.load_rgb(filename, color))
        cond_im_rgb = torch.stack([cond_im_rgb] * self.total_views, dim=0)

        data["image_cond_rgb"] = cond_im_rgb
        data["normal_prompt_embeddings"] = self.normal_text_embeds
        data["color_prompt_embeddings"] = self.color_text_embeds
        data["filename"] = filename.split('/')[-1]

        return data

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

    def tensor_to_image(self, tensor):
        return Image.fromarray(np.uint8(tensor.numpy() * 255.))


@dataclass
class TestConfig:
    pretrained_model_name_or_path: str
    pretrained_unet_path:Optional[str]
    revision: Optional[str]
    validation_dataset: Dict
    save_dir: str
    seed: Optional[int]
    validation_batch_size: int
    dataloader_num_workers: int
    save_mode: str
    local_rank: int

    pipe_kwargs: Dict
    pipe_validation_kwargs: Dict
    unet_from_pretrained_kwargs: Dict
    validation_grid_nrow: int
    camera_embedding_lr_mult: float

    num_views: int
    camera_embedding_type: str

    pred_type: str
    regress_elevation: bool
    enable_xformers_memory_efficient_attention: bool

    cond_on_normals: bool
    cond_on_colors: bool
    
    regress_elevation: bool
    regress_focal_length: bool
    


def convert_to_numpy(tensor):
    return tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()

def save_image(tensor, fp):
    ndarr = convert_to_numpy(tensor)
    save_image_numpy(ndarr, fp)
    return ndarr

def save_image_numpy(ndarr, fp):
    im = Image.fromarray(ndarr)
    # pad to square
    if im.size[0] != im.size[1]:
        size = max(im.size)
        new_im = Image.new("RGB", (size, size))
        # set to white
        new_im.paste((255, 255, 255), (0, 0, size, size))
        new_im.paste(im, ((size - im.size[0]) // 2, (size - im.size[1]) // 2))
        im = new_im
    # resize to 1024x1024
    im = im.resize((1024, 1024), Image.LANCZOS)
    im.save(fp)

def run_multiview_infer(dataloader, pipeline, cfg: TestConfig, save_dir, num_levels=3):
    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device="cuda" if torch.cuda.is_available else "cpu").manual_seed(cfg.seed)
    
    images_cond = []
    for _, batch in tqdm(enumerate(dataloader)):
        torch.cuda.empty_cache()
        images_cond.append(batch['image_cond_rgb'][:, 0].cuda()) 
        imgs_in = torch.cat([batch['image_cond_rgb']]*2, dim=0).cuda()
        num_views = imgs_in.shape[1]
        imgs_in = rearrange(imgs_in, "B Nv C H W -> (B Nv) C H W")# (B*Nv, 3, H, W)

        target_h, target_w = imgs_in.shape[-2], imgs_in.shape[-1]

        normal_prompt_embeddings, clr_prompt_embeddings = batch['normal_prompt_embeddings'].cuda(), batch['color_prompt_embeddings'].cuda()
        prompt_embeddings = torch.cat([normal_prompt_embeddings, clr_prompt_embeddings], dim=0)
        prompt_embeddings = rearrange(prompt_embeddings, "B Nv N C -> (B Nv) N C")

        # B*Nv images
        unet_out = pipeline(
            imgs_in, None, prompt_embeds=prompt_embeddings,
            generator=generator, guidance_scale=3.0, output_type='pt', num_images_per_prompt=1,
            height=cfg.height, width=cfg.width,
            num_inference_steps=40, eta=1.0,
            num_levels=num_levels,
        )

        for level in range(num_levels):
            out = unet_out[level].images
            bsz = out.shape[0] // 2

            normals_pred = out[:bsz]
            images_pred = out[bsz:] 

            cur_dir = save_dir 
            os.makedirs(cur_dir, exist_ok=True)

            for i in range(bsz//num_views):
                scene = batch['filename'][i].split('.')[0]
                scene_dir = os.path.join(cur_dir, scene, f'level{level}')
                os.makedirs(scene_dir, exist_ok=True)

                img_in_ = images_cond[-1][i].to(out.device)
                for j in range(num_views):
                    view = VIEWS[j]
                    idx = i*num_views + j
                    normal = normals_pred[idx]
                    color = images_pred[idx]

                    ## save color and normal---------------------
                    normal_filename = f"normal_{j}.png"
                    rgb_filename = f"color_{j}.png"
                    save_image(normal, os.path.join(scene_dir, normal_filename))
                    save_image(color, os.path.join(scene_dir, rgb_filename))

    torch.cuda.empty_cache()    

def load_multiview_pipeline(cfg):
    pipeline = StableUnCLIPImg2ImgPipeline.from_pretrained(
        cfg.pretrained_path,
        torch_dtype=torch.float16,)
    pipeline.unet.enable_xformers_memory_efficient_attention()
    if torch.cuda.is_available():
        pipeline.to(device)
        if cfg.low_vram:
            print("Using Model CPU Offload and VAE Slicing to save VRAM Usage.")
            pipeline.enable_model_cpu_offload()
            pipeline.enable_vae_slicing()
    return pipeline

def main(
    cfg: TestConfig
):
    set_seed(cfg.seed)
    pipeline = load_multiview_pipeline(cfg)
    if torch.cuda.is_available():
        pipeline.to(device)

    image_transforms = [transforms.Resize(int(max(cfg.height, cfg.width))),
                        transforms.CenterCrop((cfg.height, cfg.width)),
                        transforms.ToTensor(),
                        transforms.Lambda(lambda x: x * 2. - 1),
                        ]
    image_transforms = transforms.Compose(image_transforms)
    dataset = SingleImageData(image_transforms=image_transforms, input_dir=cfg.input_dir, total_views=cfg.num_views)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False
    )
    os.makedirs(cfg.output_dir, exist_ok=True)

    with torch.no_grad():
        run_multiview_infer(dataloader, pipeline, cfg, cfg.output_dir, num_levels=cfg.num_levels)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--num_levels", type=int, default=3)
    parser.add_argument("--pretrained_path", type=str, default='./ckpt/StdGEN-multiview-1024')
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=576)
    parser.add_argument("--input_dir", type=str, default='./result/apose')
    parser.add_argument("--output_dir", type=str, default='./result/multiview')
    parser.add_argument("--low_vram", action='store_true')
    cfg = parser.parse_args()

    if cfg.num_views == 6:
        VIEWS = ['front', 'front_right', 'right', 'back', 'left', 'front_left']
    else:
        raise NotImplementedError(f"Number of views {cfg.num_views} not supported")
    main(cfg)
