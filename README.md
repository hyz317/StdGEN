<p align="center"><img src="https://stdgen.github.io/static/images/stdgen2.png" width="30%"></p>
<p align="center"><img src="https://stdgen.github.io/static/images/teaser.png" width="100%"></p>

# StdGEN: Semantic-Decomposed 3D Character Generation from Single Images
*"Std" stands for [S]eman[t]ic-[D]ecomposed, also inspired by the "std" namespace in C++ for standardization.*

<a href="https://stdgen.github.io/"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Github&color=blue&logo=github-pages"></a>&ensp;<a href="https://arxiv.org/abs/2411.05738"><img src="https://img.shields.io/badge/ArXiv-2411.05738-brightgreen"></a>&ensp;<a href="https://huggingface.co/hyz317/StdGEN"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model_Card-Huggingface-orange"></a>&ensp;<a href="https://huggingface.co/spaces/hyz317/StdGEN"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a><br>



https://github.com/user-attachments/assets/ce1b534f-fd1c-41c2-b03d-1208969a9d47

## 🔥 Updates

**[2025/03/17]** Online HuggingFace Gradio demo is released!

**[2025/03/04]** Inference code, dataset, pretrained checkpoints are released!

**[2025/02/27]** Our paper is accepted by **CVPR 2025**!



## 🔍 Table of Contents
- [⚙️ Deployment](#deployment)
- [🖥️ Run StdGEN](#run-stdgen)
- [💼 Anime3D++ dataset](#dataset)
- [📝 Citation](#citation)



<a name="deployment"></a>

## ⚙️ Deployment

Set up a Python environment and install the required packages:

```bash
conda create -n stdgen python=3.9 -y
conda activate stdgen

# Install torch, torchvision, xformers, torch-scatter based on your machine configuration
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install other dependencies
pip install -r requirements.txt
```

Download pretrained weights from our 🤗 Huggingface repo (<a href="https://huggingface.co/hyz317/StdGEN">download here</a>) and [ViT-H SAM model](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth), place the files in the `./ckpt/` directory.



<a name="run-stdgen"></a>

## 🖥️ Run StdGEN

```bash
# stage 1.1: reference image to A-pose image
python infer_canonicalize.py --input_dir ./input_cases --output_dir ./result/apose

# stage 1.2: A-pose image to multi-view image
# decomposed generation (by default)
python infer_multiview.py --input_dir ./result/apose --output_dir ./result/multiview
# with model offloading to save VRAM (may be slower)
python infer_multiview.py --input_dir ./result/apose --output_dir ./result/multiview --low_vram
# If you only need non-decomposed generation, then run:
python infer_multiview.py --input_dir ./result/apose --output_dir ./result/multiview --num_levels 1

# stage 2: S-LRM reconstruction
python infer_slrm.py --input_dir ./result/multiview --output_dir ./result/slrm

# stage 3: multi-layer refinement
# decomposed generation (by default)
python infer_refine.py --input_mv_dir result/multiview --input_obj_dir ./result/slrm --output_dir ./result/refine 
# If you only need non-decomposed generation, then run:
python infer_refine.py --input_mv_dir result/multiview --input_obj_dir ./result/slrm --output_dir ./result/refine --no_decompose
```

### Tips

- The script `infer_canonicalize.py` automatically determines whether to use `rm_anime_bg` for background removal based on the alpha channel.
- `rm_anime_bg` may not perform as effectively for other styles like 2.5D or real-world images, and the background can significantly impact the results if it isn't entirely removed. You can try using other background removal tools (e.g. Clipdrop or Google's multimodal flash model) before uploading the image.
- Currently our training data consists of full-body images, so inputting half-body photos (e.g., missing feet or legs) may lead to suboptimal results. You can try to use image uncropping tools (e.g. Clipdrop) to get a full-body prediction.
- After running `infer_canonicalize.py`, you can modify the A-pose character images in `result/apose` to achieve more desirable outcomes.
- The refinement of the hair section partially depends on the hair mask prediction. Check the results in `result/refined/CASE_NAME/distract_mask.png` to decide whether to adjust the `--outside_ratio` parameter (default is 0.20). If the mask includes unwanted information, decrease the value; otherwise, increase it.
- If the results are not satisfactory, try experimenting with different seeds.



<a name="dataset"></a>

## 💼 Anime3D++ dataset

Due to policy restrictions, we are unable to redistribute the raw VRM format 3D character data. However, you can download the Vroid dataset by following the instructions provided in [PAniC-3D](https://github.com/ShuhongChen/panic3d-anime-reconstruction). In place of the raw data, we are offering the train/test data list rendering script to render images and semantic maps.

First, install [Blender](https://www.blender.org/) and download the [VRM Blender Add-on](https://vrm-addon-for-blender.info/en/). Then, install the add-on using the following command:

```bash
blender --background --python blender/install_addon.py -- VRM_Addon_for_Blender-release.zip
```

Next, execute the Blender rendering script:

```bash
cd blender
python distributed_uniform.py --input_dir /PATH/TO/YOUR/VROIDDATA --save_dir /PATH/TO/YOUR/SAVEDIR --workers_per_gpu 4
```

The train/test data list can be found at `data/train_list.json` and `data/test_list.json`.



<a name="citation"></a>

## 📝 Citation

If you find our work useful, please kindly cite:

```
@article{he2024stdgen,
  title={StdGEN: Semantic-Decomposed 3D Character Generation from Single Images},
  author={He, Yuze and Zhou, Yanning and Zhao, Wang and Wu, Zhongkai and Xiao, Kaiwen and Yang, Wei and Liu, Yong-Jin and Han, Xiao},
  journal={arXiv preprint arXiv:2411.05738},
  year={2024}
}
```



## Acknowledgements

Some of the code in this repo is borrowed from [InstantMesh](https://github.com/TencentARC/InstantMesh), [Unique3D](https://github.com/AiuniAI/Unique3D), [Era3D](https://github.com/pengHTYX/Era3D) and [CharacterGen](https://github.com/zjp-shadow/CharacterGen). We sincerely thank them all.



## Disclaimer

Our released checkpoints are also for research purposes only. Users are granted the freedom to create models using this tool, but they are obligated to comply with local laws and utilize it responsibly. The developers disclaim responsibility for user-generated content and will not assume any responsibility for potential misuse by users.









