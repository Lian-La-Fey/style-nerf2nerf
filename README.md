# Style-NeRF2NeRF: 3D Style Transfer from Style-Aligned Multi-View Images (SIGGRAPH Asia 2024)

[Project Page](https://haruolabs.github.io/style-n2n/) / [arXiv](https://arxiv.org/abs/2406.13393)

Our work is based on [Instruct-NeRF2NeRF](https://github.com/ayaanzhaque/instruct-nerf2nerf). We also modify the code provided in ["Style Aligned Image Generation via Shared Attention"](https://github.com/google/style-aligned) and ["A Sliced Wasserstein Loss for Neural Texture Synthesis"](https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis).
We sincerely thank the authors for releasing their code!!

# Installation

## 1. Install Nerfstudio dependencies

Style-NeRF2NeRF is built with Nerfstudio and therefore has the same dependency reqirements. Specfically [PyTorch](https://pytorch.org/) and [tinycudann](https://github.com/NVlabs/tiny-cuda-nn) are required.

Run the following to create an environment and install [nerfstudio](https://docs.nerf.studio/quickstart/installation.html) dependencies. 

```bash
conda create --name {ENV_NAME} -y python=3.10
conda activate {ENV_NAME}

python -m pip install --upgrade pip

pip uninstall torch torchvision functorch tinycudann
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

## 2. Setup Style-NeRF2NeRF
Once you have finished installing dependencies, run the following:
```bash
cd style-nerf2nerf
pip install -e .
```

## 3. Checking the install

The following command should include `sn2n` as one of the options:
```bash
ns-train -h
```

# Testing Style-NeRF2NeRF

You need to first train a standard `nerfacto` scene using the source views. To process your custom data, please refer to [this](https://docs.nerf.studio/quickstart/custom_dataset.html) documentation.

Once you have your data ready, you can train your initial NeRF with the following command:

```bash
ns-train nerfacto --data {PROCESSED_DATA_DIR} --max-num-iterations 60000
```

For more details on training a vanilla `nerfacto` model, see [Nerfstudio documentation](https://docs.nerf.studio/quickstart/first_nerf.html).

Once you've finished training the source scene, a checkpoint will be saved to the `outputs/nerfstudio_models` directory. Copy the path to the `nerfstudio_models` folder.

To generate stylized images from the original multi-views, please refer to a juptyer notebook `generate_stylized_views.ipynb`.

After generating the stylized images, the source NeRF, and placing the [pre-trained VGG-19](https://github.com/tchambon/A-Sliced-Wasserstein-Loss-for-Neural-Texture-Synthesis) model file under home dir (required for calculating the Sliced Wasserstein distance loss), run the following to fine-tune the NeRF model:

```bash
ns-train sn2n --data {TARGET_DATA_PATH} --load-dir {PROCESSED_DATA_DIR} --pipeline.model.swd-loss-mult 1.0 --pipeline.model.orientation-loss-mult 0 --pipeline.model.pred-normal-loss-mult 0
```

The `{PROCESSED_DATA_DIR}` must be the same path as used in training the original NeRF. It will train for 15,000 iterations by default.

After the NeRF is trained, you can render novel views from NeRF using the standard Nerfstudio workflow, found [here](https://docs.nerf.studio/quickstart/viewer_quickstart.html).

## Training Notes
If the source NeRF shows visible artifacts, consider increasing the number of iterations or using a larger distortion loss coefficient via `--pipeline.model.distortion-loss-mult {default 0.002}` when training the source "nerfacto" model.

## Tips
Depending on the degree of ambiguity in stylized images for fine-tuning the soure NeRF, the result may be unsatisfactory. In that case, a more detailed text prompt or a lower text guidance scale can help reduce ambiguity and improve the 3D style transfer quality. (e.g. "A campsite" -> "A campsite with red tents")

## Example
Using the dataset provided by [Instruct-NeRF2NeRF](https://instruct-nerf2nerf.github.io/), following is an example of running Style-NeRF2NeRF on "bear" scene:

```bash
# Pre-train the source NeRF
# Place the bear dir under ./dataset
ns-train nerfacto --data ./dataset/bear --max-num-iterations 60000

# Replace {XXX} with path to pre-trained nerfacto model from previous step
ns-train sn2n --data data/bear-grizzly --load-dir outputs/bear/nerfacto/{XXX}/nerfstudio_models --pipeline.model.use-l1 True --pipeline.model.swd-loss-mult 1.0 --pipeline.model.orientation-loss-mult 0 --pipeline.model.pred-normal-loss-mult 0
```

## ToDos
- More code cleaning-up
- Add support for style blending
- And possibly more...

## Citation
```
@inproceedings{fujiwara2024sn2n,
    title     = {Style-NeRF2NeRF: 3D Style Transfer from Style-Aligned Multi-View Images},
    author    = {Haruo Fujiwara and Yusuke Mukuta and Tatsuya Harada},
    booktitle = {SIGGRAPH Asia 2024 Conference Papers},
    year      = {2024}
}
```