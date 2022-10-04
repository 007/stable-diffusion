import argparse
import sys
import json
import os
from contextlib import nullcontext
from itertools import islice
import time


# set before torch import to try and bypass all caching
#os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = "1"
import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from pytorch_lightning import seed_everything
from torch import autocast
from torchvision.utils import make_grid
from tqdm import tqdm, trange

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.util import instantiate_from_config

try:
    # this silences the annoying "Some weights of the model checkpoint were not used when initializing..." message at start.

    from transformers import logging

    logging.set_verbosity_error()
except:
    pass


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def sleeprint(x):
    print(f"\n{x}\n")
    time.sleep(20)


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.half()
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        # for some reason we need to RGB2BGR to make wm_encoder happy, then BGR2RGB to get back to normal colors
        img = np.array(img)[:, :, ::-1]
        img = wm_encoder.encode(img, "dwtDct")
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y) / 255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    # load safety model
    from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
    from transformers import AutoFeatureExtractor

    safety_model_id = "CompVis/stable-diffusion-safety-checker"
    safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
    safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def img_callback(pred, i):
    global model, sample_path, base_count, metadata
    image_tensors = samples_to_images(pred, model)
    x_image_torch = torch.from_numpy(image_tensors).permute(0, 3, 1, 2)
    rebase_count = base_count
    for x_sample in x_image_torch:
        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
        img = Image.fromarray(x_sample.astype(np.uint8))
        img.save(os.path.join(sample_path, f"{rebase_count:05}-{i:03}.png"), pnginfo=metadata)
        rebase_count += 1


# if we're _really_ tight on memory we won't be able to decode all tensors at once
# iterate on samples(iterations(x)) individually to pull them over to CPU memory
def samples_to_images(samples, model):
    global opt
    if opt.ultra_low_vram:
        previous_device = model.device
        model.cpu().to(torch.float32)
    output_samples = []
    for sample in torch.split(samples, 1):
        sample_list = []
        for iteration in torch.split(sample, 1):
            if model.device.type == 'cuda':
                torch.cuda.empty_cache()
            if opt.ultra_low_vram:
                one_sample = model.decode_first_stage(iteration.cpu())
            else:
                one_sample = model.decode_first_stage(iteration).cpu()
            # must cast to float32, `half` doesn't implement some CPU operations?
            sample_list.append(one_sample.to(torch.float32))
        output_samples.append(torch.cat(sample_list))
    output_samples = torch.cat(output_samples)

    output_samples = torch.clamp((output_samples + 1.0) / 2.0, min=0.0, max=1.0)
    output_samples = output_samples.permute(0, 2, 3, 1).numpy()
    if opt.ultra_low_vram:
        model.to(previous_device)
    return output_samples


def main():
    global model, opt, sample_path, base_count, metadata
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples",
    )
    parser.add_argument(
        "--skip_grid",
        action="store_true",
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action="store_true",
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--skip_metadata",
        action="store_true",
        help="do not save prompt/seed/step metadata in images.",
    )
    parser.add_argument(
        "--save_steps",
        action="store_true",
        help="save images of each step alongside final result",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action="store_true",
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action="store_true",
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action="store_true",
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/stable-diffusion-v1.4.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast",
    )
    parser.add_argument(
        "--skip_watermark",
        action="store_true",
        help="disable watermarking output images",
    )
    parser.add_argument(
        "--skip_safety_check",
        action="store_true",
        help="disable safety check for output images, allows NSFW output",
    )
    parser.add_argument(
        "--ultra_low_vram",
        action="store_true",
        help="for extremely low vram, do tensor->image conversion entirely in system RAM",
    )
    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    # set tensor default to half to complement model.half
    torch.set_default_tensor_type(torch.HalfTensor)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    if not opt.skip_watermark:
        from imwatermark import WatermarkEncoder

        print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
        wm = "StableDiffusionV1"
        wm_encoder = WatermarkEncoder()
        wm_encoder.set_watermark("bytes", wm.encode("utf-8"))

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    if opt.save_steps:
        print(f"Saving images for each of {opt.ddim_steps} steps")
        img_cb = img_callback
    else:
        img_cb = None

    metadata = PngInfo()
    if not opt.skip_metadata:
        if opt.from_file:
            metadata.add_text("prompt", str(prompts))
        else:
            metadata.add_text("prompt", str(opt.prompt))
        metadata.add_text("checkpoint", str(opt.ckpt))
        metadata.add_text("iter", str(opt.n_iter))
        metadata.add_text("samples", str(opt.n_samples))
        metadata.add_text("scale", str(opt.scale))
        metadata.add_text("seed", str(opt.seed))
        metadata.add_text("steps", str(opt.ddim_steps))

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                        try:
                            samples_ddim, _ = sampler.sample(
                                S=opt.ddim_steps,
                                conditioning=c,
                                batch_size=opt.n_samples,
                                shape=shape,
                                img_callback=img_cb,
                                verbose=False,
                                unconditional_guidance_scale=opt.scale,
                                unconditional_conditioning=uc,
                                eta=opt.ddim_eta,
                                x_T=start_code,
                            )
                        except torch.cuda.OutOfMemoryError:
                            print("ran out of memory, dumping snapshot")
                            print(json.dumps(torch.cuda.memory_snapshot()))
                            sys.exit(1)

                        x_samples_ddim = samples_to_images(samples_ddim, model)

                        if opt.skip_safety_check:
                            x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        else:
                            x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            x_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                        if not opt.skip_save:
                            for x_sample in x_image_torch:
                                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                if not opt.skip_watermark:
                                    img = put_watermark(img, wm_encoder)
                                img.save(os.path.join(sample_path, f"{base_count:05}.png"), pnginfo=metadata)
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_image_torch)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    img = Image.fromarray(grid.astype(np.uint8))
                    if not opt.skip_watermark:
                        img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(outpath, f"grid-{grid_count:04}.png"), pnginfo=metadata)
                    grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath}\nEnjoy.")


if __name__ == "__main__":
    main()
