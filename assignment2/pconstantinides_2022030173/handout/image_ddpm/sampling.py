import argparse
import json
from pathlib import Path

import numpy as np
import torch
from dataset import tensor_to_pil_image
from model import DiffusionModule
from network import UNet
from scheduler import DDPMScheduler


def load_checkpoint_config(ckpt_path: str) -> dict:
    config_path = Path(ckpt_path).parent / "config.json"
    if config_path.exists():
        with open(config_path, "r") as f:
            return json.load(f)
    return {}


def build_ddpm(config: dict) -> DiffusionModule:
    num_train_timesteps = config.get("num_diffusion_train_timesteps", 1000)
    beta_1 = config.get("beta_1", 1e-4)
    beta_T = config.get("beta_T", 0.02)
    image_resolution = config.get("image_resolution", 64)
    use_cfg = config.get("use_cfg", False)
    cfg_dropout = config.get("cfg_dropout", 0.1)
    num_classes = config.get("num_classes", 3) if use_cfg else None

    network = UNet(
        T=num_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
        use_cfg=use_cfg,
        cfg_dropout=cfg_dropout,
        num_classes=num_classes,
    )
    var_scheduler = DDPMScheduler(
        num_train_timesteps,
        beta_1=beta_1,
        beta_T=beta_T,
        mode="linear",
    )
    return DiffusionModule(network, var_scheduler)


def load_ddpm_from_checkpoint(ckpt_path: str, device: str, use_cfg: bool) -> DiffusionModule:
    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "ddpm_state_dict" in checkpoint:
        config = load_checkpoint_config(ckpt_path)
        config["use_cfg"] = use_cfg or config.get("use_cfg", False)
        ddpm = build_ddpm(config)
        ddpm.load_state_dict(checkpoint["ddpm_state_dict"])
    elif isinstance(checkpoint, dict) and "hparams" in checkpoint and "state_dict" in checkpoint:
        ddpm = DiffusionModule(None, None)
        ddpm.load(ckpt_path)
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a training checkpoint with `ddpm_state_dict` "
            "or a DiffusionModule checkpoint with `hparams` and `state_dict`."
        )

    ddpm = ddpm.to(device)
    ddpm.eval()
    return ddpm


def main(args):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    device = f"cuda:{args.gpu}"
    ddpm = load_ddpm_from_checkpoint(args.ckpt_path, device, args.use_cfg)

    total_num_samples = 500
    num_batches = int(np.ceil(total_num_samples / args.batch_size))

    for i in range(num_batches):
        sidx = i * args.batch_size
        eidx = min(sidx + args.batch_size, total_num_samples)
        B = eidx - sidx

        if args.use_cfg:  # Enable CFG sampling
            assert ddpm.network.use_cfg, f"The model was not trained to support CFG."
            samples = ddpm.sample(
                B,
                class_label=torch.randint(1, 4, (B,)),
                guidance_scale=args.cfg_scale,
            )
        else:
            samples = ddpm.sample(B)

        pil_images = tensor_to_pil_image(samples)

        for j, img in zip(range(sidx, eidx), pil_images):
            img.save(save_dir / f"{j}.png")
            print(f"Saved the {j}-th image.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str)
    parser.add_argument("--save_dir", type=str)
    parser.add_argument("--use_cfg", action="store_true")
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--cfg_scale", type=float, default=7.5)

    args = parser.parse_args()
    main(args)
