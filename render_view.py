"""Render four causal exposures from each of five calibrated cameras."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ncore.data import FrameTimepoint
from instant_nurec.config_schema.instantnurec import InstantNuRecConfig
from instant_nurec.datasets.instantnurec_ncore import NCoreInstantNuRecDataset
from instant_nurec.model import make
from instant_nurec.predict.render_preview import render_composited_frame
from instant_nurec.utils.batch import DataAndRenderingBatch, InstantNuRecDataBatch
from instant_nurec.utils.geometry import se3_matrix_to_tquat, tquat_to_se3_matrix

CAMERAS = ["camera_" + name for name in (
    "front_wide_120fov", "cross_left_120fov", "cross_right_120fov",
    "rear_left_70fov", "rear_right_70fov",
)]


def dataset(sequence, cameras, width, height):
    spec = dict(ncore_json_paths=[str(sequence)], context_camera_ids=cameras,
                supervision_camera_ids=cameras, frame_batch_sampler=dict(n_frames_per_sample=4),
                camera_subsampler=dict(frame_width=width, frame_height=height))
    config = InstantNuRecConfig(out_dir=".", release_profile="pa-multiview", dataset={"predict": spec})
    ds = NCoreInstantNuRecDataset(config.dataset.predict, width, height, 4, global_seed=38)
    loaders = ds._get_loaders_and_sensors(ds.ncore_json_paths[0], ds.all_context_camera_ids)
    return config, ds, loaders


def load_batch(ds, selection, sensors, ref, tracks):
    sub = ds._build_camera_subsampler()
    rig, mapping = ds._get_rig_trajectory("context-", selection, sensors, ref, tracks, sub)
    data = ds._load_data_batch(selection, mapping, sensors, sub, {})
    batch = InstantNuRecDataBatch(context=[DataAndRenderingBatch(data=data)], context_rig=[rig]).to("cuda")
    batch.maybe_compute_rendering_data("cuda")
    return batch


def select_frames(times, start, cutoff, pose_times, distance):
    bounds = np.interp([start, cutoff], pose_times, distance)
    # Use four positions from the 18-point arc-length grid over the causal window.
    targets = np.interp(np.linspace(*bounds, 18)[[0, 6, 11, 17]], distance, pose_times)
    targets[[0, -1]] = [start, cutoff]
    selection = {}
    for name, stamps in times.items():
        indices = np.searchsorted(stamps, targets, "right") - 1
        if len(np.unique(indices)) < 4 or indices.min() < 0:
            first = np.searchsorted(stamps, start)
            last = np.searchsorted(stamps, cutoff, "right") - 1
            if last - first < 3:
                raise ValueError(f"{name} has fewer than four exposures in this window")
            indices = np.linspace(first, last, 4).round().astype(int)
        selection[name] = indices
    return selection


class Scene:
    def __init__(self, sequence):
        config, self.train, self.source = dataset(sequence, CAMERAS, 504, 280)
        _, self.target, self.front = dataset(sequence, CAMERAS[:1], 1024, 576)
        self.model = make(config).to("cuda").eval()
        self.named = {str(k): v for k, v in self.source.camera_sensors.items()}
        self.times = {k: v.get_frames_timestamps_us(FrameTimepoint.END) for k, v in self.named.items()}
        self.poses, self.pt = self.source.T_rig_worlds_with_timestamps_us["main"]
        self.arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(self.poses[:, :3, 3], axis=0), axis=1))]
        self.first = max(ts[0] for ts in self.times.values())
        self.last = min(ts[-1] for ts in self.times.values())

    @torch.inference_mode()
    def prepare(self, seconds):
        cutoff = self.first + round(seconds * 1e6)
        if cutoff > self.last:
            raise ValueError(f"Requested time exceeds common coverage {(self.last-self.first)/1e6:.3f} s")
        keep = self.pt <= cutoff
        if keep.sum() < 2:
            raise ValueError("Insufficient pose history")
        selection = select_frames(self.times, max(self.first, cutoff-2_000_000), cutoff,
                                  self.pt[keep], self.arc[keep])
        ref = self.named[CAMERAS[0]].get_frames_T_source_sensor(
            source_node="world", frame_indices=int(selection[CAMERAS[0]][0]), frame_timepoint=FrameTimepoint.END)
        tracks = {"main": (self.poses[keep], self.pt[keep])}
        inputs = load_batch(self.train, selection, self.source.camera_sensors, ref, tracks)
        front = load_batch(self.target, {CAMERAS[0]: [int(selection[CAMERAS[0]][-1])]},
                           self.front.camera_sensors, ref, tracks)
        torch.cuda.current_stream().synchronize()
        return inputs, front

    @torch.inference_mode()
    def reconstruct(self, inputs):
        prediction = self.model.model.reconstruct(inputs.context, None)[0]
        return prediction.preprocess_for_export(inputs.context[0], self.model.export_preprocess)


@torch.inference_mode()
def render(scene, front, yaw, pitch, height, rear):
    ctx = front.context[0]
    camera = ctx.rendering.camera
    poses = tquat_to_se3_matrix(camera.poses_tquat_startend[0], unbatch=False)
    ext = front.context_rig[0].camera_calibrations[CAMERAS[0]].T_sensor_rig.to(poses)
    a, p = map(math.radians, (yaw, pitch))
    c, s, cp, sp = math.cos(a), math.sin(a), math.cos(p), math.sin(p)
    virtual = poses.new_tensor([[s,-sp*c,cp*c,-rear*c],[-c,-sp*s,cp*s,-rear*s],
                                [0,-cp,-sp,height],[0,0,0,1]])
    change = poses[0] @ torch.linalg.inv(ext) @ virtual @ torch.linalg.inv(poses[0])
    # Transform both origins and directions, preserving the calibrated front lens.
    rays = camera.rays[0]
    original_rays = rays.clone()
    original_poses = camera.poses_tquat_startend[0].clone()
    try:
        rays[..., :3] = rays[..., :3] @ change[:3,:3].T + change[:3,3]
        rays[..., 3:] = rays[..., 3:] @ change[:3,:3].T
        camera.poses_tquat_startend[0] = se3_matrix_to_tquat(change @ poses, unbatch=False)
        return render_composited_frame(scene, ctx)[0]
    finally:
        rays.copy_(original_rays)
        camera.poses_tquat_startend[0].copy_(original_poses)


def load_difix(root):
    import sys
    root = root.resolve()
    sys.path[:0] = [str(root / "vae"), str(root / "unet"), str(root)]
    from pipeline_difix import DifixPipeline
    from autoencoder_kl import AutoencoderKL
    from unet_2d_condition import UNet2DConditionModel
    opts = dict(torch_dtype=torch.bfloat16, local_files_only=True)
    return DifixPipeline.from_pretrained(root, vae=AutoencoderKL.from_pretrained(root/"vae", **opts),
        unet=UNet2DConditionModel.from_pretrained(root/"unet", **opts), **opts).to("cuda")


@torch.inference_mode()
def enhance(pipe, rgb, front):
    def tensor(x):
        return x.clamp(0,1).mul(255).round().div(255).permute(2,0,1)[None]
    fixed = pipe("remove degradation", image=tensor(rgb),
                 ref_image=tensor(front.context[0].data.camera.labels.rgb[0]),
                 num_inference_steps=1, timesteps=[199], guidance_scale=0.0,
                 generator=torch.Generator(device="cuda").manual_seed(0), output_type="pt").images
    return fixed


def save(rgb, path):
    if not torch.isfinite(rgb).all():
        raise ValueError("Non-finite render")
    Image.fromarray(rgb.clamp(0,1).mul(255).round().byte().cpu().numpy()).save(path)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output", type=Path, help="Output directory for PNG sequences")
    mode.add_argument("--serve", type=int, metavar="PORT", help="Open local interactive viewer")
    parser.add_argument("--difix", type=Path, help="Optional local nvidia/difix_ref snapshot")
    parser.add_argument("--start", type=float, default=2.0, help="Seconds from first common exposure")
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--step", type=float, default=0.2, help="Scene seconds between outputs")
    parser.add_argument("--yaw", type=float, default=0)
    parser.add_argument("--pitch", type=float, default=12)
    parser.add_argument("--height", type=float, default=2.4)
    parser.add_argument("--rear", type=float, default=4.5)
    args = parser.parse_args()
    if args.frames < 1 or args.step <= 0 or args.start < 0:
        parser.error("Expected frames >= 1, step > 0 and start >= 0")
    for path in (args.sequence, args.checkpoint):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    if args.serve is not None and not 0 <= args.serve <= 65535:
        parser.error("Port must be in [0, 65535]")
    if args.difix is not None and not args.difix.is_dir():
        parser.error(f"Missing Difix directory: {args.difix}")
    if not all(map(math.isfinite, (args.start, args.step, args.yaw, args.pitch, args.height, args.rear))):
        parser.error("Time and camera parameters must be finite")
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
    os.environ["INSTANT_NUREC_FULL_PT"] = str(args.checkpoint.resolve())
    driver = Scene(args.sequence.resolve())
    if args.output and round((args.start + (args.frames-1)*args.step)*1e6) > driver.last-driver.first:
        parser.error("Requested export extends beyond common camera coverage")
    pipe = load_difix(args.difix) if args.difix else None
    if args.serve is not None:
        from viewer import serve
        serve(driver, pipe, args, render, enhance, CAMERAS)
        return
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        pending = prefetch.submit(driver.prepare, args.start)
        for j in range(args.frames):
            inputs, front = pending.result()
            if j+1 < args.frames:
                pending = prefetch.submit(driver.prepare, args.start+(j+1)*args.step)
            scene = driver.reconstruct(inputs)
            rgb = render(scene, front, args.yaw, args.pitch, args.height, args.rear)
            save(rgb, args.output / f"gs-{j:05}.png")
            if pipe is not None:
                fixed = enhance(pipe, rgb, front)
                save(fixed[0].float().permute(1,2,0), args.output / f"difix-{j:05}.png")


if __name__ == "__main__":
    main()
