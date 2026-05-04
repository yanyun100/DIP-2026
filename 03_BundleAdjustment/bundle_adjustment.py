import argparse
import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


IMAGE_SIZE = 1024
CX = IMAGE_SIZE / 2.0
CY = IMAGE_SIZE / 2.0


def euler_xyz_to_matrix(angles):
    """Convert XYZ Euler angles in radians to rotation matrices."""
    x, y, z = angles.unbind(-1)
    cx, sx = torch.cos(x), torch.sin(x)
    cy, sy = torch.cos(y), torch.sin(y)
    cz, sz = torch.cos(z), torch.sin(z)

    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cx, -sx], dim=-1),
            torch.stack([zeros, sx, cx], dim=-1),
        ],
        dim=-2,
    )
    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rz = torch.stack(
        [
            torch.stack([cz, -sz, zeros], dim=-1),
            torch.stack([sz, cz, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rz @ ry @ rx


def load_observations(data_dir):
    points2d = np.load(data_dir / "points2d.npz")
    keys = sorted(points2d.files)
    obs = np.stack([points2d[k] for k in keys], axis=0).astype(np.float32)
    xy = torch.from_numpy(obs[:, :, :2])
    visible = torch.from_numpy(obs[:, :, 2] > 0.5)
    colors = np.load(data_dir / "points3d_colors.npy").astype(np.float32)
    return keys, xy, visible, colors


def initialize_points(xy, visible, focal, depth):
    """Back-project every point from the average normalized visible observation."""
    views, points, _ = xy.shape
    vis_f = visible.float()
    counts = vis_f.sum(dim=0).clamp_min(1.0)
    mean_xy = (xy * vis_f[:, :, None]).sum(dim=0) / counts[:, None]
    x = -((mean_xy[:, 0] - CX) / focal) * depth
    y = ((mean_xy[:, 1] - CY) / focal) * depth
    z = torch.zeros(points)
    pts = torch.stack([x, y, z], dim=-1)
    pts = pts + 0.02 * torch.randn_like(pts)
    return pts


def project(points3d, euler, translation, log_focal):
    focal = torch.exp(log_focal) + 1e-6
    r = euler_xyz_to_matrix(euler)
    cam = torch.einsum("vij,nj->vni", r, points3d) + translation[:, None, :]
    z = cam[..., 2].clamp_max(-1e-4)
    u = -focal * cam[..., 0] / z + CX
    v = focal * cam[..., 1] / z + CY
    return torch.stack([u, v], dim=-1), focal


def write_obj(path, points, colors):
    with open(path, "w", encoding="utf-8") as f:
        for p, c in zip(points, colors):
            f.write(
                "v %.6f %.6f %.6f %.6f %.6f %.6f\n"
                % (p[0], p[1], p[2], c[0], c[1], c[2])
            )


def write_ply(path, points, colors):
    rgb = np.clip(colors * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, rgb):
            f.write("%.6f %.6f %.6f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))


def plot_loss(path, history):
    plt.figure(figsize=(7, 4))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Mean reprojection error (px)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_point_cloud(path, points, colors):
    stride = max(1, len(points) // 6000)
    pts = points[::stride]
    cols = colors[::stride]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(pts[:, 0], pts[:, 2], pts[:, 1], c=cols, s=1.0, linewidths=0)
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("Y")
    ax.view_init(elev=8, azim=-90)
    max_range = np.ptp(pts, axis=0).max()
    center = pts.mean(axis=0)
    ax.set_xlim(center[0] - max_range / 2, center[0] + max_range / 2)
    ax.set_ylim(center[2] - max_range / 2, center[2] + max_range / 2)
    ax.set_zlim(center[1] - max_range / 2, center[1] + max_range / 2)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="outputs/ba")
    parser.add_argument("--iters", type=int, default=1200)
    parser.add_argument("--lr-points", type=float, default=0.015)
    parser.add_argument("--lr-cameras", type=float, default=0.003)
    parser.add_argument("--lr-focal", type=float, default=0.001)
    parser.add_argument("--focal", type=float, default=900.0)
    parser.add_argument("--depth", type=float, default=2.5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(7)
    np.random.seed(7)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keys, xy, visible, colors = load_observations(data_dir)
    xy = xy.to(args.device)
    visible = visible.to(args.device)
    views, n_points, _ = xy.shape

    points3d = torch.nn.Parameter(initialize_points(xy.cpu(), visible.cpu(), args.focal, args.depth).to(args.device))

    euler_init = torch.zeros(views, 3, device=args.device)
    euler_init[:, 1] = torch.linspace(
        math.radians(-55.0), math.radians(55.0), views, device=args.device
    )
    euler = torch.nn.Parameter(euler_init)

    translation = torch.nn.Parameter(torch.zeros(views, 3, device=args.device))
    translation.data[:, 2] = -args.depth
    log_focal = torch.nn.Parameter(torch.tensor(math.log(args.focal), device=args.device))

    optimizer = torch.optim.Adam(
        [
            {"params": [points3d], "lr": args.lr_points},
            {"params": [euler, translation], "lr": args.lr_cameras},
            {"params": [log_focal], "lr": args.lr_focal},
        ]
    )

    history = []
    obs_count = int(visible.sum().item())
    print(f"Loaded {views} views, {n_points} points, {obs_count} visible observations")

    for it in range(args.iters):
        optimizer.zero_grad(set_to_none=True)
        pred, focal = project(points3d, euler, translation, log_focal)
        residual = pred - xy
        robust = torch.sqrt((residual.square().sum(dim=-1) + 1e-4))
        data_loss = robust[visible].mean()

        center_loss = 0.01 * points3d.mean(dim=0).square().sum()
        scale_loss = 0.001 * (points3d.square().mean() - 0.18).square()
        loss = data_loss + center_loss + scale_loss
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            translation[:, 2].clamp_(max=-0.2)
            log_focal.clamp_(math.log(250.0), math.log(3000.0))

        history.append(float(data_loss.detach().cpu()))
        if it % 50 == 0 or it == args.iters - 1:
            print(f"iter {it:04d}: reprojection={history[-1]:.4f}px focal={float(focal.detach()):.2f}")

    points_np = points3d.detach().cpu().numpy()
    euler_np = euler.detach().cpu().numpy()
    trans_np = translation.detach().cpu().numpy()
    focal_value = float((torch.exp(log_focal)).detach().cpu())

    write_obj(out_dir / "reconstruction.obj", points_np, colors)
    write_ply(out_dir / "reconstruction.ply", points_np, colors)
    plot_loss(out_dir / "loss_curve.png", history)
    plot_point_cloud(out_dir / "point_cloud_preview.png", points_np, colors)

    np.save(out_dir / "points3d.npy", points_np)
    np.save(out_dir / "camera_euler_xyz.npy", euler_np)
    np.save(out_dir / "camera_translation.npy", trans_np)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "views": views,
                "points": n_points,
                "visible_observations": obs_count,
                "iterations": args.iters,
                "initial_reprojection_px": history[0],
                "final_reprojection_px": history[-1],
                "focal": focal_value,
                "outputs": [
                    "reconstruction.obj",
                    "reconstruction.ply",
                    "loss_curve.png",
                    "point_cloud_preview.png",
                    "points3d.npy",
                    "camera_euler_xyz.npy",
                    "camera_translation.npy",
                ],
            },
            f,
            indent=2,
        )
    print(f"Saved Bundle Adjustment results to {out_dir}")


if __name__ == "__main__":
    main()
