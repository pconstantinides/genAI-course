import argparse
import os

import numpy as np
import torch

from model import Decoder
from utils import normalize_pts, normalize_normals, showMeshReconstruction


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def createGrid(points, resolution=128):
	"""
	constructs a 3D grid containing the point cloud
	each grid point will store the implicit function value
	Args:
		points: 3D points of the point cloud
		resolution: grid resolution i.e., grid will be NxNxN where N=resolution
	Returns:
		X,Y,Z coordinates of grid vertices
		max and min dimensions of the bounding box of the point cloud
	"""
	max_dimensions = np.max(points, axis=0)
	min_dimensions = np.min(points, axis=0)
	bounding_box_dimensions = max_dimensions - min_dimensions
	max_dimensions = max_dimensions + bounding_box_dimensions / 10
	min_dimensions = min_dimensions - bounding_box_dimensions / 10

	X, Y, Z = np.meshgrid(
		np.linspace(min_dimensions[0], max_dimensions[0], resolution),
		np.linspace(min_dimensions[1], max_dimensions[1], resolution),
		np.linspace(min_dimensions[2], max_dimensions[2], resolution),
	)

	return X, Y, Z, max_dimensions, min_dimensions


def loadCheckpoint(model, checkpoint_folder, resume_file):
	checkpoint_path = os.path.join(checkpoint_folder, resume_file)
	checkpoint = torch.load(checkpoint_path, map_location=device)
	model.load_state_dict(checkpoint["state_dict"])
	model.eval()
	return model


def nnReconstruction(model, X, Y, Z, clamping_distance=0.1, batch_size=2048):
	"""
	evaluates the trained decoder on the 3D grid and returns the implicit field.
	"""
	Q = np.array([X.reshape(-1), Y.reshape(-1), Z.reshape(-1)]).transpose()
	number_samples = Q.shape[0]
	implicit = np.zeros((number_samples, ), dtype=np.float32)

	start_idx = 0
	for batch_start in range(0, number_samples, batch_size):
		batch_end = min(batch_start + batch_size, number_samples)
		xyz_tensor = torch.from_numpy(Q[batch_start:batch_end]).float().to(device)

		with torch.no_grad():
			pred_sdf_tensor = model(xyz_tensor)
			pred_sdf_tensor = torch.clamp(
				pred_sdf_tensor, -clamping_distance, clamping_distance
			)

		pred_sdf = pred_sdf_tensor.cpu().squeeze().numpy()
		implicit[start_idx:start_idx + (batch_end - batch_start)] = pred_sdf
		start_idx += batch_end - batch_start

	return implicit.reshape(X.shape)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Neural network surface reconstruction")
	parser.add_argument("--file", type=str, default="data/bunny-1000.pts", help="input point cloud filename")
	parser.add_argument("--checkpoint_folder", type=str, default="checkpoints/", help="folder containing trained checkpoints")
	parser.add_argument("--resume_file", type=str, default="model_best.pth.tar", help="best checkpoint file name")
	parser.add_argument("--grid_N", type=int, default=128, help="grid resolution")
	parser.add_argument("--clamping_distance", type=float, default=0.1, help="clamping distance for sdf values")
	args = parser.parse_args()

	data = np.loadtxt(args.file)
	points = normalize_pts(data[:, :3])
	normals = normalize_normals(data[:, 3:6])

	X, Y, Z, max_dimensions, min_dimensions = createGrid(points, args.grid_N)

	model = Decoder(
		dims=[512, 512, 512, 512, 512, 512, 512, 512],
		dropout=[0, 1, 2, 3, 4, 5, 6, 7],
		norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
		latent_in=[4],
	).to(device)

	model = loadCheckpoint(model, args.checkpoint_folder, args.resume_file)
	print(f"Loaded checkpoint from {os.path.join(args.checkpoint_folder, args.resume_file)}")
	print(f"Running neural reconstruction on {args.file}")

	implicit = nnReconstruction(
		model,
		X,
		Y,
		Z,
		clamping_distance=args.clamping_distance,
	)

	showMeshReconstruction(implicit)
