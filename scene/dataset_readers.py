#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.color_space import srgb_to_linear_np, process_image_for_training
import copy
import OpenEXR

class CameraInfo(NamedTuple):
    """
    Camera information container.
    
    Coordinate System Convention:
    - R: Transposed world-to-camera rotation matrix (R_w2c.T)
         Stored transposed for CUDA compatibility
         Usage: R_w2c = cam.R.T, R_c2w = cam.R
    - T: World-to-camera translation vector (t_w2c)
         Usage: t_w2c = cam.T, t_c2w = -cam.R @ cam.T
    
    Depth Convention:
    - depth: Depth map (may be in different resolution than image)
    - depth_conf: Confidence map for depth (VGGT only, same size as depth)
    """
    uid: int
    R: np.array          # R_w2c.T (transposed for CUDA)
    T: np.array          # t_w2c (world-to-camera translation)
    FovY: np.array
    FovX: np.array
    cx: np.array
    cy: np.array
    image: np.array
    image_path: str
    image_name: str
    depth: np.array
    depth_conf: np.array  # VGGT depth confidence (same size as depth)
    mask: np.array
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            cx = intr.params[1]
            cy = intr.params[2]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            cx = intr.params[2]
            cy = intr.params[3]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        # get rid of too many opened files
        image = copy.deepcopy(image)
        cx = (cx - width / 2) / width * 2
        cy = (cy - height / 2) / height * 2
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, 
                              cx=cx, cy=cy,
                              image=image,
                              image_path=image_path, 
                              image_name=image_name, 
                              width=width, 
                              height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    
    # Check if normals exist, if not create zero normals
    if 'nx' in vertices and 'ny' in vertices and 'nz' in vertices:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        print(f"[INFO] PLY file has no normals, creating zero normals")
        normals = np.zeros_like(positions)
    
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readMultiScale(path, white_background,split, only_highres=False):
    cam_infos = []
    
    print("read split:", split)
    with open(os.path.join(path, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[split]
        
    meta = {k: np.array(meta[k]) for k in meta}
    
    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    for idx, relative_path in enumerate(meta['file_path']):
        if only_highres and not relative_path.endswith("d0.png"):
            continue
        image_path = os.path.join(path, relative_path)
        image_name = Path(image_path).stem
        
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = meta["cam2world"][idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(meta["focal"][idx], image.size[0])
        fovy = focal2fov(meta["focal"][idx], image.size[1])
        FovY = fovy 
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                        image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
    return cam_infos


def readMultiScaleNerfSyntheticInfo(path, white_background, eval, load_allres=False):
    print("Reading train from metadata.json")
    train_cam_infos = readMultiScale(path, white_background, "train", only_highres=(not load_allres))
    print("number of training images:", len(train_cam_infos))
    print("Reading test from metadata.json")
    test_cam_infos = readMultiScale(path, white_background, "test", only_highres=False)
    print("number of testing images:", len(test_cam_infos))
    if not eval:
        print("adding test cameras to training")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def fill_point_cloud_colors_from_images(pcd, cameras, path):
    """
    为无颜色的点云从图像中采样颜色
    
    Args:
        pcd: BasicPointCloud without colors
        cameras: list of CameraInfo
        path: scene directory
    
    Returns:
        BasicPointCloud with colors
    """
    print("[INFO] Filling point cloud colors from images...")
    
    points = pcd.points
    colors = np.zeros((len(points), 3), dtype=np.float32)
    vis_mask = np.zeros(len(points), dtype=bool)
    
    # 为每个点找最佳可见相机并采样颜色
    for cam_idx, cam in enumerate(cameras):
        if cam_idx % 5 == 0:
            print(f"  Processing camera {cam_idx+1}/{len(cameras)}...")
        
        # 构建w2c矩阵
        R_w2c = cam.R.T  # cam.R存储的是R_w2c的转置
        t_w2c = cam.T
        
        # 投影点到相机
        points_cam = (R_w2c @ points.T).T + t_w2c
        
        # 过滤相机后方的点
        valid_depth = points_cam[:, 2] > 0.01
        
        if not valid_depth.any():
            continue
        
        # 投影到图像平面
        fx = fov2focal(cam.FovX, cam.width)
        fy = fov2focal(cam.FovY, cam.height)
        cx = cam.width / 2 + cam.cx * cam.width / 2
        cy = cam.height / 2 + cam.cy * cam.height / 2
        
        u = fx * points_cam[:, 0] / (points_cam[:, 2] + 1e-8) + cx
        v = fy * points_cam[:, 1] / (points_cam[:, 2] + 1e-8) + cy
        
        # 检查是否在图像内
        valid_u = (u >= 0) & (u < cam.width)
        valid_v = (v >= 0) & (v < cam.height)
        valid = valid_depth & valid_u & valid_v
        
        if not valid.any():
            continue
        
        # 加载图像
        image = np.array(cam.image).astype(np.float32) / 255.0
        
        # 双线性采样
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            if vis_mask[idx]:  # 已经有颜色了
                continue
            
            ui, vi = int(u[idx]), int(v[idx])
            # 简单的最近邻采样（避免越界）
            ui = np.clip(ui, 0, cam.width - 1)
            vi = np.clip(vi, 0, cam.height - 1)
            
            colors[idx] = image[vi, ui]
            vis_mask[idx] = True
    
    # 未采样到的点用灰色
    colors[~vis_mask] = 0.5
    
    print(f"  ✅ Filled colors: {vis_mask.sum()}/{len(points)} points visible")
    print(f"  ⚠️  Gray fallback: {(~vis_mask).sum()} points")
    
    return BasicPointCloud(
        points=points,
        colors=colors,
        normals=pcd.normals if hasattr(pcd, 'normals') else np.zeros_like(points)
    )


def readVGGTCamerasFromJSON(path, json_file="vggt_poses.json"):
    """
    Read camera parameters from VGGT exported JSON file.
    
    Args:
        path: scene directory
        json_file: JSON filename (default: "vggt_poses.json")
    
    Returns:
        list of CameraInfo
    
    Note:
        - VGGT exports w2c (world-to-camera) extrinsics
        - JSON contains dual intrinsics (image size + depth size)
        - Depth maps are in VGGT processing size (e.g., 518x518)
        - Images are in original size (e.g., 2048x2048)
    """
    json_path = os.path.join(path, json_file)
    
    if not os.path.exists(json_path):
        return None
    
    print(f"Reading VGGT poses from {json_file}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Verify format
    metadata = data.get("metadata", {})
    pose_type = metadata.get("pose_type", "")
    
    if pose_type != "w2c_opencv":
        print(f"Warning: Expected pose_type='w2c_opencv', got '{pose_type}'")
    
    print(f"  Version: {metadata.get('version', 'unknown')}")
    print(f"  Pose type: {pose_type}")
    print(f"  FoV unit: {metadata.get('fov_unit', 'unknown')}")
    
    cam_infos = []
    frames = data["frames"]
    
    for idx, frame in enumerate(frames):
        # Read image
        image_filename = frame["image_path"]
        
        # Try different path combinations
        possible_paths = [
            os.path.join(path, "images", image_filename),  # Standard: scene_dir/images/filename
            os.path.join(path, image_filename),             # Direct: scene_dir/filename
            image_filename                                   # Absolute path
        ]
        
        image_path = None
        for p in possible_paths:
            if os.path.exists(p):
                image_path = p
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Could not find image: {image_filename}")
        
        image_name = Path(image_path).stem
        image = Image.open(image_path)
        
        # Read intrinsics (use original image size intrinsics for rendering)
        intr = frame["intrinsic"]
        fx, fy = intr["fx"], intr["fy"]
        cx, cy = intr["cx"], intr["cy"]
        width, height = frame["width"], frame["height"]
        
        # Normalize principal point to [-1, 1]
        cx_norm = (cx - width / 2) / width * 2
        cy_norm = (cy - height / 2) / height * 2
        
        # Convert to FoV (Skyfall-GS uses FoV)
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)
        
        # Read extrinsics (w2c format from VGGT)
        extr = frame["extrinsic_w2c"]
        R_w2c = np.array(extr["R"])  # 3x3 rotation matrix (world-to-camera)
        t_w2c = np.array(extr["t"])  # 3x1 translation vector (world-to-camera)
        
        # CameraInfo storage convention (for CUDA compatibility):
        # - R stores R_w2c.T (transposed world-to-camera rotation)
        # - T stores t_w2c (world-to-camera translation)
        # Usage:
        # - To get w2c: R_w2c = cam.R.T, t_w2c = cam.T
        # - To get c2w: R_c2w = cam.R, t_c2w = -cam.R @ cam.T
        R = np.transpose(R_w2c)  # Store transposed for CUDA
        T = t_w2c                # Store w2c translation
        
        # Read depth map (VGGT size, e.g., 518x518)
        depth = None
        depth_conf = None
        
        if "depth_path" in frame:
            depth_path = os.path.join(path, frame["depth_path"])
            if os.path.exists(depth_path):
                depth = np.load(depth_path)
        
        if "depth_conf_path" in frame:
            conf_path = os.path.join(path, frame["depth_conf_path"])
            if os.path.exists(conf_path):
                depth_conf = np.load(conf_path)
        
        # Read mask
        mask_path = os.path.join(path, "masks", image_name + ".npy")
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            mask = mask.astype(np.uint8)
        else:
            # Create binary mask: 0 if all pixels are (0,0,0), else 1
            mask = 1 - np.all(np.array(image) == 0, axis=-1).astype(np.uint8)
        
        cam_info = CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            cx=cx_norm,
            cy=cy_norm,
            image=image,
            image_path=image_path,
            image_name=image_name,
            depth=depth,
            depth_conf=depth_conf,
            mask=mask,
            width=width,
            height=height
        )
        cam_infos.append(cam_info)
    
    print(f"  Loaded {len(cam_infos)} cameras from VGGT")
    
    return cam_infos


def readSatelliteInfo(path, white_background, eval, extension=".png"):
    # Check for VGGT initialization first
    vggt_json_path = os.path.join(path, "vggt_poses.json")
    
    if os.path.exists(vggt_json_path):
        print("=" * 80)
        print("Found vggt_poses.json - Using VGGT initialization")
        print("=" * 80)
        
        # Read VGGT cameras
        train_cam_infos = readVGGTCamerasFromJSON(path, "vggt_poses.json")
        
        if train_cam_infos is None:
            print("Failed to read VGGT poses, falling back to transforms_train.json")
        else:
            # For VGGT, we don't have separate test set in the JSON
            # Use the same approach as original: split or use all for training
            test_cam_infos = []
            
            if not eval:
                # Training mode: use all cameras
                pass
            else:
                # Evaluation mode: could split here if needed
                # For now, keep all in training
                pass
            
            # Get normalization parameters
            nerf_normalization = getNerfppNorm(train_cam_infos)
            
            # Load point cloud from VGGT with priority order:
            # 1. Downsampled point cloud (if exists)
            # 2. Original VGGT point cloud
            # 3. Standard points3D.ply
            # 4. Generate from depth maps (last resort)
            
            downsampled_ply = os.path.join(path, "vggt_points3d_downsampled.ply")
            original_ply = os.path.join(path, "vggt_points3d.ply")
            fallback_ply = os.path.join(path, "points3D.ply")
            
            ply_path = None
            if os.path.exists(downsampled_ply):
                ply_path = downsampled_ply
                print(f"✅ Using downsampled point cloud: {downsampled_ply}")
            elif os.path.exists(original_ply):
                ply_path = original_ply
                print(f"Using original VGGT point cloud: {original_ply}")
            elif os.path.exists(fallback_ply):
                ply_path = fallback_ply
                print(f"Falling back to points3D.ply")
            
            # Load point cloud
            if ply_path:
                try:
                    pcd = fetchPly(ply_path)
                    print(f"Loaded {len(pcd.points)} points from {os.path.basename(ply_path)}")
                    
                    # 检查是否有颜色
                    if not hasattr(pcd, 'colors') or pcd.colors is None or len(pcd.colors) == 0:
                        print("[WARN] Point cloud has no colors!")
                        print("[INFO] Filling colors from VGGT images...")
                        pcd = fill_point_cloud_colors_from_images(pcd, train_cam_infos, path)
                    else:
                        print(f"  ✅ Point cloud has colors (range: [{pcd.colors.min():.3f}, {pcd.colors.max():.3f}])")
                        
                except Exception as e:
                    print(f"Warning: Could not load point cloud from {ply_path}: {e}")
                    print("Generating point cloud from VGGT depth maps...")
                    pcd = generate_pcd_from_vggt_depth(train_cam_infos)
            else:
                print("No point cloud file found, generating from VGGT depth maps...")
                pcd = generate_pcd_from_vggt_depth(train_cam_infos)
            
            scene_info = SceneInfo(point_cloud=pcd,
                                 train_cameras=train_cam_infos,
                                 test_cameras=test_cam_infos,
                                 nerf_normalization=nerf_normalization,
                                 ply_path=ply_path)
            return scene_info
    
    # Original Skyfall-GS initialization (fallback)
    print("=" * 80)
    print("Using original Skyfall-GS initialization (transforms_train.json)")
    print("=" * 80)
    print("Reading Training Transforms")
    train_cam_infos, R, T = readSatelliteCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos, _, _ = readSatelliteCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    print(f"Number of training images: {len(train_cam_infos)}")
    print(f"Number of testing images: {len(test_cam_infos)}")
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    # nerf_normalization = {"translate": np.array([0.0, 0.0, 0.0]), "radius": 128.0}
    # Generate .ply file from point3D.txt
    ply_path = os.path.join(path, "points3D.ply")
    txt_path = os.path.join(path, "points3D.txt")
    print("Converting point3D.txt to .ply, will happen only the first time you open the scene.")
    try:
        xyz, rgb, _ = read_points3D_text(txt_path)
        if R is not None and T is not None:
            print("Normalizing point cloud")
            xyz = np.matmul(xyz, R.T) - T
            # Get the radius of the point cloud using 99% of the points
            radius = np.percentile(np.linalg.norm(xyz, axis=1), 99)
            # Resize the point cloud to fit in a sphere of radius 256
            scale = 256 / radius
            print(f"Point cloud radius: {radius}, scale: {scale}")
            xyz = xyz * scale
            # Make the point cloud lies in z = 0
            z_min = np.percentile(xyz[:, 2], 1)
            xyz = xyz - np.array([0, 0, z_min])
            print("Point cloud z_min: ", z_min)
            # print("Point cloud z_avg: ", np.mean(xyz[:, 2]))
            # Also resize the camera pose
            new_train_cam_infos = []
            new_test_cam_infos = []
            print("Normalizing camera poses")
            for cam in train_cam_infos:
                # cam_info is NamedTuple, we can't directly modify it
                # 1. Reconstruct the original c2w matrix from w2c components
                R_w2c = cam.R  # Already transposed in your code
                T_w2c = cam.T
                
                # Build the full w2c matrix
                w2c_matrix = np.eye(4)
                w2c_matrix[:3, :3] = R_w2c.T  # Transpose back for matrix construction
                w2c_matrix[:3, 3] = T_w2c
                
                # Get the c2w matrix
                c2w_matrix = np.linalg.inv(w2c_matrix)
                
                # 2. Apply the transformations in world space
                # Apply scaling
                c2w_matrix[:3, 3] *= scale
                
                # Apply z-shift (only to the z component)
                c2w_matrix[2, 3] -= z_min  # Note the sign - subtracting in world space
                
                # 3. Convert back to w2c
                w2c_transformed = np.linalg.inv(c2w_matrix)
                
                # 4. Extract the components
                R_new = np.transpose(w2c_transformed[:3, :3])  # Remember to transpose for CUDA code
                T_new = w2c_transformed[:3, 3]
                
                # Create the new camera info
                new_train_cam_infos.append(cam._replace(R=R_new, T=T_new))
            for cam in test_cam_infos:
                # cam_info is NamedTuple, we can't directly modify it
                # 1. Reconstruct the original c2w matrix from w2c components
                R_w2c = cam.R  # Already transposed in your code
                T_w2c = cam.T
                
                # Build the full w2c matrix
                w2c_matrix = np.eye(4)
                w2c_matrix[:3, :3] = R_w2c.T  # Transpose back for matrix construction
                w2c_matrix[:3, 3] = T_w2c
                
                # Get the c2w matrix
                c2w_matrix = np.linalg.inv(w2c_matrix)
                
                # 2. Apply the transformations in world space
                # Apply scaling
                c2w_matrix[:3, 3] *= scale
                
                # Apply z-shift (only to the z component)
                c2w_matrix[2, 3] -= z_min  # Note the sign - subtracting in world space
                
                # 3. Convert back to w2c
                w2c_transformed = np.linalg.inv(c2w_matrix)
                
                # 4. Extract the components
                R_new = np.transpose(w2c_transformed[:3, :3])  # Remember to transpose for CUDA code
                T_new = w2c_transformed[:3, 3]
                
                # Create the new camera info
                new_test_cam_infos.append(cam._replace(R=R_new, T=T_new))
            train_cam_infos = new_train_cam_infos
            test_cam_infos = new_test_cam_infos
            nerf_normalization = {"translate": np.array([0.0, 0.0, 0.0]), "radius": 128.0}
        else:
            print("No rotation matrix found, skipping normalization")
            nerf_normalization = {"translate": np.array([0.0, 0.0, 0.0]), "radius": 128.0}
        print(f"Nerf Normalization: {nerf_normalization}")
        storePly(ply_path, xyz, rgb)
    except Exception as e:
        print(f"Error converting point3D.txt to .ply: {e}")

    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        print("Loading point cloud from", ply_path)
        pcd = fetchPly(ply_path)
        # print number of points
        print(f"Number of points in the point cloud: {len(pcd.points)}")
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readSatelliteCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        # The scene has been normalized, so that the up vector is (0, 0, 1)
        if "R" in contents:
            R_fix = np.array(contents["R"])[:3, :3]
            assert R_fix.shape == (3, 3), f"R_fix.shape = {R_fix.shape}"
            T_fix = np.array(contents["T"])
            c2w_key = "transform_matrix_rotated"
        else:
            R_fix = None
            T_fix = None
            c2w_key = "transform_matrix"
        # width = contents["w"]
        # height = contents["h"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"])

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame[c2w_key])

            # No need for this change in satellite data, we use COLMAP coordinates system
            # c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            mask_path = os.path.join(path, "masks", image_name+".npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                mask = mask.astype(np.uint8)
            else:
                # assert False, "No mask found for image: {}".format(image_path)
                # create a binary mask, if all pixel value is (0, 0, 0), set it to 0, otherwise 1
                mask = 1 - np.all(np.array(image) == 0, axis=-1).astype(np.uint8)

            
            depth_path = os.path.join(path, "depths_moge", image_name+".exr")
            if os.path.exists(depth_path):
                depth = read_exr(depth_path)
            else:
                depth = None

            
            focal_x = frame["fl_x"]
            focal_y = frame["fl_y"]
            cx = frame["cx"]
            cy = frame["cy"]
            height = image.size[1]
            width = image.size[0]
            cx = (cx - width / 2) / width * 2
            cy = (cy - height / 2) / height * 2

            FovX = focal2fov(focal_x, image.size[0])
            FovY = focal2fov(focal_y, image.size[1])

            cam_infos.append(
                CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, 
                            cx=cx, cy=cy,
                            image=image,
                            image_path=image_path, 
                            image_name=image_name,
                            depth=depth,
                            depth_conf=None,  # No confidence for MoGe depth
                            mask=mask,
                            width=image.size[0], 
                            height=image.size[1])
            )
    return cam_infos, R_fix, T_fix


def generate_pcd_from_vggt_depth(cam_infos, conf_threshold=1.5):
    """
    Generate point cloud from VGGT depth maps.
    
    Args:
        cam_infos: list of CameraInfo with VGGT depth and depth_conf
        conf_threshold: confidence threshold for filtering points
    
    Returns:
        BasicPointCloud
    """
    print(f"Generating point cloud from VGGT depth maps (conf > {conf_threshold})")
    
    all_points = []
    all_colors = []
    
    for cam in cam_infos:
        if cam.depth is None or cam.depth_conf is None:
            continue
        
        depth = cam.depth
        conf = cam.depth_conf
        image = np.array(cam.image) / 255.0  # Normalize to [0, 1]
        
        H, W = depth.shape
        
        # Get intrinsics for depth size
        # Note: depth is in VGGT size, need to use depth-size intrinsics
        # For now, we'll compute from FoV
        fx = W / (2 * np.tan(cam.FovX / 2))
        fy = H / (2 * np.tan(cam.FovY / 2))
        cx = W / 2
        cy = H / 2
        
        # Create pixel grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Filter by confidence
        valid_mask = conf > conf_threshold
        
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        depth_valid = depth[valid_mask]
        
        # Unproject to camera coordinates
        x_cam = (u_valid - cx) * depth_valid / fx
        y_cam = (v_valid - cy) * depth_valid / fy
        z_cam = depth_valid
        
        points_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
        
        # Transform to world coordinates
        # cam.R stores R_w2c.T (transposed), cam.T stores t_w2c
        # To get c2w: R_c2w = cam.R (already transposed), t_c2w = -cam.R @ cam.T
        R_c2w = cam.R  # This is R_w2c.T = R_c2w
        t_c2w = -R_c2w @ cam.T
        
        points_world = (R_c2w @ points_cam.T).T + t_c2w
        
        # Get colors (need to resize image to depth size)
        from PIL import Image as PILImage
        image_resized = PILImage.fromarray((image * 255).astype(np.uint8)).resize((W, H), PILImage.BILINEAR)
        image_resized = np.array(image_resized) / 255.0
        
        colors = image_resized[valid_mask]
        
        all_points.append(points_world)
        all_colors.append(colors)
    
    if len(all_points) == 0:
        print("Warning: No valid points generated from VGGT depth")
        # Return empty point cloud
        return BasicPointCloud(
            points=np.zeros((0, 3)),
            colors=np.zeros((0, 3)),
            normals=np.zeros((0, 3))
        )
    
    all_points = np.concatenate(all_points, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)
    
    print(f"Generated {len(all_points)} points from VGGT depth maps")
    
    return BasicPointCloud(
        points=all_points,
        colors=all_colors,
        normals=np.zeros_like(all_points)
    )

def read_exr(filename: str) -> np.ndarray:
    """
    Read EXR file with its original metadata and attributes
    """
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    
    # Get data window
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    # Get pixel type and channels
    pixel_type = header['channels']['R'].type if 'R' in header['channels'] else header['channels']['Y'].type
    
    # Read channel data
    if 'R' in header['channels']:  # RGB format
        channels = ['R', 'G', 'B']
        pixel_data = [np.frombuffer(exr_file.channel(c, pixel_type), dtype=np.float32) for c in channels]
        img = np.stack([d.reshape(height, width) for d in pixel_data], axis=-1)
    else:  # Grayscale format
        pixel_data = np.frombuffer(exr_file.channel('Y', pixel_type), dtype=np.float32)
        img = pixel_data.reshape(height, width)
    
    return img


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Multi-scale": readMultiScaleNerfSyntheticInfo,
    "Satellite": readSatelliteInfo,
}
