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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=False, ply_path=None, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}
        
        self.train_idu_cameras = {}  # NOTE: for iterative datasets update

        
        if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            if os.path.exists(os.path.join(args.source_path, "points3D.txt")) or os.path.exists(os.path.join(args.source_path, "depths_moge")):
                print("Found transforms_train.json and points3D.txt files, assuming multi-scale Satellite data set!")
                scene_info = sceneLoadTypeCallbacks["Satellite"](args.source_path, args.white_background, args.eval)
            else:
                print("Found transforms_train.json file, assuming Blender data set!")
                scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Multi-scale"](args.source_path, args.white_background, args.eval, args.load_allres)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            assert False, "Shuffling is not supported anymore!"
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            # random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            print("Loading point cloud from iteration {}".format(self.loaded_iter))
            if ply_path:
                self.gaussians.load_ply(os.path.join(ply_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            else:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            # === 专家建议: 场景尺度归一化（卫星场景关键修复）===
            # 将extent从~1570.85归一化到~100，确保投影半径在1-2px范围
            if self.cameras_extent > 120.0:  # 只对大场景归一化（卫星场景通常>120）
                print(f"\n[EXPERT] Scene extent={self.cameras_extent:.2f} > 120, applying normalization...")
                
                from utils.scene_normalization import normalize_scene_scale
                
                # 收集所有相机
                all_cameras = []
                for scale in resolution_scales:
                    all_cameras.extend(self.train_cameras[scale])
                    all_cameras.extend(self.test_cameras[scale])
                
                # 统一归一化
                scale_factor, normalized_pcd, normalized_cameras = normalize_scene_scale(
                    scene_info.point_cloud,
                    all_cameras,
                    target_extent=100.0,
                    verbose=True
                )
                
                # 使用归一化后的点云创建高斯
                self.gaussians.create_from_pcd(normalized_pcd, 100.0)
                
                # 更新相机（按scale分组）
                train_count = len(self.train_cameras[resolution_scales[0]])
                test_count = len(self.test_cameras[resolution_scales[0]])
                
                idx = 0
                for scale in resolution_scales:
                    self.train_cameras[scale] = normalized_cameras[idx:idx+train_count]
                    idx += train_count
                    self.test_cameras[scale] = normalized_cameras[idx:idx+test_count]
                    idx += test_count
                
                # === 专家建议：手动修复相机T和缓存 ===
                # 完整相似变换：t' = s * t + s * (R @ μ)
                print("[EXPERT] Applying similarity transform to cameras...")
                from utils.graphics_utils import getWorld2View2
                import torch
                import numpy as np
                
                # 计算场景中心
                pcd_points = normalized_pcd.points
                scene_center_original = scene_info.point_cloud.points.mean(axis=0)
                
                for scale in resolution_scales:
                    for cam in self.train_cameras[scale]:
                        # 应用完整相似变换
                        R = np.array(cam.R) if not isinstance(cam.R, np.ndarray) else cam.R
                        t_old = np.array(cam.T) if not isinstance(cam.T, np.ndarray) else cam.T
                        
                        # t' = s * t + s * (R @ μ)
                        t_new = scale_factor * t_old + scale_factor * (R @ scene_center_original)
                        cam.T = t_new
                        
                        # 重新计算world_view_transform
                        cam.world_view_transform = torch.tensor(
                            getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)
                        ).transpose(0, 1).cuda()
                        # 重新计算camera_center
                        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                        # 重新计算full_proj_transform
                        cam.full_proj_transform = (
                            cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
                        ).squeeze(0)
                    
                    for cam in self.test_cameras[scale]:
                        R = np.array(cam.R) if not isinstance(cam.R, np.ndarray) else cam.R
                        t_old = np.array(cam.T) if not isinstance(cam.T, np.ndarray) else cam.T
                        t_new = scale_factor * t_old + scale_factor * (R @ scene_center_original)
                        cam.T = t_new
                        
                        cam.world_view_transform = torch.tensor(
                            getWorld2View2(cam.R, cam.T, cam.trans, cam.scale)
                        ).transpose(0, 1).cuda()
                        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                        cam.full_proj_transform = (
                            cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))
                        ).squeeze(0)
                
                print(f"[EXPERT] ✅ Applied similarity transform to {len(all_cameras)} cameras")
                
                # 验证：打印第一个相机的参数
                if len(self.train_cameras[resolution_scales[0]]) > 0:
                    cam0 = self.train_cameras[resolution_scales[0]][0]
                    print(f"[DEBUG] Camera 0 center: {cam0.camera_center.cpu().numpy()}")
                    print(f"[DEBUG] Camera 0 T: {cam0.T}")
                    print(f"[DEBUG] Camera 0 trans: {cam0.trans}")
                    print(f"[DEBUG] Camera 0 scale: {cam0.scale}")
                
                # 更新extent
                self.cameras_extent = 100.0
                self.scene_normalized = True  # 标记已归一化
                print(f"[EXPERT] ✅ Scene normalized: extent={self.cameras_extent:.2f}, scale_factor={scale_factor:.6f}\n")
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
                self.scene_normalized = False
        
        # === CRITICAL: Compute adaptive near/far clip planes ===
        # Expert recommendation: Avoid hardcoded values for satellite scenes
        # Note: 归一化后Near/Far已经同步缩放，不需要重新计算
        if not hasattr(self, 'scene_normalized') or not self.scene_normalized:
            self._compute_adaptive_clip_planes()
        else:
            print("[EXPERT] Skipping adaptive clip planes (already normalized)")

    def _compute_adaptive_clip_planes(self):
        """
        Compute adaptive near/far clip planes based on point cloud and camera distribution
        This prevents the "all points invalid" problem in satellite scenes
        """
        try:
            import numpy as np
            from utils.adaptive_clip_planes import compute_adaptive_clip_planes
            
            # Get point cloud
            points_3d = self.gaussians.get_xyz.detach().cpu().numpy()
            
            # Get camera centers from all training cameras
            camera_centers = []
            for scale, cameras in self.train_cameras.items():
                for cam in cameras:
                    # Camera center from w2c: C = -R^T @ t
                    R = cam.R
                    T = cam.T
                    C = -R.T @ T
                    camera_centers.append(C)
            
            camera_centers = np.array(camera_centers)
            
            if len(camera_centers) > 0 and len(points_3d) > 0:
                znear, zfar = compute_adaptive_clip_planes(points_3d, camera_centers)
                
                # Apply to all cameras
                for scale, cameras in self.train_cameras.items():
                    for cam in cameras:
                        cam.znear = znear
                        cam.zfar = zfar
                
                for scale, cameras in self.test_cameras.items():
                    for cam in cameras:
                        cam.znear = znear
                        cam.zfar = zfar
                
                print(f"✅ Applied adaptive clip planes: znear={znear:.2f}, zfar={zfar:.2f}")
            else:
                print("[WARN] Could not compute adaptive clip planes (no points or cameras)")
        
        except Exception as e:
            print(f"[WARN] Failed to compute adaptive clip planes: {e}")
            print("      Using default values")

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getTrainIDUCameras(self, scale=1.0):
        return self.train_idu_cameras[scale]