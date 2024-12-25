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
import torch
from random import randint
import random
from utils.loss_utils import l1_loss, ssim, lncc 
from utils.graphics_utils import patch_offsets, patch_warp
import torch.nn.functional as F

from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import numpy as np
    

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

#linear schedule udf opacity weight
def weight_schedule(iteration, start_iter, end_iter, start_weight, end_weight):
    if iteration < start_iter:
        return start_weight
    elif iteration > end_iter:
        return end_weight
    else:
        return start_weight + (end_weight - start_weight) * (iteration - start_iter) / (end_iter - start_iter)

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0 
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # if iteration == 15001:
        #     # use udfw disable opacities
        #     gaussians.disable_opacities()
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        # breakpoint()
        # udfw = weight_schedule(iteration, 15001, 15002, 0., 1.0)
        udfw = 1.0
        # udfw = 0.0
        
        # print(udfw)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background,lamda=udfw)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # gt_image = viewpoint_cam.original_image.cuda()
        gt_image, gt_image_gray = viewpoint_cam.get_image()
        
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        # regularization
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0
        # lambda_dist = opt.lambda_dist if iteration > 3000 and iteration < 15000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0
        
        lambda_scale = 0.5 if iteration > 30000 else 0.0
        lambda_opacity = 0.1 if iteration > 30000 else 0.0
        rend_dist = render_pkg["rend_dist"] 
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        # if lambda_dist > 0.0:
        #     print(f"dist_loss: {dist_loss.item()}")

        scale_loss = lambda_scale * (gaussians.get_scaling ** 2).mean()
        opacity_reg = lambda_opacity * ((gaussians.get_kappas.log() / 6. -  gaussians.get_opacity) ** 2).mean()
        # if iteration % 10 == 0:
        #     print(dist_loss.item())
        # loss
        # multi-view loss
        geo_loss = None
        ncc_loss = None
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            use_virtul_cam = False
            # breakpoint()
            
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                pixel_noise_th = opt.multi_view_pixel_noise_th
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['surf_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['surf_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, background,lamda=udfw)

                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['surf_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['surf_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                d_mask = d_mask & (pixel_noise < pixel_noise_th) & (~pixel_noise.isinf()) & (~pixel_noise.isnan())
                
                # breakpoint()
                weights = (1.0 / torch.exp(pixel_noise)).detach()
                weights[~d_mask] = 0
                # if iteration % 200 == 0:
                #     import cv2
                #     debug_path = os.path.join(opt.output_dir, "debug")
                #     os.makedirs(debug_path, exist_ok=True)
                #     gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                #     if 'app_image' in render_pkg:
                #         img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                #     else:
                #         img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                #     normal = render_pkg['rendered_normal'].permute(1,2,0).detach().cpu().numpy()
                #     depth_normal = render_pkg['surf_normal'].permute(1,2,0).detach().cpu().numpy()
                #     normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                #     depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                #     d_mask_show = (weights.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                #     d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                #     depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                #     depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                #     depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                #     depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                #     row0 = np.concatenate([gt_img_show, img_show, normal_show], axis=1)
                #     row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show], axis=1)
                #     image_to_show = np.concatenate([row0, row1], axis=0)
                #     cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), image_to_show)

                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                
                    loss += geo_loss
                    # if use_virtul_cam is False:
                    #     with torch.no_grad():
                    #         ## sample mask
                    #         d_mask = d_mask.reshape(-1)
                    #         valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                    #         if d_mask.sum() > sample_num:
                    #             index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                    #             valid_indices = valid_indices[index]

                    #         weights = weights.reshape(-1)[valid_indices]
                    #         ## sample ref frame patch
                    #         pixels = pixels.reshape(-1,2)[valid_indices]
                    #         offsets = patch_offsets(patch_size, pixels.device)
                    #         ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                    #         H, W = gt_image_gray.squeeze().shape
                    #         pixels_patch = ori_pixels_patch.clone()
                    #         pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
                    #         pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
                    #         ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                    #         ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                    #         ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                    #         ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                    #     ## compute Homography
                    #     ref_local_n = render_pkg["rend_normal"].permute(1,2,0)
                    #     ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                    #     # ref_local_d = render_pkg['rendered_distance'].squeeze()
                    #     rays_d = viewpoint_cam.get_rays().reshape(-1,3)[valid_indices]
                    #     mask = (ref_local_n * rays_d).sum(-1).abs() != 0
                    #     ref_local_d = render_pkg['surf_depth'].view(-1)[valid_indices] * ((ref_local_n * rays_d).sum(-1).abs())
                        
                    #     # ref_local_d = ref_local_d.reshape(H,W)
                    #     # ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        
                    #     H_ref_to_neareast = ref_to_neareast_r[None] - \
                    #         torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                    #                     ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                    #     H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                    #     H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                    #     # H_ref_to_neareast = H_ref_to_neareast
                    #     ## compute neareast frame patch
                    #     grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                    #     grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
                    #     grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
                    #     _, nearest_image_gray = nearest_cam.get_image()
                    #     sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                    #     sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                    #     ## compute loss
                    #     ncc, ncc_mask = lncc(ref_gray_val[mask], sampled_gray_val[mask])
                    #     mask = ncc_mask.reshape(-1)
                    #     ncc = ncc.reshape(-1) * weights
                    #     ncc = ncc[mask].squeeze()

                    #     if mask.sum() > 0:
                    #         ncc_loss = ncc_weight * ncc.mean()
                    #         loss += ncc_loss
                    #         if iteration > 7000:
                    #             ncc_grad = torch.autograd.grad(ncc.mean(), render_pkg['surf_depth'],retain_graph=True)[0]
                    #             geo_grad = torch.autograd.grad(geo_loss, render_pkg['surf_depth'],retain_graph=True)[0]
                    #             if ncc_grad.isnan().any() or geo_grad.isnan().any():
                    #                 breakpoint()
                                
        total_loss = loss + dist_loss + normal_loss + scale_loss + opacity_reg
        
        total_loss.backward()
        # exit()
        # if iteration > 7000 and udfw > 0.:
            # breakpoint()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item() 
            ema_dist_for_log =  dist_loss.item() 
            ema_normal_for_log = normal_loss.item()
            if iteration > opt.multi_view_weight_from_iter:
                ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
                ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            import math
            if ema_dist_for_log == math.nan:
                breakpoint()
                # print((gaussians.get_kappas_grad().sign() * gaussians.get_opacity_grad().sign() < 0).nonzero().shape)

            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}",
                    "scales": f"{gaussians.get_scaling[...,:2].mean().item():.{5}f}",
                }
                progress_bar.set_postfix(loss_dict)

                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            if tb_writer is not None:
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)
                if iteration > opt.multi_view_weight_from_iter:
                    tb_writer.add_scalar('train_loss_patches/ema_multi_view_geo', ema_multi_view_geo_for_log, iteration)
                    tb_writer.add_scalar('train_loss_patches/ema_multi_view_pho', ema_multi_view_pho_for_log, iteration)

            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background,udfw))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # Prune Gaussians according to kappa
            # if iteration > 15000 and iteration % opt.densification_interval == 0:
            #     gaussians.prune_kappas(5)
            
            # Densification
            if iteration < opt.densify_until_iter:
                # try:
                # if gaussians.get_scale_grad().isnan().any():
                #     print("NaN in scale grad")
                #     torch.save((gaussians.get_scaling_orig()), "./debug/debug_scaling.pth")
                #     exit()
                    
                #     da = radii.float().mean()
                #     torch.cuda.synchronize()
                # except Exception as e:
                #     import traceback
                #     print(traceback.format_exc())
                #     breakpoint()

                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    # opacity_cull = weight_schedule(iteration, 10001, 12000, opt.opacity_cull, opt.opacity_cull * 0.2)
                    # opacity_cull = opt.opacity_cull
                    
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                # gaussians.clip_grad_norm(1)
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass

                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 15_000, 20_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[20000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[15002])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")