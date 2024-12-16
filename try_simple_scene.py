import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scene import gaussian_model,cameras
from gaussian_renderer import simple_render

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

def getProjectionMatrix(znear, zfar, fovX, fovY):
    import math
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def focal2fov(focal, pixels):
    import math
    return 2*math.atan(pixels/(2*focal))

def homogeneous(points):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

def homogeneous_vec(vec):
    """
    homogeneous points
    :param points: [..., 3]
    """
    return torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)

def build_covariance_2d(
    mean3d, cov3d, viewmatrix, tan_fovx, tan_fovy, focal_x, focal_y
):
    import math
    t = (mean3d @ viewmatrix[:3,:3]) + viewmatrix[-1:,:3]
    tz = t[..., 2]
    tx = t[..., 0]
    ty = t[..., 1]

    # Eq.29 locally affine transform
    # perspective transform is not affine so we approximate with first-order taylor expansion
    # notice that we multiply by the intrinsic so that the variance is at the sceen space
    J = torch.zeros(mean3d.shape[0], 3, 3).to(mean3d)
    J[..., 0, 0] = 1 / tz * focal_x
    J[..., 0, 2] = -tx / (tz * tz) * focal_x
    J[..., 1, 1] = 1 / tz * focal_y
    J[..., 1, 2] = -ty / (tz * tz) * focal_y
    W = viewmatrix[:3,:3].T # transpose to correct viewmatrix
    cov2d = J @ W @ cov3d @ W.T @ J.permute(0,2,1)

    # add low pass filter here according to E.q. 32
    filter = torch.eye(2,2).to(cov2d) * 0.0
    return cov2d[:, :2, :2] + filter[None]

def build_covariance_3d(s, r):
    L = build_scaling_rotation(s, r).permute(0,2,1)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def projection_ndc(points, viewmatrix, projmatrix):
    points_o = homogeneous(points) # object space
    points_h = points_o @ viewmatrix @ projmatrix # screen space # RHS
    p_w = 1.0 / (points_h[..., -1:] + 0.000001)
    p_proj = points_h * p_w
    p_view = points_o @ viewmatrix
    in_mask = p_view[..., 2] >= 0.2
    return p_proj, p_view, in_mask

def get_radius(cov2d):
    det = cov2d[:, 0, 0] * cov2d[:,1,1] - cov2d[:, 0, 1] * cov2d[:,1,0]
    mid = 0.5 * (cov2d[:, 0,0] + cov2d[:,1,1])
    lambda1 = mid + torch.sqrt((mid**2-det).clip(min=0.1))
    lambda2 = mid - torch.sqrt((mid**2-det).clip(min=0.1))
    return 3.0 * torch.sqrt(torch.max(lambda1, lambda2)).ceil()

def volume_splatting(means3D, scales, quats, colors, opacities, intrins, viewmat, projmat):
    projmat = torch.zeros(4,4).cuda()
    projmat[:3,:3] = intrins
    projmat[-1,-2] = 1.0
    projmat = projmat.T

    mean_ndc, mean_view, in_mask = projection_ndc(means3D, viewmatrix=viewmat, projmatrix=projmat)

    depths = mean_view[:,2]
    mean_coord_x = mean_ndc[..., 0]
    mean_coord_y = mean_ndc[..., 1]

    means2D = torch.stack([mean_coord_x, mean_coord_y], dim=-1)
    # scales = torch.cat([scales[..., :2], scales[..., -1:]*1e-2], dim=-1)
    cov3d = build_covariance_3d(scales, quats)

    W, H = (intrins[0,-1] * 2).long().item(), (intrins[1,-1] * 2).long().item()
    fx, fy = intrins[0,0], intrins[1,1]
    tan_fovx = W / (2 * fx)
    tan_fovy = H / (2 * fy)
    cov2d = build_covariance_2d(means3D, cov3d, viewmat, tan_fovx, tan_fovy, fx, fy)
    radii = get_radius(cov2d)

    # Rasterization
    # generate pixels
    pix = torch.stack(torch.meshgrid(torch.arange(H), torch.arange(W), indexing='xy'), dim=-1).to('cuda').flatten(0,-2)
    sorted_conic = cov2d.inverse() # inverse of variance
    dx = (pix[:,None,:] - means2D[None,:]) # B P 2
    dist2 = dx[:, :, 0]**2 * sorted_conic[:, 0, 0] + dx[:, :, 1]**2 * sorted_conic[:, 1, 1]+ dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 0, 1]+ dx[:,:,0]*dx[:,:,1] * sorted_conic[:, 1, 0]
    depth_acc = depths[None].expand_as(dist2)

    image, depthmap = alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W)
    return image, depthmap, means2D, radii, dist2

def alpha_blending(alpha, colors):
  T = torch.cat([torch.ones_like(alpha[-1:]), (1-alpha).cumprod(dim=0)[:-1]], dim=0)
  image = (T * alpha * colors).sum(dim=0).reshape(-1, colors.shape[-1])
  alphamap = (T * alpha).sum(dim=0).reshape(-1, 1)
  return image, alphamap


def alpha_blending_with_gaussians(dist2, colors, opacities, depth_acc, H, W):
    colors = colors.reshape(-1,1,colors.shape[-1])
    depth_acc = depth_acc.T[..., None]
    depth_acc = depth_acc.repeat(1,1,1)

    # evaluate gaussians
    # just for visualization, the actual cut off can be 3 sigma!
    cutoff = 1**2
    dist2 = dist2.T
    gaussians = torch.exp(-0.5*dist2) * (dist2 < cutoff)
    gaussians = gaussians[..., None]
    alpha = opacities.unsqueeze(1) * gaussians

    # accumulate gaussians
    image, _ = alpha_blending(alpha, colors)
    depthmap, alphamap = alpha_blending(alpha, depth_acc)
    depthmap = depthmap / alphamap
    depthmap = torch.nan_to_num(depthmap, 0, 0)
    return image.reshape(H,W,-1), depthmap.reshape(H,W,-1)

def get_inputs(num_points=8):
    length = 0.5
    x = np.linspace(-1, 1, num_points) * length
    y = np.linspace(-1, 1, num_points) * length
    x, y = np.meshgrid(x, y)
    means3D = torch.from_numpy(np.stack([x,y, 0 * np.random.rand(*x.shape)], axis=-1).reshape(-1,3)).cuda().float()
    quats = torch.zeros(1,4).repeat(len(means3D), 1).cuda()
    quats[..., 0] = 1.
    scale = length /(num_points-1)
    scales = torch.zeros(1,3).repeat(len(means3D), 1).fill_(scale).cuda()
    return means3D, scales, quats

def get_cameras():
    intrins = torch.tensor([[711.1111,   0.0000, 256.0000,   0.0000],
               [  0.0000, 711.1111, 256.0000,   0.0000],
               [  0.0000,   0.0000,   1.0000,   0.0000],
               [  0.0000,   0.0000,   0.0000,   1.0000]]).cuda()
    c2w = torch.tensor([[-8.6086e-01,  3.7950e-01, -3.3896e-01,  6.7791e-01],
         [ 5.0884e-01,  6.4205e-01, -5.7346e-01,  1.1469e+00],
         [ 1.0934e-08, -6.6614e-01, -7.4583e-01,  1.4917e+00],
         [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]).cuda()

    width, height = 512, 512
    focal_x, focal_y = intrins[0, 0], intrins[1, 1]
    viewmat = torch.linalg.inv(c2w).permute(1,0)
    FoVx = focal2fov(focal_x, width)
    FoVy = focal2fov(focal_y, height)
    projmat = getProjectionMatrix(znear=0.2, zfar=1000, fovX=FoVx, fovY=FoVy).transpose(0,1).cuda()
    projmat = viewmat @ projmat
    return intrins, viewmat, projmat, height, width

def main():
    num_points1=4
    means3D, scales, quats = get_inputs(num_points=num_points1)
    intrins, viewmat, projmat, height, width = get_cameras()
    focal_x, focal_y = intrins[0, 0], intrins[1, 1]
    FoVx = focal2fov(focal_x, width)
    FoVy = focal2fov(focal_y, height)
    cam = cameras.MiniCam(width,height,FoVy,FoVx,0.2,1000,viewmat,projmat)
    intrins = intrins[:3,:3]
    colors = matplotlib.colormaps['Accent'](np.random.randint(1,64, 64)/64)[..., :3]
    colors = torch.from_numpy(colors).cuda()

    opacity = torch.ones_like(means3D[:,:1])
    kappas = torch.ones_like(means3D[:,:1]) * 10
    # gaussians = gaussian_model.GaussianModel(sh_degree=0)
    # gaussians.set_properties(means3D,colors,scales,quats,opacity,kappas)
    
    # our rendering 
    scales[:,2] = scales[:,2] * 0.1
    with torch.no_grad():
        image1, depthmap1, render_normal1,surf_normal1 = simple_render(
            cam, means3D,colors.float(),scales,quats,opacity,kappas,lamda=1.)
        image1 = image1.permute(1,2,0)
        depthmap1 = depthmap1.permute(1,2,0)
        image2, depthmap2, render_normal2,surf_normal2 = simple_render(
            cam, means3D,colors.float(),scales,quats,opacity,kappas,lamda=0.)
        image2 = image2.permute(1,2,0)
        depthmap2 = depthmap2.permute(1,2,0)
        
    # image1, depthmap1, center1, radii1, dist1 = surface_splatting(means3D, scales, quats, colors, opacity, intrins, viewmat, projmat)
    # image2, depthmap2, center2, radii2, dist2 = volume_splatting(means3D, scales, quats, colors, opacity, intrins, viewmat, projmat)

    # Visualize 3DGS and 2DGS
    fig1, (ax1,ax2) = plt.subplots(1,2)
    fig2, (ax3,ax4) = plt.subplots(1,2)

    from matplotlib.patches import Rectangle
    # point_image = center1.cpu().detach().numpy()
    # half_extend = radii1.cpu().numpy() * 1/3 # only show one sigma
    # lb = np.floor(point_image - half_extend)[..., :2]
    # hw = np.ceil(2*(half_extend)[..., :2])

    ax1.set_aspect('equal')
    ax1.set_axis_off()
    ax1.set_title('2D Gaussian splatting - color')
    ax2.set_aspect('equal')
    ax2.set_axis_off()
    ax2.set_title('3D Gaussian splatting - color')
    ax1.set_aspect('equal')
    ax1.set_axis_off()

    ax3.set_title('2D Gaussian splatting - depth')
    ax3.set_axis_off()
    ax3.set_aspect('equal')
    ax4.set_axis_off()
    ax4.set_title('3D Gaussian splatting - depth')
    fig1.tight_layout()
    # fig2.tight_layout()
    # visualize AABB
    # for k in range(len(half_extend)):
    #     ax1.add_patch(Rectangle(lb[k], hw[k, 0], hw[k, 1], facecolor='none', edgecolor='white'))
    #     # ax3.add_patch(Rectangle(lb[k], hw[k, 0], hw[k, 1], facecolor='none', edgecolor='white'))

    img1 = image1.cpu().numpy()
    img2 = image2.cpu().numpy()
    ax1.imshow(img1)
    ax2.imshow(img2)

    img1 = depthmap1.cpu().numpy()
    img2 = depthmap2.cpu().numpy()

    ax3.imshow(img1)
    ax4.imshow(img2)
    plt.show()
    # import open3d as o3d
    # rays_d = cam.get_rays()
    # pcd_t = rays_d.reshape(-1,3) * depthmap1.reshape(-1,1)
    # pcd = o3d.geometry.PointCloud()
    # pcd_2d = rays_d.reshape(-1,3) * depthmap2.reshape(-1,1)
    # pcd.points = o3d.utility.Vector3dVector(pcd_t.cpu().numpy())
    # pcd_2 = o3d.geometry.PointCloud()
    # pcd_2.points = o3d.utility.Vector3dVector(pcd_2d.cpu().numpy())
    # # o3d.visualization.draw_geometries([pcd,pcd_2])
    # o3d.visualization.draw_geometries([pcd])
    
    # plt.savefig('test1.png', transparent=True, dpi=300)
    
if __name__ == '__main__':
    main()