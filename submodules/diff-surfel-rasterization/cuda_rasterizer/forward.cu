/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// __device__ float analytic_integral(tn,tf,kappa,cos,depth){
// 	float ftn = cos * (depth - tn);
//     float ftf = cos * (depth - tf);
//     float E = expf(kappa * ftf);
//     float F = expf(kappa * ftn);
//     float A = expf(-kappa * cos * (tf - tn));
//     float B = 1 + E;
//     float C = 1 + F;
// 	return A * C / B;
// }

__device__ float computeOpacityGUDF(const glm::vec3 &p_world, const float *normal, const float kappa, const float depth,
	const int W, const int H,float focal_x, float focal_y, const float *view2gaussian, const float2 &pixf, 
	const glm::vec4 &quat, const glm::vec3 &scale, float2 &t_range
){ 

	const glm::vec3 cam_pos = glm::vec3(view2gaussian[12], view2gaussian[13], view2gaussian[14]);
	glm::vec3 ray_view = glm::normalize(glm::vec3((pixf.x - W / 2.0f) / focal_x,
		(pixf.y - H / 2.0f) / focal_y, 1));		
	glm::mat3 R = glm::mat3(view2gaussian[0], view2gaussian[1], view2gaussian[2],
		view2gaussian[4], view2gaussian[5], view2gaussian[6],
		view2gaussian[8], view2gaussian[9], view2gaussian[10]);
	// project normal ray to gaussian coordinate system
	glm::vec3 normal_g = R * glm::vec3(normal[0], normal[1], normal[2]);
	glm::vec3 ray_gauss = R * ray_view;
	float cos_theta = glm::dot(ray_gauss, normal_g);
	if(cos_theta < 0) cos_theta = -cos_theta; 
	glm::vec3 ray_o_scaled = glm::vec3(cam_pos[0] / 3 / (double)scale[0],cam_pos[1] / 3 / (double)scale[1],cam_pos[2] / 3 / (double)scale[2]);
	glm::vec3 ray_d_scaled = glm::vec3(ray_gauss[0] / 3 / (double)scale[0],ray_gauss[1] / 3 / (double)scale[1],ray_gauss[2] / 3 / (double)scale[2]);
	// compute the ray-ellipse intersection in the 2D plane

	float A = glm::dot(ray_d_scaled, ray_d_scaled); 
	float B = 2 * glm::dot(ray_o_scaled, ray_d_scaled);
	float C = glm::dot(ray_o_scaled, ray_o_scaled) - 1;
	float delta = B*B - 4*A*C;
	if (delta < 0) {
		// ray does not intersect the ellipse
		return 1.0f;
	}
	float sd = sqrt(delta);
	float tn = (-B - sd) / (2*A);
	float tf = (-B + sd) / (2*A);

	if(tf < depth || tn > depth){
		// intersection is outside the ellipse
		return 1.0f;
	}
	

	// compute the opacity
	t_range.y = tf;
	t_range.x = tn;
	float ftn = cos_theta * (depth - tn);
	float ftf = cos_theta * (depth - tf);
	float E = expf(kappa * ftf);
	float F = expf(kappa * ftn);
	A = expf(-kappa * cos_theta * (tf - tn));
	B = 1 + E;
	C = 1 + F;
	return A * C / B;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Compute a 2D-to-2D mapping matrix from a tangent plane into a image plane
// given a 2D gaussian parameters.
__device__ bool computeTransMat(const glm::vec3 &p_world, const glm::vec4 &quat, const glm::vec3 &scale, const float *viewmat, const float4 &intrins, float tan_fovx, float tan_fovy, float* transMat, float3 &normal) {
	// Setup cameras
	// Currently only support ideal pinhole camera
	// but more advanced intrins can be implemented
	const glm::mat3 W = glm::mat3(
		viewmat[0],viewmat[1],viewmat[2],
		viewmat[4],viewmat[5],viewmat[6],
		viewmat[8],viewmat[9],viewmat[10]
	); 
	const glm::vec3 cam_pos = glm::vec3(viewmat[12], viewmat[13], viewmat[14]); // camera center
	const glm::mat4 P = glm::mat4(
		intrins.x, 0.0, 0.0, 0.0,
		0.0, intrins.y, 0.0, 0.0,
		intrins.z, intrins.w, 1.0, 1.0,
		0.0, 0.0, 0.0, 0.0
	);

	// Make the geometry of 2D Gaussian as a Homogeneous transformation matrix 
	// under the camera view, See Eq. (5) in 2DGS' paper.
	glm::vec3 p_view = W * p_world + cam_pos;
	glm::mat3 R = quat_to_rotmat(quat) * scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
	glm::mat3 M = glm::mat3(W * R[0], W * R[1], p_view);
	glm::vec3 tn = W*R[2];
	float cos = glm::dot(-tn, p_view);

#if BACKFACE_CULL
	if (cos == 0.0f) return false;
#endif

#if RENDER_AXUTILITY and DUAL_VISIABLE
	// This means a 2D Gaussian is dual visiable.
	// Experimentally, turning off the dual visiable works eqully.
	float multiplier = cos > 0 ? 1 : -1;
	tn *= multiplier;
#endif
	// projection into screen space, see Eq. (7)
	glm::mat4x3 T = glm::transpose(P * glm::mat3x4(
		glm::vec4(M[0], 0.0),
		glm::vec4(M[1], 0.0),
		glm::vec4(M[2], 1.0)
	));

	transMat[0] = T[0].x;
	transMat[1] = T[0].y;
	transMat[2] = T[0].z;
	transMat[3] = T[1].x;
	transMat[4] = T[1].y;
	transMat[5] = T[1].z;
	transMat[6] = T[2].x;
	transMat[7] = T[2].y;
	transMat[8] = T[2].z;
	normal = {tn.x, tn.y, tn.z};
	return true;
}
__device__ void computeView2Gaussian(const glm::vec3& mean, const glm::vec4 rot, const float* viewmatrix,  float* view2gaussian)
{
	// glm matrices use column-major order
	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// transform 3D points in gaussian coordinate system to world coordinate system as follows
	// new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
	// so the rots is the gaussian to world transform

	// Gaussian to world transform
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[1][0], R[2][0], 0.0f,
		R[0][1], R[1][1], R[2][1], 0.0f,
		R[0][2], R[1][2], R[2][2], 0.0f,
		mean.x, mean.y, mean.z, 1.0f
	);

	// could be simplied by using pointer
	// viewmatrix is the world to view transformation matrix
	glm::mat4 W2V = glm::mat4(
		viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	);

	// Gaussian to view transform
	glm::mat4 G2V = W2V * G2W;

	// inverse of Gaussian to view transform
	// glm::mat4 V2G_inverse = glm::inverse(G2V);
	// R = G2V[:, :3, :3]
	// t = G2V[:, :3, 3]
	
	// t2 = torch.bmm(-R.transpose(1, 2), t[..., None])[..., 0]
	// V2G = torch.zeros((N, 4, 4), device='cuda')
	// V2G[:, :3, :3] = R.transpose(1, 2)
	// V2G[:, :3, 3] = t2
	// V2G[:, 3, 3] = 1.0
	glm::mat3 R_transpose = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);

	glm::vec3 t = glm::vec3(G2V[3][0], G2V[3][1], G2V[3][2]);
	glm::vec3 t2 = -R_transpose * t;

	view2gaussian[0] = R_transpose[0][0];
	view2gaussian[1] = R_transpose[0][1];
	view2gaussian[2] = R_transpose[0][2];
	view2gaussian[3] = 0.0f;
	view2gaussian[4] = R_transpose[1][0];
	view2gaussian[5] = R_transpose[1][1];
	view2gaussian[6] = R_transpose[1][2];
	view2gaussian[7] = 0.0f;
	view2gaussian[8] = R_transpose[2][0];
	view2gaussian[9] = R_transpose[2][1];
	view2gaussian[10] = R_transpose[2][2];
	view2gaussian[11] = 0.0f;
	view2gaussian[12] = t2.x;
	view2gaussian[13] = t2.y;
	view2gaussian[14] = t2.z;
	view2gaussian[15] = 1.0f;
}
// Computing the bounding box of the 2D Gaussian and its center,
// where the center of the bounding box is used to create a low pass filter
// in the image plane
__device__ bool computeAABB(const float *transMat, float2 & center, float2 & extent) {
	glm::mat4x3 T = glm::mat4x3(
		transMat[0], transMat[1], transMat[2],
		transMat[3], transMat[4], transMat[5],
		transMat[6], transMat[7], transMat[8],
		transMat[6], transMat[7], transMat[8]
	);

	float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);
	
	if (d == 0.0f) return false;

	glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]), 
		glm::dot(f, T[2] * T[3]));
	
	glm::vec3 h0 = p * p - 
		glm::vec3(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1]), 
			glm::dot(f, T[2] * T[2])
		);

	glm::vec3 h = sqrt(max(glm::vec3(0.0), h0)) + glm::vec3(0.0, 0.0, 1e-2);
	center = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, const float tan_fovy,
	const float focal_x, const float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* transMats,
	float* view2gaussians,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	glm::vec3 p_world = glm::vec3(orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2]);
	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;
	
	float4 intrins = {focal_x, focal_y, float(W)/2.0, float(H)/2.0};
	glm::vec3 scale = scales[idx];
	glm::vec4 quat = rotations[idx];
	
	const float* transMat;
	bool ok;
	float3 normal;
	if (transMat_precomp != nullptr)
	{
		transMat = transMat_precomp + idx * 9;
	}
	else
	{
		ok = computeTransMat(p_world, quat, scale, viewmatrix, intrins, tan_fovx, tan_fovy, transMats + idx * 9, normal);
		if (!ok) return;
		transMat = transMats + idx * 9;
	}
	
	//  compute center and extent
	float2 center;
	float2 extent;
	ok = computeAABB(transMat, center, extent);
	if (!ok) return;

	// add the bounding of countour
#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float truncated_R = sqrtf(max(9.f + logf(opacities[idx]), 0.000001));
#else
	float truncated_R = 3.f;
#endif
	float radius = ceil(truncated_R * max(max(extent.x, extent.y), FilterSize));
	uint2 rect_min, rect_max;
	getRect(center, radius, rect_min, rect_max, grid);
	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// compute colors 
	if (colors_precomp == nullptr) {
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	depths[idx] = p_view.z;
	radii[idx] = (int)radius;
	points_xy_image[idx] = center;
	// store them in float4
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
	// view to gaussian coordinate system
	computeView2Gaussian(p_world, rotations[idx], viewmatrix, view2gaussians + idx * 16);

}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ view2gaussian,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ points3d,
	const float* __restrict__ kappas,
	const glm::vec4* __restrict__ quats,
	const glm::vec3* __restrict__ scales,
	const float lambda, 
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	float* __restrict__ out_others)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x + 0.5, (float)pix.y + 0.5};

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float3 collected_points3d[BLOCK_SIZE];
	__shared__ float collected_kappas[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];
	__shared__ glm::vec4 collected_quats[BLOCK_SIZE];
	__shared__ glm::vec3 collected_scales[BLOCK_SIZE];
	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float D = { 0 };
	float N[3] = {0};
	float dist1 = {0};
	float dist2 = {0};
	float distortion = {0};
	float median_depth = {0};
	float median_weight = {0};
	float median_contributor = {-1};

#endif

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			collected_points3d[block.thread_rank()].x = points3d[3*coll_id];
			collected_points3d[block.thread_rank()].y = points3d[3*coll_id+1];
			collected_points3d[block.thread_rank()].z = points3d[3*coll_id+2];
			collected_kappas[block.thread_rank()] = kappas[coll_id];
			collected_quats[block.thread_rank()] = quats[coll_id];
			for (int ii = 0; ii < 16; ii++)
				collected_view2gaussian[16 * block.thread_rank() + ii] = view2gaussian[coll_id * 16 + ii];
			collected_scales[block.thread_rank()] = scales[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];
			float3 k = {-Tu.x + pixf.x * Tw.x, -Tu.y + pixf.x * Tw.y, -Tu.z + pixf.x * Tw.z};
			float3 l = {-Tv.x + pixf.y * Tw.x, -Tv.y + pixf.y * Tw.y, -Tv.z + pixf.y * Tw.z};
			// cross product of two planes is a line (i.e., homogeneous point), See Eq. (10)
			float3 p = crossProduct(k, l);
#if BACKFACE_CULL
			// May hanle this by replacing a low pass filter,
			// but this case is extremely rare.
			if (p.z == 0.0) continue; // there is not intersection
#endif
			// 3d homogeneous point to 2d point on the splat
			float2 s = {p.x / p.z, p.y / p.z};
			// 3d distance. Compute Mahalanobis distance in the canonical splat' space
			float rho3d = (s.x * s.x + s.y * s.y); 
			
			// Add low pass filter according to Botsch et al. [2005], 
			// see Eq. (11) from 2DGS paper. 
			float2 xy = collected_xy[j];
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			// 2d screen distance
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); 
			float rho = min(rho3d, rho2d);
			
			float depth = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z; // splat depth
			if (depth < NEAR_PLANE) continue;
			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};
			float power = -0.5f * rho;
			// power = -0.5f * 100.f * max(rho - 1, 0.0f);
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// float alpha = min(0.99f, nor_o.w * exp(power));
			glm::vec3 p_world(collected_points3d[j].x, collected_points3d[j].y, collected_points3d[j].z);
			float2 t_tange;
			float G = exp(power);
			float UDF_opacity = 0;
			if (lambda > 0.0f) {
				UDF_opacity = 1 - computeOpacityGUDF(p_world,normal,collected_kappas[j],depth,W,H,focal_x,focal_y,
					collected_view2gaussian+j*16,pixf,collected_quats[j],collected_scales[j],t_tange);
			}
			
			// if(pix.x == W / 2 && pix.y == H / 2 && UDF_opacity > 0.0f){
			// 	// printf("pixf: %f %f\n",pixf.x,pixf.y);
			// 	// printf("W H: %d %d\n",W,H);
			// 	// printf("focal_x focal_y: %f %f\n",focal_x,focal_y);
			// 	// // print viewmat
			// 	// printf("viewmat: ");
			// 	// for(int i=0;i<4;i++){
			// 	// 	for(int j=0;j<4;j++){
			// 	// 		printf("%f ",viewmat[i*4+j]);
			// 	// 	}
			// 	// 	printf("\n");
			// 	// }
			// 	// printf("quat: %f %f %f %f\n",collected_quats[j].x,collected_quats[j].y,collected_quats[j].z,collected_quats[j].w);	
			// 	// glm::mat3 q_rot = quat_to_rotmat(collected_quats[j]);
			// 	// for(int i=0;i<3;i++){
			// 	// 	for(int j=0;j<3;j++){
			// 	// 		printf("%f ",q_rot[i][j]);
			// 	// 	}
			// 	// 	printf("\n");
			// 	// }
			// 	// printf("scale: %f %f\n",collected_scales[j].x,collected_scales[j].y); 
			// 	// printf("normal: %f %f %f\n",normal[0],normal[1],normal[2]);
			// 	// printf("p_world: %f %f %f\n",p_world.x,p_world.y,p_world.z);
			// 	// printf("kappa: %f\n",collected_kappas[j]);
			// 	// // printf("t_tange: %f %f\n",t_tange.x,t_tange.y);
			// 	printf("UDF_opacity G: %f %f\n",UDF_opacity,G);
			// }
		 	// float alpha = min(0.99f, nor_o.w * UDF_opacity * G);
			float alpha = min(0.99f, nor_o.w * (lambda * UDF_opacity + (1 - lambda) * G));

			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}


#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			float A = 1-T;
			float mapped_depth = (FAR_PLANE * depth - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * depth);
			float error = mapped_depth * mapped_depth * A + dist2 - 2 * mapped_depth * dist1;
			distortion += error * alpha * T;

			if (T > 0.5) {
				median_depth = depth;
				median_weight = alpha * T;
				median_contributor = contributor;
			}
			// Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * alpha * T;

			// Render depth map
			D += depth * alpha * T;
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.
			dist1 += mapped_depth * alpha * T;
			dist2 += mapped_depth * mapped_depth * alpha * T;
#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

#if RENDER_AXUTILITY
		n_contrib[pix_id + H * W] = median_contributor;
		final_T[pix_id + H * W] = dist1;
		final_T[pix_id + 2 * H * W] = dist2;
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
		out_others[pix_id + DISTORTION_OFFSET * H * W] = distortion;
		out_others[pix_id + MEDIAN_WEIGHT_OFFSET * H * W] = median_weight;
#endif
	}
}


void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* means3D,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float2* means2D,
	const float* kappas,
	const float* viewmatrix,
	const float* colors,
	const float* transMats,
	const float* view2gaussians,
	const float* depths,
	const float4* normal_opacity,
	const float lambda,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	float* out_others)
{	
	renderCUDA<NUM_CHANNELS> << <grid, block >> > (
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		means2D,
		colors,
		transMats,
		view2gaussians,
		depths,
		normal_opacity,
		means3D,
		kappas,
		rotations,
		scales,
		lambda,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		out_others);
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* transMat_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, const int H,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* transMats,
	float* view2gaussians,
	float* rgb,
	float4* normal_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		transMat_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		transMats,
		view2gaussians,
		rgb,
		normal_opacity,
		grid,
		tiles_touched,
		prefiltered
		);
}
