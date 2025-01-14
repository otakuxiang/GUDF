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
__device__ inline float log1_ex(float x){
	float a= 1+expf(x),res=logf(a);
	if (isinf(a)){
		res=x;
	}
	return res;
}

__device__ inline float computeMaxOpacity(const glm::vec3 &scale, const float kappa){
	return 1 - expf(-kappa * scale[2]);
}

__device__ float computeOpacityGUDF(const glm::vec3 &p_world, const float *normal, const float kappa, 
	const int W, const int H,float focal_x, float focal_y, const float *view2gaussian, const float2 &pixf, 
	const glm::vec3 &scale, float2 &t_range, float *int_depth, float *inter_depth
){ 
	if (scale[2] < 1e-6) return 1.0f;
	const glm::vec<3,double,glm::packed_highp> scaled(scale[0],scale[1],scale[2]);
	const glm::vec3 cam_pos = glm::vec3(view2gaussian[12], view2gaussian[13], view2gaussian[14]);
	glm::vec3 ray_view = glm::normalize(glm::vec3((pixf.x + 0.5 - W / 2.0f) / focal_x,
		(pixf.y + 0.5 - H / 2.0f) / focal_y, 1));		
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

	double A = glm::dot(ray_d_scaled, ray_d_scaled); 
	double B = 2 * glm::dot(ray_o_scaled, ray_d_scaled);
	double C = glm::dot(ray_o_scaled, ray_o_scaled) - 1;
	double delta = B*B - 4*A*C;
	if (delta < 0) {
		// ray does not intersect the ellipse
		return 1.0f;
	}
	
	double sd = sqrt(delta);
	float tn = (-B - sd) / (2*A);
	float tf = (-B + sd) / (2*A);	
	float depth_o = glm::dot(-cam_pos,normal_g) / glm::dot(ray_gauss,normal_g);
	// inter_depth[0] = depth_o;
	// if (tn > depth_o || tf < depth_o){
	// 	return 1.0f;
 	// }
	// float mid_d = (tf - tn) / 2.0f;

	// float mid_d = sqrt(expf(2*(logf(-B)-logf(A))) - 4 * expf(logf(C) - logf(A))) / 2;
	// tn = depth_o - mid_d;
	// tf = depth_o + mid_d;
	// tn = 0.0f;
	// tf = 100.0f;
	
	// compute the opacity
	t_range.y = tf;
	t_range.x = tn;
	float ftn = cos_theta * (depth_o - tn);
	float ftf = cos_theta * (depth_o - tf);

	float lnE = -kappa * ftf,lnF = -kappa * ftn;
	float lnB = log1_ex(lnE), lnC = log1_ex(lnF);
	float T = expf(lnC - lnB),int_T;
	if (lnC > 19){
		int_T = 1 / kappa / cos_theta * (1-T);
	}
	else{
		int_T = expf(lnC)/kappa/cos_theta*(kappa*cos_theta*(tf-tn)+lnC-lnB);
	}
	int_depth[0] = - T * tf + tn + int_T;
	return expf(lnC - lnB);
	// return expf(-kappa * cos_theta * mid_d);
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
__device__ void compute_transmat(
	const float3& p_orig,
	const glm::vec3 scale,
	float mod,
	const glm::vec4 rot,
	const float* projmatrix,
	const float* viewmatrix,
	const int W,
	const int H, 
	glm::mat3 &T,
	float3 &normal
) {

	glm::mat3 R = quat_to_rotmat(rot);
	glm::mat3 S = scale_to_mat(scale, mod);
	S[2][2] = 1.0f;
	glm::mat3 L = R * S;

	// center of Gaussians in the camera coordinate
	glm::mat3x4 splat2world = glm::mat3x4(
		glm::vec4(L[0], 0.0),
		glm::vec4(L[1], 0.0),
		glm::vec4(p_orig.x, p_orig.y, p_orig.z, 1)
	);

	glm::mat4 world2ndc = glm::mat4(
		projmatrix[0], projmatrix[4], projmatrix[8], projmatrix[12],
		projmatrix[1], projmatrix[5], projmatrix[9], projmatrix[13],
		projmatrix[2], projmatrix[6], projmatrix[10], projmatrix[14],
		projmatrix[3], projmatrix[7], projmatrix[11], projmatrix[15]
	);

	glm::mat3x4 ndc2pix = glm::mat3x4(
		glm::vec4(float(W) / 2.0, 0.0, 0.0, float(W-1) / 2.0),
		glm::vec4(0.0, float(H) / 2.0, 0.0, float(H-1) / 2.0),
		glm::vec4(0.0, 0.0, 0.0, 1.0)
	);

	T = glm::transpose(splat2world) * world2ndc * ndc2pix;
	normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);

}
__device__ void computeView2Gaussian(const glm::vec3& mean, const glm::vec4 rot, const float* viewmatrix,  float* view2gaussian)
{
	// glm matrices use column-major order
	// Normalize quaternion to get valid rotation
	// glm::vec4 q = rot;// / glm::length(rot);
	// float r = q.x;
	// float x = q.y;
	// float y = q.z;
	// float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = quat_to_rotmat(rot);

	// transform 3D points in gaussian coordinate system to world coordinate system as follows
	// new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
	// so the rots is the gaussian to world transform

	// Gaussian to world transform
	glm::mat4 G2W = glm::mat4(
		R[0][0], R[0][1], R[0][2], 0.0f,
		R[1][0], R[1][1], R[1][2], 0.0f,
		R[2][0], R[2][1], R[2][2], 0.0f,
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
__device__ bool compute_aabb(
	glm::mat3 T, 
	float cutoff,
	float2& point_image,
	float2& extent
) {
	glm::vec3 t = glm::vec3(cutoff * cutoff, cutoff * cutoff, -1.0f);
	float d = glm::dot(t, T[2] * T[2]);
	if (d == 0.0) return false;
	glm::vec3 f = (1 / d) * t;

	glm::vec2 p = glm::vec2(
		glm::dot(f, T[0] * T[2]),
		glm::dot(f, T[1] * T[2])
	);

	glm::vec2 h0 = p * p - 
		glm::vec2(
			glm::dot(f, T[0] * T[0]),
			glm::dot(f, T[1] * T[1])
		);

	glm::vec2 h = sqrt(max(glm::vec2(1e-4, 1e-4), h0));
	point_image = {p.x, p.y};
	extent = {h.x, h.y};
	return true;
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float* kappas,
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
	
	glm::mat3 T;
	float3 normal;
	if (transMat_precomp != nullptr)
	{
		glm::vec3 *T_ptr = (glm::vec3*)transMat_precomp;
		T = glm::mat3(
			T_ptr[idx * 3 + 0], 
			T_ptr[idx * 3 + 1],
			T_ptr[idx * 3 + 2]
		);
		normal = make_float3(0.0, 0.0, 1.0);
	}
	else
	{
		compute_transmat(((float3*)orig_points)[idx], scales[idx], scale_modifier, rotations[idx], projmatrix, viewmatrix, W, H, T, normal);
		float3 *T_ptr = (float3*)transMats;
		T_ptr[idx * 3 + 0] = {T[0][0], T[0][1], T[0][2]};
		T_ptr[idx * 3 + 1] = {T[1][0], T[1][1], T[1][2]};
		T_ptr[idx * 3 + 2] = {T[2][0], T[2][1], T[2][2]};
	}

#if DUAL_VISIABLE
	float cos = -sumf3(p_view * normal);
	if (cos == 0) return;
	float multiplier = cos > 0 ? 1: -1;
	normal = multiplier * normal;
#endif
	// add the bounding of countour
#if TIGHTBBOX // no use in the paper, but it indeed help speeds.
	// the effective extent is now depended on the opacity of gaussian.
	float truncated_R = sqrtf(max(9.f + logf(opacities[idx]), 0.000001));
#else
	float truncated_R = 3.f;
#endif
	float2 point_image;
	float radius;
	{
		float2 extent;
		bool ok = compute_aabb(T, truncated_R, point_image, extent);
		if (!ok) return;
		radius = ceil(max(max(extent.x, extent.y), truncated_R * FilterSize));
	}

	uint2 rect_min, rect_max;
	getRect(point_image, radius, rect_min, rect_max, grid);
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
	points_xy_image[idx] = point_image;
	// store them in float4
	normal_opacity[idx] = {normal.x, normal.y, normal.z, opacities[idx]};
	tiles_touched[idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x); 
	// if(idx == 10000){
	// 	printf("depth: %f\n", depths[idx]);
	// 	printf("radii: %d\n", radii[idx]);
	// 	printf("points_xy_image: %f %f\n", points_xy_image[idx].x, points_xy_image[idx].y);
	// 	printf("normal_opacity:%f %f %f %f\n", normal.x, normal.y, normal.z, opacities[idx]);
	// 	printf("tiles_touched: %d\n", tiles_touched[idx]);
	// }	
	// view to gaussian coordinate system
	computeView2Gaussian(p_world, rotations[idx], viewmatrix, view2gaussians + idx * 16);
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_2DGS(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ depths,
	const float4* __restrict__ normal_opacity,
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
	float2 pixf = { (float)pix.x, (float)pix.y};

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
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Fisrt compute two homogeneous planes, See Eq. (8)
			float2 xy = collected_xy[j];
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
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
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			// 2d screen distance
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y);
			float rho = min(rho3d, rho2d);
			
			float depth = (s.x * Tw.x + s.y * Tw.y) + Tw.z; // splat depth
			
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
			float alpha = min(0.99f, nor_o.w * exp(power) );
			
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
			float mapped_depth =  FAR_PLANE / (FAR_PLANE - NEAR_PLANE) * (1 - NEAR_PLANE / depth);
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

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_GUDF(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float* __restrict__ transMats,
	const float* __restrict__ view2gaussian,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ points3d,
	const float* __restrict__ kappas,
	const glm::vec3* __restrict__ scales,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	int* __restrict__ out_observe,
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
	float2 pixf = { (float)pix.x, (float)pix.y};

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
	__shared__ float3 collected_points3d[BLOCK_SIZE];
	__shared__ float collected_kappas[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];
	__shared__ glm::vec3 collected_scales[BLOCK_SIZE];
	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	// float last_depth = 0.0f;
	// float last_weight = 0.0f;
	float C[CHANNELS] = { 0 };


#if RENDER_AXUTILITY
	// render axutility ouput
	float D = { 0 };
	float N[3] = {0};
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
			collected_points3d[block.thread_rank()].x = points3d[3*coll_id];
			collected_points3d[block.thread_rank()].y = points3d[3*coll_id+1];
			collected_points3d[block.thread_rank()].z = points3d[3*coll_id+2];
			collected_kappas[block.thread_rank()] = kappas[coll_id];
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

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};


			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			// float alpha = min(0.99f, nor_o.w * exp(power));
			glm::vec3 p_world(collected_points3d[j].x, collected_points3d[j].y, collected_points3d[j].z);
			float2 t_tange;
			float UDF_opacity = 0,depth = 0,inter_depth = 0;
			UDF_opacity = 1 - computeOpacityGUDF(p_world,normal,collected_kappas[j],W,H,focal_x,focal_y,
				collected_view2gaussian+j*16,pixf,collected_scales[j],t_tange,&depth,&inter_depth);
			if (UDF_opacity < 1e-6f) continue; 
			if (depth < 0.1) continue;
			
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
			//
		 	// float alpha = min(0.99f, UDF_opacity);
			float alpha = min(0.99f, nor_o.w * UDF_opacity );
			// alpha = alpha / collected_max_alphas[j];
			if (alpha < 1.0f / 255.0f)
				continue;

			// alpha = alpha / collected_max_alphas[j];

			// if (pix_id == 70241 && collected_id[j] == 9356){
			// 	float *view2gaussian = collected_view2gaussian + j*16;
			// 	printf("%f %f\n%d %d\n%f %f\n%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f\n",
			// 		// depth,
			// 		pixf.x,pixf.y,
			// 		W,H,
			// 		focal_x,focal_y,
			// 		view2gaussian[0],view2gaussian[1],view2gaussian[2],view2gaussian[3],
			// 		view2gaussian[4],view2gaussian[5],view2gaussian[6],view2gaussian[7],
			// 		view2gaussian[8],view2gaussian[9],view2gaussian[10],view2gaussian[11],
			// 		view2gaussian[12],view2gaussian[13],view2gaussian[14],view2gaussian[15],
			// 		collected_scales[j].x,collected_scales[j].y,collected_scales[j].z,
			// 		normal[0],normal[1],normal[2],
			// 		p_world.x,p_world.y,p_world.z,
			// 		collected_kappas[j]
			// 	);
			// 	printf("UDF_opacity: %f, max_alpha: %f, nor_o.w: %f\n", UDF_opacity, collected_max_alphas[j], nor_o.w );
			// }

			float test_T = T * (1 - alpha);
			// if (pix_id == 70241){
			// 	printf("pix_id: %d, g_id: %d, 1 - alpha: %f, T: %f, test_T: %f\n",pix_id, collected_id[j], 1 - alpha, T, test_T);
			// }
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}


#if RENDER_AXUTILITY
			// Render depth distortion map
			// Efficient implementation of distortion loss, see 2DGS' paper appendix.

			// printf("mapped_depth: %f, error: %f, distortion: %f\n", mapped_depth, error, distortion);
 
			if (T > 0.5) {
				median_depth = depth;
				median_weight = nor_o.w * T;
				median_contributor = contributor;
			}
			// // Render normal map
			for (int ch=0; ch<3; ch++) N[ch] += normal[ch] * alpha * T;

			// // Render depth map
			D += nor_o.w * T * depth;
			// // // Efficient implementation of distortion loss, see 2DGS' paper appendix.

#endif

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;
			if(T > 0.5){
				atomicAdd(&(out_observe[collected_id[j]]), 1);
			}
			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
			// last_depth = mapped_depth;
			// last_weight = nor_o.w * T;
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
		out_others[pix_id + DEPTH_OFFSET * H * W] = D;
		out_others[pix_id + ALPHA_OFFSET * H * W] = 1 - T;
		for (int ch=0; ch<3; ch++) out_others[pix_id + (NORMAL_OFFSET+ch) * H * W] = N[ch];
		out_others[pix_id + MIDDEPTH_OFFSET * H * W] = median_depth;
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
	int* out_observe,
	float* out_color,
	float* out_others)
{	
	if(lambda == 0.0f){
		renderCUDA_2DGS<NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			means2D,
			colors,
			transMats,
			depths,
			normal_opacity,
			final_T,
			n_contrib,
			bg_color,
			out_color,
			out_others);
	}
	else{
		renderCUDA_GUDF<NUM_CHANNELS> << <grid, block >> > (
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			means2D,
			colors,
			transMats,
			view2gaussians,
			normal_opacity,
			means3D,
			kappas,
			scales,
			final_T,
			n_contrib,
			bg_color,
			out_observe,
			out_color,
			out_others);
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float* kappas,
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
		kappas,
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
