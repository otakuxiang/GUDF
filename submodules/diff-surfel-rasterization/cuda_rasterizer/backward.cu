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

#include "backward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

// backward pass of GUDF opacity
__device__ float computeOpacityGUDF(const glm::vec3 &p_world, const float *normal, const float kappa, const float depth, 
	const int W, const int H,float focal_x, float focal_y, const float *view2gaussian, const float2 &pixf, 
	const glm::vec4 &quat, const glm::vec3 &scale, float2 &t_range, float* dT_dkappa, float *dT_dV2G,glm::vec3 &dT_dnormal3D,
	float *dT_dscale
){ 
	const glm::vec<3,double,glm::packed_highp> scaled(scale[0],scale[1],scale[2]);
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
    glm::vec3 dcosdn = ray_gauss;
	glm::vec3 dcosdr = normal_g - glm::dot(ray_gauss,normal_g) * ray_gauss;
	if(cos_theta < 0){
		cos_theta = -cos_theta; 
		dcosdn = -dcosdn;
		dcosdr = -dcosdr;
	}
	float dnd = glm::dot(normal_g, ray_gauss);
	float don = glm::dot(-cam_pos, normal_g);
	glm::vec3 ddepthdn = -cam_pos / dnd - don/dnd/dnd * ray_gauss;
	glm::vec3 ddepthdr = - don/ dnd / dnd * normal_g;
	glm::vec3 ddepthdo = - normal_g / dnd; 

	glm::vec3 ray_o_scaled = glm::vec3(cam_pos[0] / 3 / scaled[0],cam_pos[1] / 3 / scaled[1],cam_pos[2] / 3 / scaled[2]);
	glm::vec3 ray_d_scaled = glm::vec3(ray_gauss[0] / 3 / scaled[0],ray_gauss[1] / 3 / scaled[1],ray_gauss[2] / 3 / scaled[2]);
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
	float depth_o = glm::dot(-cam_pos,normal_g) / glm::dot(ray_gauss,normal_g);
	float flagtf = depth_o < tf ? -1 : 1, flagtn = depth_o > tn ? 1 : -1;
	glm::vec3 p_tn = ray_o_scaled + ray_d_scaled * tn;
	glm::vec3 p_tf = ray_o_scaled + ray_d_scaled * tf;
	
	float tnogrg = glm::dot(p_tn,ray_d_scaled);
	float tfogrg = glm::dot(p_tf,ray_d_scaled);



	// compute the opacity
	t_range.y = tf;
	t_range.x = tn;
	float ftn = cos_theta * (depth_o - tn);
	float ftf = cos_theta * (depth_o - tf);
	float E = expf(kappa * ftf);
	float F = expf(kappa * ftn);
	const float kappacos = kappa * cos_theta;
	float lnE = kappa * ftf,lnF = kappa * ftn;
	float lnA = -kappacos * (tf - tn) ,lnB = logf(1+E), lnC = logf(1+F);
	if(isinf(E)  || isnan(E)){
		if (lnE > 0) lnB = lnE;
		else lnB = 1e-10;

	} 
	if(isinf(F) || isnan(F)){
		if (lnF > 0) lnC = lnF;
		else lnC = 1e-10;
	} 
	float lncos = logf(cos_theta),lnkappa = logf(kappa),lntfd = logf(abs(tf - depth_o)),lntnd = logf(abs(depth_o - tn));
	float lnkc = lncos + lnkappa, lntftn = logf(tf - tn);
	float lndTdA = lnC - lnB, lndTdB = lnA + lnC - 2 * lnB, lndTdC = lnA - lnB;

	// A = expf(-kappacos * (tf - tn));
	// B = 1 + E;
	// C = 1 + F;
	// float B2 = B*B;
	

	// float dA_dkappa = -cos_theta * (tf - tn) * A;
	// float dB_dkappa = E * ftf;
	// float dC_dkappa = F * ftn;
	// float dT_dA = C / B;
	// float dT_dB = -A * C / B2;
	// float dT_dC = A / B;
	
	// dT_dkappa[0] = (dT_dA * dA_dkappa + dT_dB * dB_dkappa + dT_dC * dC_dkappa) ;
	
	// float lndAdkappa = lncos + lntftn + lnA,lndBdkappa = lnE + lncos + lntfd,
	// 	lndCdkappa = lnF + lncos + lntnd; 
	dT_dkappa[0] = - expf(lncos + lntftn + lnA + lndTdA) - flagtf * expf(lndTdB + lnE + lncos + lntfd) 
		+ flagtn * expf(lndTdC + lnF + lncos + lntnd);
	// float dBddepth = kappacos * E;
	// float dCddepth = kappacos * F;
	// float dT_ddepth = dT_dC * dCddepth + dT_dB * dBddepth;
	// float lndB_ddepth = lnkc + lnE,lndCddepth = lnkc + lnF;
	float dT_ddepth = expf(lndTdC + lnkc + lnF) - expf(lndTdB + lnkc + lnE);

	// dAdtn = kappa * cos * A;  dCdtn = - F * kappa * cos
	// float dTdtn = dT_dA * kappacos * A - dT_dC * kappacos * F;
	// dAdtf = - kappa * cos * A; dBdtf = - E * kappa * cos
	// float dTdtf = - dT_dA * kappacos * A - dT_dB * kappacos * E;
	float dTdtn = expf(lndTdA + lnkc + lnA) - expf(lndTdC + lnkc + lnF);
	float dTdtf = -expf(lndTdA + lnkc + lnA) + expf(lndTdB + lnkc + lnE);

	// dAdcos = (tn - tf) * A * kappa, dBdcos = E * kappa * (depth - tf), dCdcos = F * kappa * (depth - tn)
	// float dAdcos = (tn - tf) * A * kappa;
	// float dBdcos = E * kappa * (depth_o - tf);
	// float dCdcos = F * kappa * (depth_o - tn);
	// float dTdcos = dT_dA * dAdcos + dT_dB * dBdcos + dT_dC * dCdcos;
	float dTdcos = -expf(lndTdA + lntftn + lnkappa + lnA) - flagtf * expf(lndTdB + lntfd + lnkappa + lnE) + flagtn * expf(lndTdC + lntnd + lnkappa + lnF); 
	glm::vec3 dTdno = dT_ddepth * ddepthdn + dTdcos * dcosdn;
	dT_dnormal3D[0] = dTdno[0] * view2gaussian[0] + dTdno[1] * view2gaussian[1] + dTdno[2] * view2gaussian[2];
	dT_dnormal3D[1] = dTdno[0] * view2gaussian[4] + dTdno[1] * view2gaussian[5] + dTdno[2] * view2gaussian[6];
	dT_dnormal3D[2] = dTdno[0] * view2gaussian[8] + dTdno[1] * view2gaussian[9] + dTdno[2] * view2gaussian[10];

	// dtndrg = - tn / tnogrg * p_tn; dtfdrg = - tf / tfogrg * p_tf;
	glm::vec3 dTdrg = - dTdtn * tn / tnogrg * p_tn - dTdtf * tf / tfogrg * p_tf;
	// dtndog = - p_tn / tnogrg; dtfdog = - p_tf / tfogrg;
	glm::vec3 dTdog = - dTdtn * p_tn / tnogrg - dTdtf * p_tf / tfogrg;

	glm::vec3 dTdo = glm::vec3(dTdog[0] / 3 / scaled[0],dTdog[1] / 3 / scaled[1],dTdog[2] / 3 / scaled[2]);
	dTdo = dTdo + dT_ddepth * ddepthdo;
	glm::vec3 dTdr = glm::vec3(dTdrg[0] / 3 / scaled[0],dTdrg[1] / 3 / scaled[1],dTdrg[2] / 3 / scaled[2]);
	dTdr = dTdr + dT_ddepth * ddepthdr + dTdcos * dcosdr; 

	dT_dV2G[0] = dTdr[0] * ray_view[0] + dTdno[0] * normal[0];
	dT_dV2G[1] = dTdr[1] * ray_view[0] + dTdno[1] * normal[0];
	dT_dV2G[2] = dTdr[2] * ray_view[0] + dTdno[2] * normal[0];
	dT_dV2G[3] = 0;
	dT_dV2G[4] = dTdr[0] * ray_view[1] + dTdno[0] * normal[1]; 
	dT_dV2G[5] = dTdr[1] * ray_view[1] + dTdno[1] * normal[1];
	dT_dV2G[6] = dTdr[2] * ray_view[1] + dTdno[2] * normal[1];
	dT_dV2G[7] = 0;
	dT_dV2G[8] = dTdr[0] * ray_view[2] + dTdno[0] * normal[2];
	dT_dV2G[9] = dTdr[1] * ray_view[2] + dTdno[1] * normal[2];
	dT_dV2G[10] = dTdr[2] * ray_view[2] + dTdno[2] * normal[2];
	dT_dV2G[11] = 0;
	dT_dV2G[12] = dTdo[0];
	dT_dV2G[13] = dTdo[1]; 
	dT_dV2G[14] = dTdo[2];
	dT_dV2G[15] = 0;
	glm::vec3 drg_dscale(0,0,0);
	double s02 = 1 / (scaled[0] * scaled[0]);
	double s12 = 1 / (scaled[1] * scaled[1]);
	double s22 = 1 / (scaled[2] * scaled[2]);
	drg_dscale[0] = -ray_gauss[0] / 3 * s02;
	drg_dscale[1] = -ray_gauss[1] / 3 * s12;
	drg_dscale[2] = -ray_gauss[2] / 3 * s22;

	glm::vec3 dog_dscale(0,0,0);
	dog_dscale[0] = -cam_pos[0] / 3 * s02;
	dog_dscale[1] = -cam_pos[1] / 3 * s12;
	dog_dscale[2] = -cam_pos[2] / 3 * s22;

	dT_dscale[0] = dTdrg[0] * drg_dscale[0] + dTdog[0] * dog_dscale[0];
	dT_dscale[1] = dTdrg[1] * drg_dscale[1] + dTdog[1] * dog_dscale[1];
	dT_dscale[2] = dTdrg[2] * drg_dscale[2] + dTdog[2] * dog_dscale[2];
	// check if dT_dscale is too large
	// if(isnan(dT_dscale[0]) || isinf(dT_dscale[0]) || isnan(dT_dscale[1]) || isinf(dT_dscale[1]) || isnan(dT_dscale[2]) || isinf(dT_dscale[2])){
	// 	printf("dTdcos %f\n", dTdcos);
	// 	printf("pixf: %f %f\n",pixf.x,pixf.y);
	// 	printf("W H: %d %d\n",W,H);
	// 	printf("focal_x focal_y: %f %f\n",focal_x,focal_y);
	// 	// print viewmat
	// 	printf("V2G: ");
	// 	for(int i=0;i<4;i++){
	// 		for(int j=0;j<4;j++){
	// 			printf("%f ",view2gaussian[i*4+j]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	// printf("quat: %f %f %f %f\n",collected_quats[j].x,collected_quats[j].y,collected_quats[j].z,collected_quats[j].w);	
	// 	printf("q_rot:\n");
	// 	glm::mat3 q_rot = quat_to_rotmat(quat);
	// 	for(int i=0;i<3;i++){
	// 		for(int j=0;j<3;j++){
	// 			printf("%f ",q_rot[i][j]);
	// 		}
	// 		printf("\n");
	// 	}
	// 	printf("scale: %f %f %f\n",scale.x,scale.y,scale.z); 
	// 	printf("normal: %f %f %f\n",normal[0],normal[1],normal[2]);
	// 	printf("p_world: %f %f %f\n",p_world.x,p_world.y,p_world.z);
	// 	printf("kappa: %f\n",kappa);
	// }

	return expf(lnA + lnC - lnB);
}


// Backward pass for conversion of spherical harmonics to RGB for
// each Gaussian.
__device__ void computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, const bool* clamped, const glm::vec3* dL_dcolor, glm::vec3* dL_dmeans, glm::vec3* dL_dshs)
{
	// Compute intermediate values, as it is done during forward
	glm::vec3 pos = means[idx];
	glm::vec3 dir_orig = pos - campos;
	glm::vec3 dir = dir_orig / glm::length(dir_orig);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;

	// Use PyTorch rule for clamping: if clamping was applied,
	// gradient becomes 0.
	glm::vec3 dL_dRGB = dL_dcolor[idx];
	dL_dRGB.x *= clamped[3 * idx + 0] ? 0 : 1;
	dL_dRGB.y *= clamped[3 * idx + 1] ? 0 : 1;
	dL_dRGB.z *= clamped[3 * idx + 2] ? 0 : 1;

	glm::vec3 dRGBdx(0, 0, 0);
	glm::vec3 dRGBdy(0, 0, 0);
	glm::vec3 dRGBdz(0, 0, 0);
	float x = dir.x;
	float y = dir.y;
	float z = dir.z;

	// Target location for this Gaussian to write SH gradients to
	glm::vec3* dL_dsh = dL_dshs + idx * max_coeffs;

	// No tricks here, just high school-level calculus.
	float dRGBdsh0 = SH_C0;
	dL_dsh[0] = dRGBdsh0 * dL_dRGB;
	if (deg > 0)
	{
		float dRGBdsh1 = -SH_C1 * y;
		float dRGBdsh2 = SH_C1 * z;
		float dRGBdsh3 = -SH_C1 * x;
		dL_dsh[1] = dRGBdsh1 * dL_dRGB;
		dL_dsh[2] = dRGBdsh2 * dL_dRGB;
		dL_dsh[3] = dRGBdsh3 * dL_dRGB;

		dRGBdx = -SH_C1 * sh[3];
		dRGBdy = -SH_C1 * sh[1];
		dRGBdz = SH_C1 * sh[2];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;

			float dRGBdsh4 = SH_C2[0] * xy;
			float dRGBdsh5 = SH_C2[1] * yz;
			float dRGBdsh6 = SH_C2[2] * (2.f * zz - xx - yy);
			float dRGBdsh7 = SH_C2[3] * xz;
			float dRGBdsh8 = SH_C2[4] * (xx - yy);
			dL_dsh[4] = dRGBdsh4 * dL_dRGB;
			dL_dsh[5] = dRGBdsh5 * dL_dRGB;
			dL_dsh[6] = dRGBdsh6 * dL_dRGB;
			dL_dsh[7] = dRGBdsh7 * dL_dRGB;
			dL_dsh[8] = dRGBdsh8 * dL_dRGB;

			dRGBdx += SH_C2[0] * y * sh[4] + SH_C2[2] * 2.f * -x * sh[6] + SH_C2[3] * z * sh[7] + SH_C2[4] * 2.f * x * sh[8];
			dRGBdy += SH_C2[0] * x * sh[4] + SH_C2[1] * z * sh[5] + SH_C2[2] * 2.f * -y * sh[6] + SH_C2[4] * 2.f * -y * sh[8];
			dRGBdz += SH_C2[1] * y * sh[5] + SH_C2[2] * 2.f * 2.f * z * sh[6] + SH_C2[3] * x * sh[7];

			if (deg > 2)
			{
				float dRGBdsh9 = SH_C3[0] * y * (3.f * xx - yy);
				float dRGBdsh10 = SH_C3[1] * xy * z;
				float dRGBdsh11 = SH_C3[2] * y * (4.f * zz - xx - yy);
				float dRGBdsh12 = SH_C3[3] * z * (2.f * zz - 3.f * xx - 3.f * yy);
				float dRGBdsh13 = SH_C3[4] * x * (4.f * zz - xx - yy);
				float dRGBdsh14 = SH_C3[5] * z * (xx - yy);
				float dRGBdsh15 = SH_C3[6] * x * (xx - 3.f * yy);
				dL_dsh[9] = dRGBdsh9 * dL_dRGB;
				dL_dsh[10] = dRGBdsh10 * dL_dRGB;
				dL_dsh[11] = dRGBdsh11 * dL_dRGB;
				dL_dsh[12] = dRGBdsh12 * dL_dRGB;
				dL_dsh[13] = dRGBdsh13 * dL_dRGB;
				dL_dsh[14] = dRGBdsh14 * dL_dRGB;
				dL_dsh[15] = dRGBdsh15 * dL_dRGB;

				dRGBdx += (
					SH_C3[0] * sh[9] * 3.f * 2.f * xy +
					SH_C3[1] * sh[10] * yz +
					SH_C3[2] * sh[11] * -2.f * xy +
					SH_C3[3] * sh[12] * -3.f * 2.f * xz +
					SH_C3[4] * sh[13] * (-3.f * xx + 4.f * zz - yy) +
					SH_C3[5] * sh[14] * 2.f * xz +
					SH_C3[6] * sh[15] * 3.f * (xx - yy));

				dRGBdy += (
					SH_C3[0] * sh[9] * 3.f * (xx - yy) +
					SH_C3[1] * sh[10] * xz +
					SH_C3[2] * sh[11] * (-3.f * yy + 4.f * zz - xx) +
					SH_C3[3] * sh[12] * -3.f * 2.f * yz +
					SH_C3[4] * sh[13] * -2.f * xy +
					SH_C3[5] * sh[14] * -2.f * yz +
					SH_C3[6] * sh[15] * -3.f * 2.f * xy);

				dRGBdz += (
					SH_C3[1] * sh[10] * xy +
					SH_C3[2] * sh[11] * 4.f * 2.f * yz +
					SH_C3[3] * sh[12] * 3.f * (2.f * zz - xx - yy) +
					SH_C3[4] * sh[13] * 4.f * 2.f * xz +
					SH_C3[5] * sh[14] * (xx - yy));
			}
		}
	}

	// The view direction is an input to the computation. View direction
	// is influenced by the Gaussian's mean, so SHs gradients
	// must propagate back into 3D position.
	glm::vec3 dL_ddir(glm::dot(dRGBdx, dL_dRGB), glm::dot(dRGBdy, dL_dRGB), glm::dot(dRGBdz, dL_dRGB));

	// Account for normalization of direction
	float3 dL_dmean = dnormvdv(float3{ dir_orig.x, dir_orig.y, dir_orig.z }, float3{ dL_ddir.x, dL_ddir.y, dL_ddir.z });

	// Gradients of loss w.r.t. Gaussian means, but only the portion 
	// that is caused because the mean affects the view-dependent color.
	// Additional mean gradient is accumulated in below methods.
	dL_dmeans[idx] += glm::vec3(dL_dmean.x, dL_dmean.y, dL_dmean.z);
}


__device__ void computeView2Gaussian_backward(
	int idx, 
	const float3& mean, 
	const glm::vec4 rot, 
	const float* viewmatrix,  
	const float* view2gaussian, 
	const float* dL_dview2gaussian,
	glm::vec3* dL_dmeans, 
	glm::vec4* dL_drots)
{
	// Recompute (intermediate) results for the 3D covariance computation.
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

	glm::mat4 W2V = glm::mat4(
		viewmatrix[0], viewmatrix[1], viewmatrix[2], viewmatrix[3],
		viewmatrix[4], viewmatrix[5], viewmatrix[6], viewmatrix[7],
		viewmatrix[8], viewmatrix[9], viewmatrix[10], viewmatrix[11],
		viewmatrix[12], viewmatrix[13], viewmatrix[14], viewmatrix[15]
	);

	// Gaussian to view transform
	glm::mat4 G2V = W2V * G2W;

	// inverse of Gaussian to view transform
	glm::mat4 V2G = glm::inverse(G2V);

	// compute the gradient here
	// glm::mat4 V2G = glm::inverse(G2V);
	// G2V = [R, t], V2G = inverse(G2V) = [R^T, -R^T * t]
	// V2G_R = G2V_R^T
	// V2G_t = -G2V_R^T * G2V_t
	glm::mat3 G2V_R_t = glm::mat3(
		G2V[0][0], G2V[1][0], G2V[2][0],
		G2V[0][1], G2V[1][1], G2V[2][1],
		G2V[0][2], G2V[1][2], G2V[2][2]
	);
	glm::mat3 G2V_R = glm::transpose(G2V_R_t);
	glm::vec3 G2V_t = glm::vec3(
		G2V[3][0], G2V[3][1], G2V[3][2]
	);

	// dL_dG2V_R = dL_dV2G_R^T
	// dL_dG2V_t = -dL_dV2G_t * G2V_R^T
	glm::mat3 dL_dV2G_R_t = glm::mat3(
		dL_dview2gaussian[0], dL_dview2gaussian[4], dL_dview2gaussian[8],
		dL_dview2gaussian[1], dL_dview2gaussian[5], dL_dview2gaussian[9],
		dL_dview2gaussian[2], dL_dview2gaussian[6], dL_dview2gaussian[10]
	);
	glm::vec3 dL_dV2G_t = glm::vec3(
		dL_dview2gaussian[12], dL_dview2gaussian[13], dL_dview2gaussian[14]
	);

	// also gradient from -R^T * t
	glm::mat3 dL_dG2V_R_from_t = glm::mat3(
		-dL_dV2G_t.x * G2V_t.x, -dL_dV2G_t.x * G2V_t.y, -dL_dV2G_t.x * G2V_t.z,
		-dL_dV2G_t.y * G2V_t.x, -dL_dV2G_t.y * G2V_t.y, -dL_dV2G_t.y * G2V_t.z,
		-dL_dV2G_t.z * G2V_t.x, -dL_dV2G_t.z * G2V_t.y, -dL_dV2G_t.z * G2V_t.z
	);

	// TODO:
	glm::mat3 dL_dG2V_R = dL_dV2G_R_t + dL_dG2V_R_from_t;
	glm::vec3 dL_dG2V_t = -dL_dV2G_t * G2V_R_t;

	// dL_dG2V = [dL_dG2V_R, dL_dG2V_t]
	glm::mat4 dL_dG2V = glm::mat4(
		dL_dG2V_R[0][0], dL_dG2V_R[0][1], dL_dG2V_R[0][2], 0.0f,
		dL_dG2V_R[1][0], dL_dG2V_R[1][1], dL_dG2V_R[1][2], 0.0f,
		dL_dG2V_R[2][0], dL_dG2V_R[2][1], dL_dG2V_R[2][2], 0.0f,
		dL_dG2V_t.x, dL_dG2V_t.y, dL_dG2V_t.z, 0.0f
	);

	// Gaussian to view transform
	// glm::mat4 G2V = W2V * G2W;
	glm::mat4 dL_dG2W = glm::transpose(W2V) * dL_dG2V;

	
	// Gaussian to world transform
	// glm::mat4 G2W = glm::mat4(
	// 	R[0][0], R[1][0], R[2][0], 0.0f,
	// 	R[0][1], R[1][1], R[2][1], 0.0f,
	// 	R[0][2], R[1][2], R[2][2], 0.0f,
	// 	mean.x, mean.y, mean.z, 1.0f
	// );
	// dL_dG2W_R = dL_dG2W_R^T
	// dL_dG2W_t = dL_dG2W_t
	glm::mat3 dL_dG2W_R = glm::mat3(
		dL_dG2W[0][0], dL_dG2W[0][1], dL_dG2W[0][2],
		dL_dG2W[1][0], dL_dG2W[1][1], dL_dG2W[1][2],
		dL_dG2W[2][0], dL_dG2W[2][1], dL_dG2W[2][2]
	);
	glm::vec3 dL_dG2W_t = glm::vec3(
		dL_dG2W[3][0], dL_dG2W[3][1], dL_dG2W[3][2]
	);
	glm::mat3 dL_dR = dL_dG2W_R;

	// Gradients of loss w.r.t. means
	glm::vec3* dL_dmean = dL_dmeans + idx;
	dL_dmean->x += dL_dG2W_t.x;
	dL_dmean->y += dL_dG2W_t.y;
	dL_dmean->z += dL_dG2W_t.z;

	glm::mat3 dL_dMt = dL_dR;

	// // Gradients of loss w.r.t. normalized quaternion
	glm::vec4 dL_dq;
	dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

	// Gradients of loss w.r.t. unnormalized quaternion
	float4* dL_drot = (float4*)(dL_drots + idx);
	dL_drot->x += dL_dq.x;
	dL_drot->y += dL_dq.y;
	dL_drot->z += dL_dq.z;
	dL_drot->w += dL_dq.w;
	// *dL_drot = float4{ dL_dq.x, dL_dq.y, dL_dq.z, dL_dq.w };
}


// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float3* __restrict__ points3d,
	const float* __restrict__ kappas,
	const glm::vec4* __restrict__ quats,
	const glm::vec3* __restrict__ scales,
	const float* __restrict__ viewmat,
	const float* __restrict__ transMats,
	const float* __restrict__ view2gaussian,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float lambda, 
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float * __restrict__ dL_dview2gaussians,
	glm::vec3* __restrict__ dL_dscale,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors,
	float* __restrict__ dL_dkappas
	)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x + 0.5, (float)pix.y + 0.5};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_Tu[BLOCK_SIZE];
	__shared__ float3 collected_Tv[BLOCK_SIZE];
	__shared__ float3 collected_Tw[BLOCK_SIZE];
	__shared__ float3 collected_points3d[BLOCK_SIZE];
	__shared__ float collected_kappas[BLOCK_SIZE];
	__shared__ glm::vec4 collected_quats[BLOCK_SIZE];
	__shared__ glm::vec3 collected_scales[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];
	// __shared__ float collected_depths[BLOCK_SIZE];

	// In the forward, we stored the final value for T, the
	// product of all (1 - alpha) factors. 
	const float T_final = inside ? final_Ts[pix_id] : 0;
	float T = T_final;

	// We start from the back. The ID of the last contributing
	// Gaussian is known from each pixel from the forward.
	uint32_t contributor = toDo;
	const int last_contributor = inside ? n_contrib[pix_id] : 0;

	float accum_rec[C] = { 0 };
	float dL_dpixel[C];

#if RENDER_AXUTILITY
	float dL_dreg;
	float dL_ddepth;
	float dL_daccum;
	float dL_dnormal2D[3];
	const int median_contributor = inside ? n_contrib[pix_id + H * W] : 0;
	float dL_dmedian_depth;
	float dL_dmax_dweight;

	if (inside) {
		dL_ddepth = dL_depths[DEPTH_OFFSET * H * W + pix_id];
		dL_daccum = dL_depths[ALPHA_OFFSET * H * W + pix_id];
		dL_dreg = dL_depths[DISTORTION_OFFSET * H * W + pix_id];
		for (int i = 0; i < 3; i++) 
			dL_dnormal2D[i] = dL_depths[(NORMAL_OFFSET + i) * H * W + pix_id];

		dL_dmedian_depth = dL_depths[MIDDEPTH_OFFSET * H * W + pix_id];
		dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
	}

	// for compute gradient with respect to depth and normal
	float last_depth = 0;
	float last_normal[3] = { 0 };
	float accum_depth_rec = 0;
	float accum_alpha_rec = 0;
	float accum_normal_rec[3] = {0};
	// for compute gradient with respect to the distortion map
	const float final_D = inside ? final_Ts[pix_id + H * W] : 0;
	const float final_D2 = inside ? final_Ts[pix_id + 2 * H * W] : 0;
	const float final_A = 1 - T_final;
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_color[C] = { 0 };

	// Gradient of pixel coordinate w.r.t. normalized 
	// screen-space viewport corrdinates (-1 to 1)
	const float ddelx_dx = 0.5 * W;
	const float ddely_dy = 0.5 * H;

	// Traverse all Gaussians
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// Load auxiliary data into shared memory, start in the BACK
		// and load them in revers order.
		block.sync();
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			const int coll_id = point_list[range.y - progress - 1];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_normal_opacity[block.thread_rank()] = normal_opacity[coll_id];
			collected_Tu[block.thread_rank()] = {transMats[9 * coll_id+0], transMats[9 * coll_id+1], transMats[9 * coll_id+2]};
			collected_Tv[block.thread_rank()] = {transMats[9 * coll_id+3], transMats[9 * coll_id+4], transMats[9 * coll_id+5]};
			collected_Tw[block.thread_rank()] = {transMats[9 * coll_id+6], transMats[9 * coll_id+7], transMats[9 * coll_id+8]};
			if (lambda > 0.0f){
				collected_points3d[block.thread_rank()] = points3d[coll_id];
				collected_kappas[block.thread_rank()] = kappas[coll_id];
				collected_quats[block.thread_rank()] = quats[coll_id];
				for (int ii = 0; ii < 16; ii++)
					collected_view2gaussian[16 * block.thread_rank() + ii] = view2gaussian[coll_id * 16 + ii];
				collected_scales[block.thread_rank()] = scales[coll_id];			
			}
			
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
				// collected_depths[block.thread_rank()] = depths[coll_id];
		}
		block.sync();

		// Iterate over Gaussians
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current Gaussian ID. Skip, if this one
			// is behind the last contributor for this pixel.
			contributor--;
			if (contributor >= last_contributor)
				continue;

			// compute ray-splat intersection as before
			float3 Tu = collected_Tu[j];
			float3 Tv = collected_Tv[j];
			float3 Tw = collected_Tw[j];
			// compute two planes intersection as the ray intersection
			float3 k = {-Tu.x + pixf.x * Tw.x, -Tu.y + pixf.x * Tw.y, -Tu.z + pixf.x * Tw.z};
			float3 l = {-Tv.x + pixf.y * Tw.x, -Tv.y + pixf.y * Tw.y, -Tv.z + pixf.y * Tw.z};
			// cross product of two planes is a line (i.e., homogeneous point), See Eq. (10)
			float3 p = crossProduct(k, l);
#if BACKFACE_CULL
			// May hanle this by replacing a low pass filter,
			// but this case is extremely rare.
			if (p.z == 0.0) continue; // there is not intersection
#endif

			float2 s = {p.x / p.z, p.y / p.z}; 
			// Compute Mahalanobis distance in the canonical splat' space
			float rho3d = (s.x * s.x + s.y * s.y); // splat distance
			
			// Add low pass filter according to Botsch et al. [2005],
			// see Eq. (11) from 2DGS paper. 
			float2 xy = collected_xy[j];
			// 2d screen distance
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); // screen distance
			float rho = min(rho3d, rho2d);
			
			// Compute accurate depth when necessary
			float c_d = (rho3d <= rho2d) ? (s.x * Tw.x + s.y * Tw.y) + Tw.z : Tw.z;
			if (c_d < NEAR_PLANE) continue;

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			glm::vec3 p_world(collected_points3d[j].x, collected_points3d[j].y, collected_points3d[j].z);
			float2 t_tange;
			float dT_dkappa = 0.0f;
			glm::vec3 dT_dnormal3D = glm::vec3(0.,0.,0.);
			float dT_dV2G[16]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
			float dT_dscale[3] = {0.,0.,0.};
			float UDF_opacity = 0;
			if (lambda > 0.0f) {
				UDF_opacity = 1 - computeOpacityGUDF(p_world,normal,collected_kappas[j],c_d,W,H,focal_x,focal_y,
					collected_view2gaussian+j*16,pixf,collected_quats[j],collected_scales[j],t_tange,&dT_dkappa,dT_dV2G,dT_dnormal3D,dT_dscale);
				// if(isnan(dT_dscale[0]) || isnan(dT_dscale[1]) || isnan(dT_dscale[2])){
				// if(collected_id[j] == 69831 && pix_id == 54640){
				// 	// printf("gaussian_id %d pix_id %d\n",collected_id[j], pix_id);
				// 	printf("pix_id %d dT_dscale: %f %f %f\n",pix_id,dT_dscale[0],dT_dscale[1],dT_dscale[2]);
				// 	printf("pixf: %f %f\n",pixf.x,pixf.y);
				// 	printf("W H: %d %d\n",W,H);
				// 	printf("focal_x focal_y: %f %f\n",focal_x,focal_y);
				// 	// print viewmat
				// 	printf("viewmat: ");
				// 	for(int i=0;i<4;i++){
				// 		for(int j=0;j<4;j++){
				// 			printf("%f ",viewmat[i*4+j]);
				// 		}
				// 		printf("\n");
				// 	}
				// 	printf("V2G: ");
				// 	float * view2gaussian = collected_view2gaussian + j * 16;
				// 	for(int m=0;m<4;m++){
				// 		for(int n=0;n<4;n++){
				// 			printf("%f ",view2gaussian[m*4+n]);
				// 		}
				// 		printf("\n");
				// 	}
				// 	printf("quat: %f %f %f %f\n",collected_quats[j].x,collected_quats[j].y,collected_quats[j].z,collected_quats[j].w);	
				// 	glm::mat3 q_rot = quat_to_rotmat(collected_quats[j]);
				// 	for(int i=0;i<3;i++){
				// 		for(int j=0;j<3;j++){
				// 			printf("%f ",q_rot[i][j]);
				// 		}
				// 		printf("\n");
				// 	}
				// 	printf("scale: %f %f %f\n",collected_scales[j].x,collected_scales[j].y,collected_scales[j].z);; 
				// 	printf("normal: %f %f %f\n",normal[0],normal[1],normal[2]);
				// 	printf("p_world: %f %f %f\n",p_world.x,p_world.y,p_world.z);
				// 	printf("kappa: %f\n",collected_kappas[j]);
				// }
			}
			float alpha = min(0.99f, nor_o.w * (lambda * UDF_opacity + (1 - lambda) * G));
			
			// const float alpha = min(0.99f, nor_o.w * G);

			if (alpha < 1.0f / 255.0f)
				continue;

			T = T / (1.f - alpha);
			const float dchannel_dcolor = alpha * T;

			// Propagate gradients to per-Gaussian colors and keep
			// gradients w.r.t. alpha (blending factor for a Gaussian/pixel
			// pair).
			float dL_dalpha = 0.0f;
			const int global_id = collected_id[j];
			for (int ch = 0; ch < C; ch++)
			{
				const float c = collected_colors[ch * BLOCK_SIZE + j];
				// Update last color (to be used in the next iteration)
				accum_rec[ch] = last_alpha * last_color[ch] + (1.f - last_alpha) * accum_rec[ch];
				last_color[ch] = c;

				const float dL_dchannel = dL_dpixel[ch];
				dL_dalpha += (c - accum_rec[ch]) * dL_dchannel;
				// Update the gradients w.r.t. color of the Gaussian. 
				// Atomic, since this pixel is just one of potentially
				// many that were affected by this Gaussian.
				atomicAdd(&(dL_dcolors[global_id * C + ch]), dchannel_dcolor * dL_dchannel);
			}

			float dL_dz = 0.0f;
			float dL_dweight = 0;
#if RENDER_AXUTILITY
			float m_d = (FAR_PLANE * c_d - FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * c_d);
			float dmd_dd = (FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				dL_dweight += dL_dmax_dweight;
			}
#if DETACH_WEIGHT 
			// if not detached weight, sometimes 
			// it will bia toward creating extragated 2D Gaussians near front
			dL_dweight += 0;
#else
			dL_dweight += (final_D2 + m_d * m_d * final_A - 2 * m_d * final_D) * dL_dreg;
#endif
			dL_dalpha += dL_dweight - last_dL_dT;
			// propagate the current weight W_{i} to next weight W_{i-1}
			last_dL_dT = dL_dweight * alpha + (1 - alpha) * last_dL_dT;
			float dL_dmd = 2.0f * (T * alpha) * (m_d * final_A - final_D) * dL_dreg;
			dL_dz += dL_dmd * dmd_dd;

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_alpha * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = c_d;
			dL_dalpha += (c_d - accum_depth_rec) * dL_ddepth;
			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;

			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				atomicAdd((&dL_dnormal3D[global_id * 3 + ch]), alpha * T * dL_dnormal2D[ch]);
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;

			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];
			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;
			float dL_dT = -nor_o.w * lambda * dL_dalpha;
			// if (dL_dT > 1e2){
			// 	printf("dL_dT is too large: %f, dL_dalpha :%f\n, nor_o.w: %f lambda: %f\n", dL_dT,dL_dalpha, nor_o.w, lambda); 
			// }
			// compute dL_dkappa dalpha_dT = -nor_o.w
			if (lambda > 0){

				atomicAdd(&dL_dkappas[global_id], dL_dT * dT_dkappa);
				// compute dL_dmean3D dalpha_dT = -nor_o.w 
				atomicAdd((&dL_dnormal3D[global_id * 3]), dL_dT * dT_dnormal3D.x);
				atomicAdd((&dL_dnormal3D[global_id * 3 + 1]), dL_dT * dT_dnormal3D.y);
				atomicAdd((&dL_dnormal3D[global_id * 3 + 2]), dL_dT * dT_dnormal3D.z);
				for (int ii = 0; ii < 16; ii++) 
				{
					atomicAdd(&(dL_dview2gaussians[global_id * 16 + ii]), dL_dT * dT_dV2G[ii]);
				}
				float a = atomicAdd(&dL_dscale[global_id][0], dL_dT * dT_dscale[0]);
				
				atomicAdd(&dL_dscale[global_id][1], dL_dT * dT_dscale[1]);
				atomicAdd(&dL_dscale[global_id][2], dL_dT * dT_dscale[2]);
				// if( (isnan(dL_dscale[global_id][0]) || isnan(dL_dscale[global_id][1]) || isnan(dL_dscale[global_id][2]) || isinf(dL_dscale[global_id][1]) || isinf(dL_dscale[global_id][2]) || isinf(dL_dscale[global_id][0])) ){ 
				// 	// printf("\n", );
				// 	printf("dT_dscale: %f %f %f dL_dT: %f alpha :%f udf_o :%f \n",dT_dscale[0], dT_dscale[1], dT_dscale[2], dL_dT,alpha,UDF_opacity);
				// 	// printf("dL_dscale: %f %f %f\n", dL_dscale[global_id][0], dL_dscale[global_id][1], dL_dscale[global_id][2]);
				// // 	printf("nor_o.w dL_dalpha: %f %f\n", nor_o.w, dL_dalpha);
				// }
			}
			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]), alpha / (nor_o.w + 1e-5) * dL_dalpha);
			// if(global_id == 181){ 
			// 	printf("dL_dopacity[global_id] is nan or inf: %f\n", dL_dopacity[global_id]);
			// 	printf("nor_o.w: %f\n", nor_o.w);
			// 	printf("alpha / (nor_o.w + 1e-5): %f\n", alpha / (nor_o.w + 1e-5));
			// 	printf("dL_dalpha: %f\n", dL_dalpha);
			// }

			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * (1 - lambda) * dL_dalpha;
			if (1 - lambda > 0){
#if RENDER_AXUTILITY
				dL_dz += alpha * T * dL_ddepth; 
#endif

				if (rho3d <= rho2d) {
					// Update gradients w.r.t. covariance of Gaussian 3x3 (T)
					float2 dL_ds = {
						dL_dG * -G * s.x + dL_dz * Tw.x,
						dL_dG * -G * s.y + dL_dz * Tw.y
					};
					float3 dz_dTw = {s.x, s.y, 1.0};
					float dsx_pz = dL_ds.x / p.z;
					float dsy_pz = dL_ds.y / p.z;
					float3 dL_dp = {dsx_pz, dsy_pz, -(dsx_pz * s.x + dsy_pz * s.y)};
					float3 dL_dk = crossProduct(l, dL_dp);
					float3 dL_dl = crossProduct(dL_dp, k);

					float3 dL_dTu = {-dL_dk.x, -dL_dk.y, -dL_dk.z};
					float3 dL_dTv = {-dL_dl.x, -dL_dl.y, -dL_dl.z};
					float3 dL_dTw = {
						pixf.x * dL_dk.x + pixf.y * dL_dl.x + dL_dz * dz_dTw.x, 
						pixf.x * dL_dk.y + pixf.y * dL_dl.y + dL_dz * dz_dTw.y, 
						pixf.x * dL_dk.z + pixf.y * dL_dl.z + dL_dz * dz_dTw.z};


					// Update gradients w.r.t. 3D covariance (3x3 matrix)
					atomicAdd(&dL_dtransMat[global_id * 9 + 0],  dL_dTu.x);
					atomicAdd(&dL_dtransMat[global_id * 9 + 1],  dL_dTu.y);
					atomicAdd(&dL_dtransMat[global_id * 9 + 2],  dL_dTu.z);
					atomicAdd(&dL_dtransMat[global_id * 9 + 3],  dL_dTv.x);
					atomicAdd(&dL_dtransMat[global_id * 9 + 4],  dL_dTv.y);
					atomicAdd(&dL_dtransMat[global_id * 9 + 5],  dL_dTv.z);
					atomicAdd(&dL_dtransMat[global_id * 9 + 6],  dL_dTw.x);
					atomicAdd(&dL_dtransMat[global_id * 9 + 7],  dL_dTw.y);
					atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dTw.z);
				} else {
					// // Update gradients w.r.t. center of Gaussian 2D mean position
					float dG_ddelx = -G * FilterInvSquare * d.x;
					float dG_ddely = -G * FilterInvSquare * d.y;
					atomicAdd(&dL_dmean2D[global_id].x, dL_dG * dG_ddelx); // not scaled
					atomicAdd(&dL_dmean2D[global_id].y, dL_dG * dG_ddely); // not scaled
					atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz); // propagate depth loss
				}
			}

			
		}
	}
}

inline __device__ void computeTransMat(
	const glm::vec3 & p_world,
	const glm::vec4 & quat,
	const glm::vec2 & scale,
	const float* viewmat,
	const float4 & intrins,
	float tan_fovx, 
	float tan_fovy,
	const float* transMat,
	const float* dL_dtransMat,
	const float* dL_dnormal3D,
	glm::vec3 & dL_dmean3D,
	glm::vec2 & dL_dscale,
	glm::vec4 & dL_drot
) {
	// camera information 
	const glm::mat3 W = glm::mat3(
		viewmat[0],viewmat[1],viewmat[2],
		viewmat[4],viewmat[5],viewmat[6],
		viewmat[8],viewmat[9],viewmat[10]
	); // viewmat 

	const glm::vec3 cam_pos = glm::vec3(viewmat[12], viewmat[13], viewmat[14]); // camera center
	const glm::mat4 P = glm::mat4(
		intrins.x, 0.0, 0.0, 0.0,
		0.0, intrins.y, 0.0, 0.0,
		intrins.z, intrins.w, 1.0, 1.0,
		0.0, 0.0, 0.0, 0.0
	);

	glm::mat3 S = scale_to_mat({scale.x, scale.y, 1.0f}, 1.0f);
	glm::mat3 R = quat_to_rotmat(quat);
	glm::mat3 RS = R * S;
	glm::vec3 p_view = W * p_world + cam_pos;
	glm::mat3 M = glm::mat3(W * RS[0], W * RS[1], p_view);


	glm::mat4x3 dL_dT = glm::mat4x3(
		dL_dtransMat[0], dL_dtransMat[1], dL_dtransMat[2],
		dL_dtransMat[3], dL_dtransMat[4], dL_dtransMat[5],
		dL_dtransMat[6], dL_dtransMat[7], dL_dtransMat[8],
		0.0, 0.0, 0.0
	);

	glm::mat3x4 dL_dM_aug = glm::transpose(P) * glm::transpose(dL_dT);
	glm::mat3 dL_dM = glm::mat3(
		glm::vec3(dL_dM_aug[0]),
		glm::vec3(dL_dM_aug[1]),
		glm::vec3(dL_dM_aug[2])
	);

	glm::mat3 W_t = glm::transpose(W);
	glm::mat3 dL_dRS = W_t * dL_dM;
	glm::vec3 dL_dRS0 = dL_dRS[0];
	glm::vec3 dL_dRS1 = dL_dRS[1];
	glm::vec3 dL_dpw = dL_dRS[2];
	glm::vec3 dL_dtn = W_t * glm::vec3(dL_dnormal3D[0], dL_dnormal3D[1], dL_dnormal3D[2]);

#if DUAL_VISIABLE
	glm::vec3 tn = W*R[2];
	float cos = glm::dot(-tn, M[2]);
	float multiplier = cos > 0 ? 1 : -1;
	dL_dtn *= multiplier;
#endif

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS0 * glm::vec3(scale.x),
		dL_dRS1 * glm::vec3(scale.y),
		dL_dtn
	);
	// glm::mat3 dL_dR = glm::mat3(
	// 	glm::vec3(0.,0.,0.),
	// 	glm::vec3(0.,0.,0.),
	// 	dL_dtn
	// );
	double n = dL_dRS0[0] * dL_dRS0[0] + dL_dRS0[1] * dL_dRS0[1] + dL_dRS0[2] * dL_dRS0[2];
	// if(n > 1e-6){
	// 	// print dL_dtransMat
	// 	for(int i = 0; i < 9; i++){
	// 		printf("%f ", dL_dtransMat[i]);
	// 	}
	// 	printf("\n");
	// }
	dL_drot = quat_to_rotmat_vjp(quat, dL_dR);
	dL_dscale = glm::vec2(
		(float)glm::dot(dL_dRS0, R[0]),
		(float)glm::dot(dL_dRS1, R[1])
	);
	// if(abs(dL_dscale[0]) > 1e-5 || abs(dL_dscale[1]) > 1e-5){
	// 	printf("scale: %f, %f\n", dL_dscale[0], dL_dscale[1]); 
	// }

	dL_dmean3D = dL_dpw;
}



template<int C>
__global__ void preprocessCUDA(
	int P, int D, int M,
	const float3* means3D,
	const float* transMats,
	const float* view2gaussian, 
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, 
	const float focal_y,
	const float tan_fovx,
	const float tan_fovy,
	const glm::vec3* campos, 
	// grad input
	const float* dL_dtransMats,
	const float* dL_dview2gaussian,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	// grad output
	glm::vec3* dL_dmean3Ds,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const float* transMat = &(transMats[9 * idx]);
	const float* dL_dtransMat = &(dL_dtransMats[9 * idx]);
	const float* dL_dnormal3D = &(dL_dnormal3Ds[3 * idx]);

	glm::vec3 p_world = glm::vec3(means3D[idx].x, means3D[idx].y, means3D[idx].z);
	float4 intrins = {focal_x, focal_y, focal_x * tan_fovx, focal_y * tan_fovy};

	glm::vec3 dL_dmean3D;
	glm::vec2 dL_dscale;
	glm::vec4 dL_drot;
	glm::vec2 scale2 = glm::vec2(scales[idx].x, scales[idx].y);
	// if (isnan(dL_dscales[idx][0]) || isnan(dL_dscales[idx][1]) || isinf(dL_dscales[idx][0]) || isinf(dL_dscales[idx][1])) printf("id: %d \n",idx);
	computeTransMat(
		p_world,
		rotations[idx],
		scale2,
		viewmatrix,
		intrins,
		tan_fovx,
		tan_fovy,
		transMat, 
		dL_dtransMat,
		dL_dnormal3D,
		dL_dmean3D, 
		dL_dscale,
		dL_drot
	);
	// update 
	dL_dmean3Ds[idx] += dL_dmean3D;
	dL_dscales[idx][0] += dL_dscale[0];
	dL_dscales[idx][1] += dL_dscale[1];

	dL_drots[idx] = dL_drot;
	computeView2Gaussian_backward(idx, means3D[idx], rotations[idx], viewmatrix, view2gaussian + 16 * idx, dL_dview2gaussian + 16 * idx, dL_dmean3Ds, dL_drots);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
}

__global__ void computeAABB(int P, 
	const int * radii,
	const float W, const float H,
	const float * transMats,
	float3 * dL_dmean2Ds,
	float *dL_dtransMats) {
	
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;
	
	const float* transMat = transMats + 9 * idx;

	const float3 dL_dmean2D = dL_dmean2Ds[idx];
	glm::mat4x3 T = glm::mat4x3(
		transMat[0], transMat[1], transMat[2],
		transMat[3], transMat[4], transMat[5],
		transMat[6], transMat[7], transMat[8],
		transMat[6], transMat[7], transMat[8]
	);

	float d = glm::dot(glm::vec3(1.0, 1.0, -1.0), T[3] * T[3]);
	glm::vec3 f = glm::vec3(1.0, 1.0, -1.0) * (1.0f / d);

	glm::vec3 p = glm::vec3(
		glm::dot(f, T[0] * T[3]),
		glm::dot(f, T[1] * T[3]), 
		glm::dot(f, T[2] * T[3]));

	glm::vec3 dL_dT0 = dL_dmean2D.x * f * T[3];
	glm::vec3 dL_dT1 = dL_dmean2D.y * f * T[3];
	glm::vec3 dL_dT3 = dL_dmean2D.x * f * T[0] + dL_dmean2D.y * f * T[1];
	glm::vec3 dL_df = (dL_dmean2D.x * T[0] * T[3]) + (dL_dmean2D.y * T[1] * T[3]);
	float dL_dd = glm::dot(dL_df, f) * (-1.0 / d);
	glm::vec3 dd_dT3 = glm::vec3(1.0, 1.0, -1.0) * T[3] * 2.0f;
	dL_dT3 += dL_dd * dd_dT3;
	dL_dtransMats[9 * idx + 0] += dL_dT0.x;
	dL_dtransMats[9 * idx + 1] += dL_dT0.y;
	dL_dtransMats[9 * idx + 2] += dL_dT0.z;
	dL_dtransMats[9 * idx + 3] += dL_dT1.x;
	dL_dtransMats[9 * idx + 4] += dL_dT1.y;
	dL_dtransMats[9 * idx + 5] += dL_dT1.z;
	dL_dtransMats[9 * idx + 6] += dL_dT3.x;
	dL_dtransMats[9 * idx + 7] += dL_dT3.y;
	dL_dtransMats[9 * idx + 8] += dL_dT3.z;

	// just use to hack the projected 2D gradient here.
	float z = transMat[8];
	dL_dmean2Ds[idx].x = dL_dtransMats[9 * idx + 2] * z * W; // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[9 * idx + 5] * z * H; // to ndc
}


__global__ void computeMean2D_backward(int P, const float3* means3D, 
	const float* viewmatrix,
	const float focal_x, const float focal_y,
	const glm::vec3* dL_dmean3Ds, 
	float3* dL_dmean2Ds) {

	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;
	glm::mat3 R = glm::mat3(viewmatrix[0],viewmatrix[1],viewmatrix[2],
		viewmatrix[4],viewmatrix[5],viewmatrix[6],
		viewmatrix[8],viewmatrix[9],viewmatrix[10]);
	glm::vec3 t = glm::vec3(viewmatrix[12], viewmatrix[13], viewmatrix[14]);
	glm::vec3 p_world = glm::vec3(means3D[idx].x, means3D[idx].y, means3D[idx].z);
	glm::vec3 p_view = R * p_world + t;
	float depth = p_view[2];
	glm::vec3 dL_dmean3Ds_V =  glm::transpose(R) * dL_dmean3Ds[idx];
	dL_dmean2Ds[idx].x = dL_dmean3Ds_V.x * depth / focal_x; 
	dL_dmean2Ds[idx].y = dL_dmean3Ds_V.y * depth / focal_x; 
	// TODO: use metric proposed in GOF
	dL_dmean2Ds[idx].z = 0;
}
void BACKWARD::preprocess(
	int P, int D, int M,
	const float3* means3D,
	const int* radii,
	const float* shs,
	const bool* clamped,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float scale_modifier,
	const float* transMats,
	const float* view2gaussian,
	const float* viewmatrix,
	const float* projmatrix,
	const float focal_x, const float focal_y,
	const float tan_fovx, const float tan_fovy,
	const glm::vec3* campos, 
	float3* dL_dmean2Ds,
	const float* dL_dnormal3Ds,
	float* dL_dtransMats,
	float* dL_dview2gaussian,

	float* dL_dcolors,
	float* dL_dshs,
	glm::vec3* dL_dmean3Ds,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	// Propagate gradients for the path of 2D conic matrix computation. 
	// Somewhat long, thus it is its own kernel rather than being part of 
	// "preprocess". When done, loss gradient w.r.t. 3D means has been
	// modified and gradient w.r.t. 3D covariance matrix has been computed.	
	// propagate gradients to transMat

	// we do not use the center actually
	float W = focal_x * tan_fovx;
	float H = focal_y * tan_fovy;
	computeAABB << <(P + 255) / 256, 256 >> >(
		P,
		radii,
		W, H,
		transMats,
		dL_dmean2Ds,
		dL_dtransMats);
	
	// propagate gradients from transMat to mean3d, scale, rot, sh, color
	preprocessCUDA<NUM_CHANNELS><< <(P + 255) / 256, 256 >> > (
		P, D, M,
		(float3*)means3D,
		transMats,
		view2gaussian,
		radii,
		shs,
		clamped,
		(glm::vec3*)scales,
		(glm::vec4*)rotations,
		scale_modifier,
		viewmatrix,
		projmatrix,
		focal_x, 
		focal_y,
		tan_fovx,
		tan_fovy,
		campos,	
		dL_dtransMats,
		dL_dview2gaussian,
		dL_dnormal3Ds,
		dL_dcolors,
		dL_dshs,
		dL_dmean3Ds,
		dL_dscales,
		dL_drots
	);

	// computeMean2D_backward<<<(P + 255) / 256, 256>>>(P, means3D, viewmatrix, 
	// 	focal_x, focal_y, dL_dmean3Ds, dL_dmean2Ds);


}

void BACKWARD::render(
	const dim3 grid, const dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* bg_color,
	const float3* means3D,
	const glm::vec3* scales,
	const glm::vec4* rotations,
	const float2* means2D,
	const float* kappas,
	const float* viewmatrix,
	const float4* normal_opacity,
	const float* colors,
	const float* transMats,
	const float* view2gaussians,
	const float* depths,
	const float* final_Ts,
	const uint32_t* n_contrib,
	const float lambda,
	const float* dL_dpixels,
	const float* dL_depths,
	float * dL_dtransMat,
	float * dL_dview2gaussian,
	glm::vec3* dL_dscale,
	float3* dL_dmean2D,
	float* dL_dnormal3D,
	float* dL_dopacity,
	float* dL_dcolors,
	float* dL_dkappas
)
{
	renderCUDA<NUM_CHANNELS> << <grid, block >> >(
		ranges,
		point_list,
		W, H,
		focal_x, focal_y,
		bg_color,
		means2D,
		normal_opacity,
		means3D,
		kappas,
		rotations,
		scales,
		viewmatrix,
		transMats,
		view2gaussians,
		colors,
		depths,
		final_Ts,
		n_contrib,
		lambda,
		dL_dpixels,
		dL_depths,
		dL_dtransMat,
		dL_dview2gaussian,
		dL_dscale,
		dL_dmean2D,
		dL_dnormal3D,
		dL_dopacity,
		dL_dcolors,
		dL_dkappas
		);
	
}
