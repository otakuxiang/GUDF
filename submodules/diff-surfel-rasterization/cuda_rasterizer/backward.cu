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
__device__ inline float log1_ex(float x){
	float a= 1+expf(x),res=logf(a);
	if (isinf(a)){
		res=x;
	}
	return res;
}
// backward pass of GUDF opacity
__device__ float computeOpacityGUDF(const glm::vec3 &p_world, const float *normal, const float kappa, 
	const int W, const int H,float focal_x, float focal_y, const float *view2gaussian, const float2 &pixf, 
 	const glm::vec3 &scale, float* int_depth, float* dT_dkappa, float *dT_dV2G,glm::vec3 &dT_dnormal3D,
	float *dT_dscale, float *dD_dkappa, float *dD_dV2G, glm::vec3 &dD_dnormal3D, float *dD_dscale
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
    glm::vec3 dcosdn = ray_gauss;
	glm::vec3 dcosdr = normal_g;
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

	// if (tn > depth_o || tf < depth_o){
	// 	return 1.0f;
 	// }
	// float mid_d = (tf - tn) / 2.0f;
	// float mid_d = sqrt(expf(2*(logf(-B)-logf(A))) - 4 * expf(logf(C) - logf(A))) / 2;


	

	// tn = depth_o - mid_d;
	// tf = depth_o + mid_d;
	// int_depth[0] = depth_o;

	float flagtf = depth_o < tf ? -1 : 1, flagtn = depth_o > tn ? 1 : -1;
	glm::vec3 p_tn = ray_o_scaled + ray_d_scaled * tn;
	glm::vec3 p_tf = ray_o_scaled + ray_d_scaled * tf;
	
	float tnogrg = glm::dot(p_tn,ray_d_scaled);
	float tfogrg = glm::dot(p_tf,ray_d_scaled);



	float ftn = cos_theta * (depth_o - tn);
	float ftf = cos_theta * (depth_o - tf);

	const float kappacos = kappa * cos_theta;
	float lnE = -kappa * ftf,lnF = -kappa * ftn;
	float lnB = log1_ex(lnE), lnC = log1_ex(lnF),lnT = lnC - lnB;
	float lncos = logf(cos_theta),lnkappa = logf(kappa),lntfd = logf(abs(tf - depth_o)),lntnd = logf(abs(depth_o - tn));
	float lnkc = lncos + lnkappa, lntftn = logf(tf - tn);
	float lndTdB = lnC - 2 * lnB, lndTdC = - lnB;
	float T_1 = expf(lnT) - 1,int_T;
	//compute int_T
	if (lnC > 19){
		int_T = -T_1 / kappacos;
	}
	else{
		int_T = expf(lnC) / kappacos * (kappacos * (tf - tn) + lnT);
	}
	int_depth[0] = -(T_1 + 1) * tf + tn + int_T;
	// compute dDdtf
	
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
	dT_dkappa[0] = flagtf * expf(lndTdB + lnE + lncos + lntfd) 
		- flagtn * expf(lndTdC + lnF + lncos + lntnd);
	// float dBddepth = kappacos * E;
	// float dCddepth = kappacos * F;
	// float dT_ddepth = dT_dC * dCddepth + dT_dB * dBddepth;
	// float lndB_ddepth = lnkc + lnE,lndCddepth = lnkc + lnF;
	float dT_ddepth = - expf(lndTdC + lnkc + lnF) + expf(lndTdB + lnkc + lnE);

	// dAdtn = kappa * cos * A;  dCdtn = - F * kappa * cos
	// float dTdtn = dT_dA * kappacos * A - dT_dC * kappacos * F;
	// dAdtf = - kappa * cos * A; dBdtf = - E * kappa * cos
	// float dTdtf = - dT_dA * kappacos * A - dT_dB * kappacos * E;
	float dTdtn = expf(lndTdC + lnkc + lnF);
	float dTdtf = - expf(lndTdB + lnkc + lnE);
	// float dTdtn = - dTdtf1 / 2 + dTdtn1 / 2;
	// float dTdtf = dTdtf1 / 2 - dTdtn1 / 2;
	// dT_ddepth += dTdtn1 + dTdtf1;
	// float dT_ddepth = 0.0f;
	// float T = expf(-kappa * cos_theta * mid_d);
	// dT_dkappa[0] = -cos_theta * mid_d * T;
	// float dTdcos = - kappa * mid_d * T; 
	// float dTdtn = 1 / 2 * kappacos * T;
	// float dTdtf = - 1 / 2 * kappacos * T;

	// dAdcos = (tn - tf) * A * kappa, dBdcos = E * kappa * (depth - tf), dCdcos = F * kappa * (depth - tn)
	// float dAdcos = (tn - tf) * A * kappa;
	// float dBdcos = E * kappa * (depth_o - tf);
	// float dCdcos = F * kappa * (depth_o - tn);
	// float dTdcos = dT_dA * dAdcos + dT_dB * dBdcos + dT_dC * dCdcos;
	float dTdcos = flagtf * expf(lndTdB + lntfd + lnkappa + lnE) - flagtn * expf(lndTdC + lntnd + lnkappa + lnF); 
	
	// backprop int_depth
	float dEdtn,dEdkappa,dEddepth,dEdcos;
	
	if(lnC > 19){
		dEdtn = - 1 / kappacos * dTdtn;
		dEdkappa = (T_1 / kappa - dT_dkappa[0]) / kappacos;
		dEddepth = -dT_ddepth / kappacos;
		dEdcos = (T_1 / cos_theta - dTdcos) / kappacos;
	}
	else{
		float G = kappacos * (tf - tn),C = expf(lnC),F = expf(lnC - lnkc);
		dEdtn = (G + lnT) * expf(lnF) - 1;
	
		// float ftn = cos_theta * (depth_o - tn);
		// float dCdkappa = -ftn * expf(lnF),dBdkappa = -ftf * expf(lnE);
		float dHdkappa = -ftn * expf(lnF - lnC) + flagtf * expf(lnE + lncos + lntfd - lnB);
		float dFdkappa = - flagtn * expf(lnF + lntnd - lnkappa) - expf(lnC - lnkc - lnkappa);
		dEdkappa = (G + lnT) * dFdkappa + F * (dHdkappa + cos_theta * (tf - tn));
		float dHddepth = -expf(lnkc + lnF - lnC) + expf(lnkc + lnE - lnB);
		dEddepth = -(G + lnT) * expf(lnF) + F * dHddepth;
		float dHdcos = -flagtn * expf(lntnd + lnkappa + lnF - lnC) + flagtf * expf(lntfd + lnkappa + lnE - lnB);
		float dFdcos = -flagtn * expf(lntnd - lncos + lnF) - expf(lnC - lnkc - lncos);
		dEdcos = (G + lnT) * dFdcos + F * (dHdcos + kappa * (tf - tn));
		// if (isinf(dEdcos) || isnan(dEdcos)){
		// 	printf("G: %f, lnT");
		// }
		
	}
	float dDdtf = -dTdtf * tf, dDdtn = -tf * dTdtn + dEdtn + 1;
	dD_dkappa[0] = dEdkappa - tf * dT_dkappa[0];
	float dDddepth = dEddepth - tf * dT_ddepth, dDdcos = dEdcos - tf * dTdcos;
	
	
	glm::vec3 dTdno = dT_ddepth * ddepthdn + dTdcos * dcosdn;
	dT_dnormal3D[0] = dTdno[0] * view2gaussian[0] + dTdno[1] * view2gaussian[1] + dTdno[2] * view2gaussian[2];
	dT_dnormal3D[1] = dTdno[0] * view2gaussian[4] + dTdno[1] * view2gaussian[5] + dTdno[2] * view2gaussian[6];
	dT_dnormal3D[2] = dTdno[0] * view2gaussian[8] + dTdno[1] * view2gaussian[9] + dTdno[2] * view2gaussian[10];

	glm::vec3 dDdno = dDddepth * ddepthdn + dDdcos * dcosdn;
	dD_dnormal3D[0] = dDdno[0] * view2gaussian[0] + dDdno[1] * view2gaussian[1] + dDdno[2] * view2gaussian[2];


	dD_dnormal3D[1] = dDdno[0] * view2gaussian[4] + dDdno[1] * view2gaussian[5] + dDdno[2] * view2gaussian[6];
	dD_dnormal3D[2] = dDdno[0] * view2gaussian[8] + dDdno[1] * view2gaussian[9] + dDdno[2] * view2gaussian[10];

	// dtndrg = - tn / tnogrg * p_tn; dtfdrg = - tf / tfogrg * p_tf;
	glm::vec3 dTdrg = - dTdtn * tn / tnogrg * p_tn - dTdtf * tf / tfogrg * p_tf;
	// dtndog = - p_tn / tnogrg; dtfdog = - p_tf / tfogrg;
	glm::vec3 dTdog = - dTdtn * p_tn / tnogrg - dTdtf * p_tf / tfogrg;

	glm::vec3 dDdrg = - dDdtn * tn / tnogrg * p_tn - dDdtf * tf / tfogrg * p_tf; 
	glm::vec3 dDdog = - dDdtn * p_tn / tnogrg - dDdtf * p_tf / tfogrg;

	glm::vec3 dTdo = glm::vec3(dTdog[0] / 3 / scaled[0],dTdog[1] / 3 / scaled[1],dTdog[2] / 3 / scaled[2]);
	dTdo = dTdo + dT_ddepth * ddepthdo;
	glm::vec3 dTdr = glm::vec3(dTdrg[0] / 3 / scaled[0],dTdrg[1] / 3 / scaled[1],dTdrg[2] / 3 / scaled[2]);
	dTdr = dTdr + dT_ddepth * ddepthdr + dTdcos * dcosdr; 

	glm::vec3 dDdo = glm::vec3(dDdog[0] / 3 / scaled[0],dDdog[1] / 3 / scaled[1],dDdog[2] / 3 / scaled[2]);
	dDdo = dDdo + dDddepth * ddepthdo;
	glm::vec3 dDdr = glm::vec3(dDdrg[0] / 3 / scaled[0],dDdrg[1] / 3 / scaled[1],dDdrg[2] / 3 / scaled[2]);
	dDdr = dDdr + dDddepth * ddepthdr + dDdcos * dcosdr; 

	dT_dV2G[0] = dTdr[0] * ray_view[0] + dTdno[0] * normal[0];
	dT_dV2G[1] = dTdr[1] * ray_view[0] + dTdno[1] * normal[0];
	dT_dV2G[2] = dTdr[2] * ray_view[0] + dTdno[2] * normal[0];
	dT_dV2G[4] = dTdr[0] * ray_view[1] + dTdno[0] * normal[1]; 
	dT_dV2G[5] = dTdr[1] * ray_view[1] + dTdno[1] * normal[1];
	dT_dV2G[6] = dTdr[2] * ray_view[1] + dTdno[2] * normal[1];
	dT_dV2G[8] = dTdr[0] * ray_view[2] + dTdno[0] * normal[2];
	dT_dV2G[9] = dTdr[1] * ray_view[2] + dTdno[1] * normal[2];
	dT_dV2G[10] = dTdr[2] * ray_view[2] + dTdno[2] * normal[2];
	dT_dV2G[12] = dTdo[0];
	dT_dV2G[13] = dTdo[1]; 
	dT_dV2G[14] = dTdo[2];

	dD_dV2G[0] = dDdr[0] * ray_view[0] + dDdno[0] * normal[0];
	dD_dV2G[1] = dDdr[1] * ray_view[0] + dDdno[1] * normal[0];
	dD_dV2G[2] = dDdr[2] * ray_view[0] + dDdno[2] * normal[0];
	dD_dV2G[4] = dDdr[0] * ray_view[1] + dDdno[0] * normal[1]; 
	dD_dV2G[5] = dDdr[1] * ray_view[1] + dDdno[1] * normal[1];
	dD_dV2G[6] = dDdr[2] * ray_view[1] + dDdno[2] * normal[1];
	dD_dV2G[8] = dDdr[0] * ray_view[2] + dDdno[0] * normal[2];
	dD_dV2G[9] = dDdr[1] * ray_view[2] + dDdno[1] * normal[2];
	dD_dV2G[10] = dDdr[2] * ray_view[2] + dDdno[2] * normal[2];
	dD_dV2G[12] = dDdo[0];
	dD_dV2G[13] = dDdo[1]; 
	dD_dV2G[14] = dDdo[2];

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

	dD_dscale[0] = dDdrg[0] * drg_dscale[0] + dDdog[0] * dog_dscale[0];
	dD_dscale[1] = dDdrg[1] * drg_dscale[1] + dDdog[1] * dog_dscale[1];
	dD_dscale[2] = dDdrg[2] * drg_dscale[2] + dDdog[2] * dog_dscale[2];
	return expf(lnT);
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
	glm::vec4 dL_dq = quat_to_rotmat_vjp(rot,dL_dMt);;
	// dL_dq.x = 2 * z * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * y * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * x * (dL_dMt[1][2] - dL_dMt[2][1]);
	// dL_dq.y = 2 * y * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * z * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * r * (dL_dMt[1][2] - dL_dMt[2][1]) - 4 * x * (dL_dMt[2][2] + dL_dMt[1][1]);
	// dL_dq.z = 2 * x * (dL_dMt[1][0] + dL_dMt[0][1]) + 2 * r * (dL_dMt[2][0] - dL_dMt[0][2]) + 2 * z * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * y * (dL_dMt[2][2] + dL_dMt[0][0]);
	// dL_dq.w = 2 * r * (dL_dMt[0][1] - dL_dMt[1][0]) + 2 * x * (dL_dMt[2][0] + dL_dMt[0][2]) + 2 * y * (dL_dMt[1][2] + dL_dMt[2][1]) - 4 * z * (dL_dMt[1][1] + dL_dMt[0][0]);

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
renderCUDA_2DGS(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float* __restrict__ transMats,
	const float* __restrict__ colors,
	const float* __restrict__ depths,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float3* __restrict__ dL_dmean2D,
	float* __restrict__ dL_dnormal3D,
	float* __restrict__ dL_dopacity,
	float* __restrict__ dL_dcolors)
{
	// We rasterize again. Compute necessary block info.
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	const uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	const uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x , (float)pix.y };

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
		// dL_dmax_dweight = dL_depths[MEDIAN_WEIGHT_OFFSET * H * W + pix_id];
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
			const float2 xy = collected_xy[j];
			const float3 Tu = collected_Tu[j];
			const float3 Tv = collected_Tv[j];
			const float3 Tw = collected_Tw[j];
			// compute two planes intersection as the ray intersection
			float3 k = pix.x * Tw - Tu;
			float3 l = pix.y * Tw - Tv;
			float3 p = cross(k, l);
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
			// 2d screen distance
			float2 d = {xy.x - pixf.x, xy.y - pixf.y};
			float rho2d = FilterInvSquare * (d.x * d.x + d.y * d.y); // screen distance
			float rho = min(rho3d, rho2d);
			
			// Compute accurate depth when necessary
			float c_d = (s.x * Tw.x + s.y * Tw.y) + Tw.z;
			if (c_d < NEAR_PLANE) continue;

			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};

			float power = -0.5f * rho;
			if (power > 0.0f)
				continue;

			const float G = exp(power);
			const float alpha = min(0.99f, nor_o.w * G);
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
			float m_d =  FAR_PLANE / (FAR_PLANE - NEAR_PLANE) * (1 - NEAR_PLANE / c_d);  
			float dmd_dd = (FAR_PLANE * NEAR_PLANE) / ((FAR_PLANE - NEAR_PLANE) * c_d * c_d);
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				// dL_dweight += dL_dmax_dweight;
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


			// Helpful reusable temporary variables
			const float dL_dG = nor_o.w * dL_dalpha;
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
				atomicAdd(&dL_dtransMat[global_id * 9 + 6],  s.x * dL_dz);
				atomicAdd(&dL_dtransMat[global_id * 9 + 7],  s.y * dL_dz);
				atomicAdd(&dL_dtransMat[global_id * 9 + 8],  dL_dz);
			}

			// Update gradients w.r.t. opacity of the Gaussian
			atomicAdd(&(dL_dopacity[global_id]),  G * dL_dalpha);
		}
	}
}
// Backward version of the rendering procedure.
template <uint32_t C>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_GUDF(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	float focal_x, float focal_y,
	const float* __restrict__ bg_color,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ normal_opacity,
	const float3* __restrict__ points3d,
	const float* __restrict__ kappas,
	const glm::vec3* __restrict__ scales,
	const float* __restrict__ view2gaussian,
	const float* __restrict__ colors,
	const float* __restrict__ final_Ts,
	const uint32_t* __restrict__ n_contrib,
	const float* __restrict__ dL_dpixels,
	const float* __restrict__ dL_depths,
	float * __restrict__ dL_dtransMat,
	float * __restrict__ dL_dview2gaussians,
	glm::vec3* __restrict__ dL_dscale,
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
	const float2 pixf = { (float)pix.x , (float)pix.y};

	const bool inside = pix.x < W&& pix.y < H;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];

	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);

	bool done = !inside;
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_normal_opacity[BLOCK_SIZE];
	__shared__ float collected_colors[C * BLOCK_SIZE];
	__shared__ float3 collected_points3d[BLOCK_SIZE];
	__shared__ float collected_kappas[BLOCK_SIZE];
	__shared__ glm::vec3 collected_scales[BLOCK_SIZE];
	__shared__ float collected_view2gaussian[BLOCK_SIZE * 16];

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
	float last_dL_dT = 0;
#endif

	if (inside){
		for (int i = 0; i < C; i++)
			dL_dpixel[i] = dL_dpixels[i * H * W + pix_id];
	}

	float last_alpha = 0;
	float last_opacity = 0;
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
			collected_points3d[block.thread_rank()] = points3d[coll_id];
			collected_kappas[block.thread_rank()] = kappas[coll_id];
			for (int ii = 0; ii < 16; ii++)
				collected_view2gaussian[16 * block.thread_rank() + ii] = view2gaussian[coll_id * 16 + ii];
			collected_scales[block.thread_rank()] = scales[coll_id];			
			for (int i = 0; i < C; i++)
				collected_colors[i * BLOCK_SIZE + block.thread_rank()] = colors[coll_id * C + i];
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


			float4 nor_o = collected_normal_opacity[j];
			float normal[3] = {nor_o.x, nor_o.y, nor_o.z};

			glm::vec3 p_world(collected_points3d[j].x, collected_points3d[j].y, collected_points3d[j].z);
			float2 t_tange;
			float dT_dkappa = 0.0f,dD_dkappa = 0.0f;
			glm::vec3 dT_dnormal3D = glm::vec3(0.,0.,0.),dD_dnormal3D = glm::vec3(0.,0.,0.);
			float dT_dV2G[16]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
			float dT_dscale[3] = {0.,0.,0.};
			float dD_dscale[3] = {0.,0.,0.};
			float dD_dV2G[16]={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
			float UDF_opacity = 0, depth = 0;

			UDF_opacity = 1 - computeOpacityGUDF(p_world,normal,collected_kappas[j],W,H,focal_x,focal_y,
				collected_view2gaussian+j*16,pixf,collected_scales[j],
				&depth,&dT_dkappa,dT_dV2G,dT_dnormal3D,dT_dscale,
				&dD_dkappa,dD_dV2G,dD_dnormal3D,dD_dscale 
				);
			if (UDF_opacity < 1e-6f) continue; 
			if (depth < 0.1) continue; 
			// if(isnan(dT_dscale[0]) || isnan(dT_dscale[1]) || isnan(dT_dscale[2]) 
			// 	|| isinf(dT_dscale[0] || isinf(dT_dscale[1]) || isinf(dT_dscale[2]))){
			// if(isinf(dT_dkappa) || isnan(dT_dkappa)){
			
			// // if (abs(dD_dscale[2] > 1000000)){

			
			// // if(collected_id[j] == 17133){
			// // 	// printf("gaussian_id %d pix_id %d\n",collected_id[j], pix_id);
			// 	float * view2gaussian = collected_view2gaussian + j * 16;
			// // 	// printf("%f\n%f %f\n%d %d\n%f %f\n%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f\n%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f\n%f %f %f\n",
			// // 	// 	depth,
			// // 	// 	pixf.x,pixf.y,
			// // 	// 	W,H,
			// // 	// 	focal_x,focal_y,
			// // 	// 	view2gaussian[0],view2gaussian[1],view2gaussian[2],view2gaussian[3],
			// // 	// 	view2gaussian[4],view2gaussian[5],view2gaussian[6],view2gaussian[7],
			// // 	// 	view2gaussian[8],view2gaussian[9],view2gaussian[10],view2gaussian[11],
			// // 	// 	view2gaussian[12],view2gaussian[13],view2gaussian[14],view2gaussian[15],
			// // 	// 	collected_scales[j].x,collected_scales[j].y,collected_scales[j].z,
			// // 	// 	normal[0],normal[1],normal[2],
			// // 	// 	p_world.x,p_world.y,p_world.z,
			// // 	// 	collected_kappas[j],
			// // 	// 	dD_dV2G[0],dD_dV2G[1],dD_dV2G[2],dD_dV2G[3],
			// // 	// 	dD_dV2G[4],dD_dV2G[5],dD_dV2G[6],dD_dV2G[7],
			// // 	// 	dD_dV2G[8],dD_dV2G[9],dD_dV2G[10],dD_dV2G[11],
			// // 	// 	dD_dV2G[12],dD_dV2G[13],dD_dV2G[14],dD_dV2G[15],
			// // 	// 	dD_dnormal3D[0],dD_dnormal3D[1],dD_dnormal3D[2],
			// // 	// 	dD_dscale[0],dD_dscale[1],dD_dscale[2]
			// // 	// );
				// printf("%f %f\n%d %d\n%f %f\n%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n%f %f %f\n%f %f %f\n%f %f %f\n%f\n",
				// 	// depth,
				// 	pixf.x,pixf.y,
				// 	W,H,
				// 	focal_x,focal_y,
				// 	view2gaussian[0],view2gaussian[1],view2gaussian[2],view2gaussian[3],
				// 	view2gaussian[4],view2gaussian[5],view2gaussian[6],view2gaussian[7],
				// 	view2gaussian[8],view2gaussian[9],view2gaussian[10],view2gaussian[11],
				// 	view2gaussian[12],view2gaussian[13],view2gaussian[14],view2gaussian[15],
				// 	collected_scales[j].x,collected_scales[j].y,collected_scales[j].z,
				// 	normal[0],normal[1],normal[2],
				// 	p_world.x,p_world.y,p_world.z,
				// 	collected_kappas[j]
				// );
			// // 	// printf("pix_id %d dT_dscale: %f %f %f\n",pix_id,dT_dscale[0],dT_dscale[1],dT_dscale[2]);
			// // 	// printf("pixf: %f %f\n",pixf.x,pixf.y);
			// // 	// printf("W H: %d %d\n",W,H);
			// // 	// printf("focal_x focal_y: %f %f\n",focal_x,focal_y);
			// // 	// // print viewmat

			// // 	// printf("V2G: ");
			// // 	// for(int m=0;m<4;m++){
			// // 	// 	for(int n=0;n<4;n++){
			// // 	// 		printf("%f ",view2gaussian[m*4+n]);
			// // 	// 	}
			// // 	// 	printf("\n");
			// // 	// }
			// // 	// printf("scale: %f %f %f\n",collected_scales[j].x,collected_scales[j].y,collected_scales[j].z);; 
			// // 	// printf("normal: %f %f %f\n",normal[0],normal[1],normal[2]);
			// // 	// printf("p_world: %f %f %f\n",p_world.x,p_world.y,p_world.z);
			// // 	// printf("kappa: %f\n",collected_kappas[j]);
			// }
			
			float alpha = min(0.99f, nor_o.w * UDF_opacity );
			// float max_alpha = collected_malphas[j];
			// alpha = alpha / max_alpha;
		
			// const float alpha = min(0.99f, UDF_opacity);

			
			if (alpha < 1.0f / 255.0f)
				continue;

			float tmp_T = T;
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

			float dL_do = 0.0f;
			float dL_dz = 0.0f;
			float dL_dweight = 0;
			float dL_dT = 0.f;
			float depth_u = depth / UDF_opacity;
#if RENDER_AXUTILITY

			
			if (contributor == median_contributor-1) {
				dL_dz += dL_dmedian_depth;
				dL_dweight += dL_dmax_dweight;
			}

			// Propagate gradients w.r.t ray-splat depths
			accum_depth_rec = last_opacity * last_depth + (1.f - last_alpha) * accum_depth_rec;
			last_depth = depth;
			dL_dalpha += - accum_depth_rec * dL_ddepth;
			// atomicAdd(&dL_dopacity[global_id], depth * T * dL_ddepth);
			dL_do += depth * T * dL_ddepth;

			// Propagate gradients w.r.t. color ray-splat alphas
			accum_alpha_rec = last_alpha * 1.0 + (1.f - last_alpha) * accum_alpha_rec;
			dL_dalpha += (1 - accum_alpha_rec) * dL_daccum;
			float dL_dnormal[3] = {0.,0.,0.};
			// Propagate gradients to per-Gaussian normals
			for (int ch = 0; ch < 3; ch++) {
				accum_normal_rec[ch] = last_alpha * last_normal[ch] + (1.f - last_alpha) * accum_normal_rec[ch];
				last_normal[ch] = normal[ch];
				dL_dalpha += (normal[ch] - accum_normal_rec[ch]) * dL_dnormal2D[ch];
				dL_dnormal[ch] += alpha * T * dL_dnormal2D[ch];
			}
#endif

			dL_dalpha *= T;
			// Update last alpha (to be used in the next iteration)
			last_alpha = alpha;
			last_opacity = nor_o.w;
			// Account for fact that alpha also influences how much of
			// the background color is added if nothing left to blend
			float bg_dot_dpixel = 0;
			for (int i = 0; i < C; i++)
				bg_dot_dpixel += bg_color[i] * dL_dpixel[i];

			dL_dalpha += (-T_final / (1.f - alpha)) * bg_dot_dpixel;

			dL_dT += -nor_o.w * dL_dalpha;
			// float dL_dT = -nor_o.w * dL_dalpha / max_alpha;
			
			// float dL_dmax_alpha = - dL_dalpha * alpha / max_alpha;
			// float ks = expf(-collected_kappas[j] * collected_scales[j].z);
			// atomicAdd(&dL_dkappas[global_id], dL_dmax_alpha * collected_scales[j].z * ks);
			// atomicAdd(&dL_dscale[global_id][2], dL_dmax_alpha * collected_kappas[j] * ks); 
			// float dL_dT = -dL_dalpha;

			// if (dL_dT > 1e2){
			// 	printf("dL_dT is too large: %f, dL_dalpha :%f\n, nor_o.w: %f lambda: %f\n", dL_dT,dL_dalpha, nor_o.w, lambda); 
			// }
			// compute dL_dkappa dalpha_dT = -nor_o.w
			
			

#if RENDER_AXUTILITY
			dL_dz += nor_o.w * T * dL_ddepth; 
#endif	
			// if(isnan(dL_dT) || isinf(dL_dT) || isnan(dL_dz) || isinf(dL_dz)){
			// 	printf("dL_dT or dL_dz is nan or inf\n");
			// }		
			atomicAdd(&dL_dscale[global_id][0], dL_dT * dT_dscale[0] + dL_dz * dD_dscale[0]);
			atomicAdd(&dL_dscale[global_id][1], dL_dT * dT_dscale[1] + dL_dz * dD_dscale[1]);
			atomicAdd(&dL_dscale[global_id][2], dL_dT * dT_dscale[2] + dL_dz * dD_dscale[2]);
			atomicAdd(&dL_dkappas[global_id], dL_dT * dT_dkappa + dL_dz * dD_dkappa);
			// compute dL_dmean3D dalpha_dT = -nor_o.w 
			dL_dnormal[0] += dL_dT * dT_dnormal3D.x + dL_dz * dD_dnormal3D.x;
			dL_dnormal[1] += dL_dT * dT_dnormal3D.y + dL_dz * dD_dnormal3D.y;
			dL_dnormal[2] += dL_dT * dT_dnormal3D.z + dL_dz * dD_dnormal3D.z;
			atomicAdd((&dL_dnormal3D[global_id * 3]), dL_dnormal[0]);
			atomicAdd((&dL_dnormal3D[global_id * 3 + 1]), dL_dnormal[1]);
			atomicAdd((&dL_dnormal3D[global_id * 3 + 2]), dL_dnormal[2]);
			for (int ii = 0; ii < 16; ii++) 
			{	
				atomicAdd(&(dL_dview2gaussians[global_id * 16 + ii]), dL_dT * dT_dV2G[ii]
					+dL_dz * dD_dV2G[ii]);
			}

			// Update gradients w.r.t. opacity of the Gaussian
			dL_do += alpha / (nor_o.w + 1e-5) * dL_dalpha;
			atomicAdd(&(dL_dopacity[global_id]), dL_do);

		}
	}
}

__device__ void compute_transmat_aabb(
	int idx, 
	const float* Ts_precomp,
	const float3* p_origs, 
	const glm::vec3* scales, 
	const glm::vec4* rots, 
	const float* projmatrix, 
	const float* viewmatrix, 
	const int W, const int H, 
	const float3* dL_dnormals,
	const float3* dL_dmean2Ds, 
	float* dL_dTs, 
	glm::vec3* dL_dmeans, 
	glm::vec3* dL_dscales,
	 glm::vec4* dL_drots)
{
	glm::mat3 T;
	float3 normal;
	glm::mat3x4 P;
	glm::mat3 R;
	glm::mat3 S;
	float3 p_orig;
	glm::vec4 rot;
	glm::vec2 scale;
	
	// Get transformation matrix of the Gaussian
	if (Ts_precomp != nullptr) {
		T = glm::mat3(
			Ts_precomp[idx * 9 + 0], Ts_precomp[idx * 9 + 1], Ts_precomp[idx * 9 + 2],
			Ts_precomp[idx * 9 + 3], Ts_precomp[idx * 9 + 4], Ts_precomp[idx * 9 + 5],
			Ts_precomp[idx * 9 + 6], Ts_precomp[idx * 9 + 7], Ts_precomp[idx * 9 + 8]
		);
		normal = {0.0, 0.0, 0.0};
	} else {
		p_orig = p_origs[idx];
		rot = rots[idx];
		scale.x = scales[idx].x;
		scale.y = scales[idx].y;
		R = quat_to_rotmat(rot);
		S = scale_to_mat(scales[idx], 1.0f);
		S[2][2] = 1.0f;
		glm::mat3 L = R * S;
		glm::mat3x4 M = glm::mat3x4(
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

		P = world2ndc * ndc2pix;
		T = glm::transpose(M) * P;
		normal = transformVec4x3({L[2].x, L[2].y, L[2].z}, viewmatrix);
	}

	// Update gradients w.r.t. transformation matrix of the Gaussian
	glm::mat3 dL_dT = glm::mat3(
		dL_dTs[idx*9+0], dL_dTs[idx*9+1], dL_dTs[idx*9+2],
		dL_dTs[idx*9+3], dL_dTs[idx*9+4], dL_dTs[idx*9+5],
		dL_dTs[idx*9+6], dL_dTs[idx*9+7], dL_dTs[idx*9+8]
	);
	float3 dL_dmean2D = dL_dmean2Ds[idx];
	if(dL_dmean2D.x != 0 || dL_dmean2D.y != 0)
	{
		glm::vec3 t_vec = glm::vec3(9.0f, 9.0f, -1.0f);
		float d = glm::dot(t_vec, T[2] * T[2]);
		glm::vec3 f_vec = t_vec * (1.0f / d);
		glm::vec3 dL_dT0 = dL_dmean2D.x * f_vec * T[2];
		glm::vec3 dL_dT1 = dL_dmean2D.y * f_vec * T[2];
		glm::vec3 dL_dT3 = dL_dmean2D.x * f_vec * T[0] + dL_dmean2D.y * f_vec * T[1];
		glm::vec3 dL_df = dL_dmean2D.x * T[0] * T[2] + dL_dmean2D.y * T[1] * T[2];
		float dL_dd = glm::dot(dL_df, f_vec) * (-1.0 / d);
		glm::vec3 dd_dT3 = t_vec * T[2] * 2.0f;
		dL_dT3 += dL_dd * dd_dT3;
		dL_dT[0] += dL_dT0;
		dL_dT[1] += dL_dT1;
		dL_dT[2] += dL_dT3;

		if (Ts_precomp != nullptr) {
			dL_dTs[idx * 9 + 0] = dL_dT[0].x;
			dL_dTs[idx * 9 + 1] = dL_dT[0].y;
			dL_dTs[idx * 9 + 2] = dL_dT[0].z;
			dL_dTs[idx * 9 + 3] = dL_dT[1].x;
			dL_dTs[idx * 9 + 4] = dL_dT[1].y;
			dL_dTs[idx * 9 + 5] = dL_dT[1].z;
			dL_dTs[idx * 9 + 6] = dL_dT[2].x;
			dL_dTs[idx * 9 + 7] = dL_dT[2].y;
			dL_dTs[idx * 9 + 8] = dL_dT[2].z;
			return;
		}
	}
	
	if (Ts_precomp != nullptr) return;

	// Update gradients w.r.t. scaling, rotation, position of the Gaussian
	glm::mat3x4 dL_dM = P * glm::transpose(dL_dT);
	float3 dL_dtn = transformVec4x3Transpose(dL_dnormals[idx], viewmatrix);
#if DUAL_VISIABLE
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);
	float cos = -sumf3(p_view * normal);
	float multiplier = cos > 0 ? 1: -1;
	dL_dtn = multiplier * dL_dtn;
#endif
	glm::mat3 dL_dRS = glm::mat3(
		glm::vec3(dL_dM[0]),
		glm::vec3(dL_dM[1]),
		glm::vec3(dL_dtn.x, dL_dtn.y, dL_dtn.z)
	);

	glm::mat3 dL_dR = glm::mat3(
		dL_dRS[0] * glm::vec3(scale.x),
		dL_dRS[1] * glm::vec3(scale.y),
		dL_dRS[2]);
	// if(idx == 11834) printf("scale: %f %f \n", scale.x,scale.y);
	glm::vec4 dldr = quat_to_rotmat_vjp(rot, dL_dR);

	dL_drots[idx] += dldr;
	// dL_drots[idx].y += dldr.y;
	// dL_drots[idx].z += dldr.z;
	// dL_drots[idx].w += dldr.w;
	dL_dscales[idx] += glm::vec3(
		(float)glm::dot(dL_dRS[0], R[0]),
		(float)glm::dot(dL_dRS[1], R[1]),
		0
	);
	dL_dmeans[idx] += glm::vec3(dL_dM[2]);
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
	float* dL_dtransMats,
	const float* dL_dview2gaussian,
	const float* dL_dnormal3Ds,
	float* dL_dcolors,
	float* dL_dshs,
	// grad output
	float3* dL_dmean2Ds,
	glm::vec3* dL_dmean3Ds,
	glm::vec3* dL_dscales,
	glm::vec4* dL_drots)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P || !(radii[idx] > 0))
		return;

	const int W = int(focal_x * tan_fovx * 2);
	const int H = int(focal_y * tan_fovy * 2);
	const float * Ts_precomp = (scales) ? nullptr : transMats;
	compute_transmat_aabb(
		idx, 
		Ts_precomp,
		means3D, scales, rotations, 
		projmatrix, viewmatrix, W, H, 
		(float3*)dL_dnormal3Ds, 
		dL_dmean2Ds,
		(dL_dtransMats), 
		dL_dmean3Ds, 
		dL_dscales, 
		dL_drots
	);
	float depth = transMats[idx * 9 + 8];

	computeView2Gaussian_backward(idx, means3D[idx], rotations[idx], viewmatrix, view2gaussian + 16 * idx, dL_dview2gaussian + 16 * idx, dL_dmean3Ds, dL_drots);

	if (shs)
		computeColorFromSH(idx, D, M, (glm::vec3*)means3D, *campos, shs, clamped, (glm::vec3*)dL_dcolors, (glm::vec3*)dL_dmean3Ds, (glm::vec3*)dL_dshs);
	dL_dmean2Ds[idx].x = dL_dtransMats[idx * 9 + 2] * depth * 0.5 * float(W); // to ndc 
	dL_dmean2Ds[idx].y = dL_dtransMats[idx * 9 + 5] * depth * 0.5 * float(H); // to ndc
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
		dL_dmean2Ds,
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
	const float* max_alphas,
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
	if(lambda == 0){
		renderCUDA_2DGS<NUM_CHANNELS> << <grid, block >> >(
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			bg_color,
			means2D,
			normal_opacity,
			transMats,
			colors,
			depths,
			final_Ts,
			n_contrib,
			dL_dpixels,
			dL_depths,
			dL_dtransMat,
			dL_dmean2D,
			dL_dnormal3D,
			dL_dopacity,
			dL_dcolors);
	}
	else{
		renderCUDA_GUDF<NUM_CHANNELS> << <grid, block >> >(
			ranges,
			point_list,
			W, H,
			focal_x, focal_y,
			bg_color,
			means2D,
			normal_opacity,
			means3D,
			kappas,
			scales,
			view2gaussians,
			colors,
			final_Ts,
			n_contrib,
			dL_dpixels,
			dL_depths,
			dL_dtransMat,
			dL_dview2gaussian,
			dL_dscale,
			dL_dnormal3D,
			dL_dopacity,
			dL_dcolors,
			dL_dkappas
		);
	}

	
}
