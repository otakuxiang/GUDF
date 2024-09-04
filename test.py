# Test the intersect model between 2d Gaussian and ray

'''
datas:
pixf: 240.500000 136.500000
W H: 400 300
focal_x focal_y: 718.557739 717.444580
viewmat: 0.792568 -0.038003 -0.608598 0.000000 
-0.509950 0.505929 -0.695692 0.000000 
0.334346 0.861738 0.381603 0.000000 
-0.222192 0.114435 2.870208 1.000000 
V2G: -0.019018 0.999735 -0.012972 0.000000 
0.999717 0.019200 0.014055 0.000000 
0.014300 -0.012701 -0.999817 0.000000 
0.010905 -0.089315 2.869614 1.000000 
quat: -0.135285 0.671427 0.485866 0.542972
-0.061768 0.505535 0.860592 
0.799358 -0.491265 0.345955 
0.597671 0.709291 -0.373759 
scale: 0.000259 0.030476 0.317039
normal: -0.012972 0.014055 -0.999817
p_world: 0.284236 -0.259203 -0.025547
kappa: 98.168884
'''
from re import T
import torch
import quaternion
import numpy as np
from torchviz import make_dot
torch.set_printoptions(precision=7)

# torch.set_default_dtype(torch.float64)
# dT_dscale = torch.tensor([-4522.342773, -4771.300781, -170.739288])
# pixf = torch.tensor([115.5, 139.5])
# W, H = 400, 300
# focal_x, focal_y = 718.557739, 717.444580
# viewmat = torch.tensor([[-0.638657, 0.031858, -0.768832, 0.000000],
#                         [-0.722815, 0.317854, 0.613602, 0.000000],
#                         [0.263925, 0.947604, -0.179972, 0.000000],
#                         [0.217348, 0.092170, 2.922817, 1.000000]]).T
# V2G = torch.tensor([[-0.663734, 0.689055, -0.290963, 0.000000],
#                     [0.285048, -0.126624, -0.950112, 0.000000],
#                     [-0.691523, -0.713561, -0.112369, 0.000000],
#                     [1.488900, 1.714491, 1.028497, 1.000000]]).T
# q_rot = torch.tensor([[0.088875, 0.987129, 0.063453],
#                       [0.964645, 0.146041, 0.219392],
#                       [0.104504, -0.976150, 0.190290]]).T
# scale = torch.tensor([0.019803, 0.663397, 0.419015])
# normal_o = torch.tensor([-0.290963, -0.950112, -0.112369])
# p_world = torch.tensor([0.518754, -0.044268, 0.712586])
# kappa = 80.127266

# pixf = torch.tensor([179.500000, 39.500000])
# W, H = 400, 300
# focal_x, focal_y = 718.557739, 717.444580
# viewmat = torch.tensor([[-0.638657, 0.031858, -0.768832, 0.000000],
#                         [-0.722815, 0.317854, 0.613602, 0.000000],
#                         [0.263925, 0.947604, -0.179972, 0.000000],
#                         [0.217348, 0.092170, 2.922817, 1.000000]]).T
# V2G = torch.tensor([[-0.663734, 0.689055, -0.290963, 0.000000],
#                     [0.285048, -0.126624, -0.950112, 0.000000],
#                     [-0.691523, -0.713561, -0.112369, 0.000000],
#                     [1.488900, 1.714491, 1.028497, 1.000000]]).T
# q_rot = torch.tensor([[0.088875, 0.987129, 0.063453],
#                       [0.964645, 0.146041, 0.219392],
#                       [0.104504, -0.976150, 0.190290]]).T
# scale = torch.tensor([0.019803, 0.663397, 0.419015])
# normal_o = torch.tensor([-0.290963, -0.950112, -0.112369])
# p_world = torch.tensor([0.518754, -0.044268, 0.712586])
# kappa = 80.127266

pixf = torch.tensor([240.500000, 136.500000])
W, H = 400, 300
focal_x, focal_y = 718.557739, 717.444580
viewmat = torch.tensor([[-0.792568, -0.038003, -0.608598, 0.000000],
                        [-0.509950, 0.505929, -0.695692, 0.000000],
                        [0.334346, 0.861738, 0.381603, 0.000000],
                        [-0.222192, 0.114435, 2.870208, 1.000000]]).T
V2G = torch.tensor([[-0.019018, 0.999735, -0.012972, 0.000000],
                    [0.999717, 0.019200, 0.014055, 0.000000],
                    [0.014300, -0.012701, -0.999817, 0.000000],
                    [0.010905, -0.089315, 2.869614, 1.000000]]).T
q_rot = torch.tensor([[-0.061768,0.505535,0.860592 ],
                      [0.799358,-0.491265,0.345955 ],
                      [0.597671,0.709291,-0.373759 ]]).T
scale = torch.tensor([0.000259, 0.030476, 0.317039])
normal_o = torch.tensor([-0.012972, 0.014055, -0.999817])
p_world = torch.tensor([0.284236, -0.259203, -0.025547])
kappa = 98.168884


p_world = p_world.clone().detach().requires_grad_(True)
scale = scale.clone().detach().requires_grad_(True)
normal_o = normal_o.clone().detach().requires_grad_(True)
# kappa = 5.

def compute_depth(p_world, scale, q_rot, viewmat, pixf, focal_x, focal_y, V2G = None):

    if V2G is None:
        G2W = torch.eye(4)
        G2W[:3,:3] = q_rot
        G2W[:3,3] = p_world
        G2V = viewmat @ G2W
        V2G = torch.linalg.inv(G2V)

    # compute the ray origin and direction
    # ray_origin0 = -viewmat[:3, :3].T @ viewmat[:3, 3:4]
    # breakpoint() 
    ray_origin1 = V2G[:3, 3]
    ray_direction = torch.tensor([(pixf[0] - W/2)/focal_x, (pixf[1] - H/2)/focal_y, 1.0])
    ray_direction = ray_direction / torch.linalg.norm(ray_direction)
    
    ray_direction = (V2G[:3, :3] @ ray_direction.unsqueeze(-1)).squeeze(-1) 

    normal = (V2G[:3, :3] @ normal_o.unsqueeze(-1)).squeeze(-1)
    cos = ray_direction.dot(normal).abs() 
    
    # print(ray_origin1,ray_direction)
    t_star = (- ray_origin1).dot(normal) / normal.dot(ray_direction)

    ray_origin1_scaled = ray_origin1 / 3 / scale
    ray_direction_scaled = ray_direction / 3 / scale 


    A = ray_direction_scaled.dot(ray_direction_scaled)
    B = 2 * (ray_origin1_scaled.dot(ray_direction_scaled))
    C = ray_origin1_scaled.dot(ray_origin1_scaled) - 1
    delta = B*B - 4*A*C
    sd = delta.sqrt()
    tn = (-B - sd) / (2*A)
    tf = (-B + sd) / (2*A)
# near_depth = ray_origin1 + tn * ray_direction
# far_depth = ray_origin1 + tf * ray_direction
    # print(t_star,(tf-tn)*cos)
    return t_star,tf,tn,cos

# inter_point = ray_origin1 + t_star * ray_direction
# print(torch.linalg.norm(inter_point))
# print(t_star,near_depth[2],far_depth[2])
# ftn = cos * (t_star - tn)
# ftf = cos * (t_star - tf)
# E = (kappa * ftf).exp()
# F = (kappa * ftn).exp()
# A = (-kappa * cos * (tf - tn)).exp()
# B = 1 + E
# C = 1 + F
# T = A * C / B
# T.backward()
# dA_dn = torch.autograd.grad(A, normal, retain_graph=True)[0]
# dcos_dn = torch.autograd.grad(cos, normal, retain_graph=True)[0]
# # print(dcos_dn)
#T.backward()
# print(T,p_world.grad)


def analytic_integral(tn,tf,kappa,cos,depth):
    ftn = cos * (depth - tn)
    ftf = cos * (depth - tf)
    E = (kappa * ftf).exp()
    F = (kappa * ftn).exp()
    A = (-kappa * cos * (tf - tn)).exp()
    B = 1 + E
    C = 1 + F
    lnE = kappa * ftf
    lnF = kappa * ftn
    lnA = -kappa * cos * (tf - tn)
    lnB = lnE + (1 / E + 1).log()
    lnC = lnF + (1 / F + 1).log()
    lnT = lnA + lnC - lnB
    
    return lnT.exp()


def SDF(t,depth,cos):
    return cos * (depth - t)

def sigma(t,kappa,cos,depth):
    return kappa * cos * (1 - torch.sigmoid(kappa * SDF(t,depth,cos)))

def numeric_integral(t0,t1,kappa,cos,depth):
    t_vals = torch.linspace(0., 1., steps=256)
    t_vals = t0 * (1 - t_vals) + t1 * t_vals
    sigmas = sigma(t_vals,kappa,cos,depth)
    dists = t_vals[1:] - t_vals[:-1]
    sample_dist = dists[:1]
    dists = torch.cat([dists, sample_dist * 2], -1) 
    samples = sigmas * dists 
    integral = -torch.sum(samples, -1)
    return 1 - integral.exp()

def analytic_diff_ln(tn,tf,kappa,cos,depth):
    flagtf = -1 if (depth < tf) else 1
    flagtn = 1 if (depth > tn) else -1

    ftn = cos * (depth - tn)
    ftf = cos * (depth - tf)

    E = (kappa * ftf).exp()
    F = (kappa * ftn).exp()
    A = (-kappa * cos * (tf - tn)).exp()
    B = 1 + E
    C = 1 + F
    lnE = kappa * ftf
    lnF = kappa * ftn
    lnA = -kappa * cos * (tf - tn)
    
    lnB = B.log() 
    lnC = C.log() 
    if E.isnan() or E.isinf():
        if lnE > 0:
            lnB = lnE 
        else:
            lnB = -1e10
    if F.isnan() or F.isinf():
        if lnF > 0:
            lnC = lnF 
        else:
            lnC = -1e10
    print(lnB,lnC)
    lntfd = (tf-depth).abs().log()
    lntnd = (depth-tn).abs().log()
    lntftn = (tf - tn).log()
    lncos = cos.log()
    lnkappa = kappa.log()
    lndT_dA = lnC - lnB
    lndT_dB = (lnA + lnC - 2*lnB) 
    lndT_dC = lnA - lnB
    print(lndT_dA,lndT_dB,lndT_dC)
    lndA_dkappa = lncos + lntftn + lnA
    lndB_dkappa = lnE + lncos + lntfd
    lndC_dkappa = lnF + lncos + lntnd
    dT_dkappa = -(lndT_dA + lndA_dkappa).exp() - flagtf * (lndT_dB + lndB_dkappa).exp() \
        + flagtn * (lndT_dC + lndC_dkappa).exp()
    lnkc = lnkappa + lncos
    lndB_ddepth = lnkc + lnE
    lndC_ddepth = lnkc + lnF
    dT_ddepth = (lndT_dC + lndC_ddepth).exp() - (lndT_dB + lndB_ddepth).exp()
    
    lndAdtn = lnkc + lnA
    lndBdtf = lnkc + lnE 
    lndCdtn = lnkc + lnF
    dTdtf = - (lndT_dA + lndAdtn).exp() + (lndT_dB + lndBdtf).exp() 
    dTdtn = (lndT_dA+lndAdtn).exp() - (lndT_dC + lndCdtn).exp()
    
    lndAdcos = lntftn + lnkappa + lnA
    lndBdcos = lntfd + lnkappa + lnE
    
    lndCdcos = lntnd + lnkappa + lnF
    dTdcos = - (lndT_dA + lndAdcos).exp() - flagtf * (lndT_dB + lndBdcos).exp() + flagtn * (lndT_dC + lndCdcos).exp()
    print(lndT_dA + lndAdcos, lndT_dB + lndBdcos,lndT_dC + lndCdcos)
    # dT_dA = torch.autograd.grad(T, A, create_graph=True)[0]
    # dT_dC = torch.autograd.grad(T, C, create_graph=True)[0]
    # dT_dB = torch.autograd.grad(T, B, create_graph=True)[0]
    # dA_dtn = torch.autograd.grad(A, tn, create_graph=True)[0]
    # dC_dtn = torch.autograd.grad(C, tn, create_graph=True)[0]
    # dT_dtn = torch.autograd.grad(T, tn, create_graph=True)[0]
    # dB_dtf = torch.autograd.grad(B, tf, create_graph=True)[0]
    # dA_dtf = torch.autograd.grad(A, tf, create_graph=True)[0]
    # print(dT_dA * dA_dcos + dT_dC * dC_dcos +dT_dB * dB_dcos,dT_dcos)
    # print(lndT_dA + lndAdtn,lnA + lnC - 2*lnB  + lnE)
    # print(lndT_dA + lndAdtn,lnA - lnB + lnF)

    # print(flagtf,dB_dcos,lndBdcos.exp())
    # print(dAdcos)
    # print(dBdcos)
    # print(dCdcos)
    # print(dTdcos)
    
    
    return dT_ddepth, dTdtn, dTdtf, dTdcos
    
def analytic_diff(tn,tf,kappa,cos,depth):
    ftn = cos * (depth - tn)
    ftf = cos * (depth - tf)
    E = (kappa * ftf).exp()
    F = (kappa * ftn).exp()

    A = (-kappa * cos * (tf - tn)).exp()
    B = 1 + E
    C = 1 + F
    B2 = B.pow(2)

    dA_dkappa = cos * (tf - tn) * A
    dB_dkappa = E * ftf
    dC_dkappa = F * ftn
    dT_dA = C / (B + 1e-6)
    dT_dB = - A * C / (B2 + 1e-6)
    dT_dC = A / (B + 1e-6)
    # print("C dT_dA",C,dT_dA,dA_dkappa)
    dT_dkappa =  - dT_dA * dA_dkappa + dT_dB * dB_dkappa + dT_dC * dC_dkappa

    dB_ddepth = cos * kappa * E
    dC_ddepth = cos * kappa * F

    dT_ddepth = dT_dC * dC_ddepth - dT_dB * dB_ddepth
    
    dAdtn = kappa * cos * A
    dAdtf = kappa * cos * A
    dBdtf =  E * kappa * cos
    dCdtn =  F * kappa * cos
    dTdtn = dT_dA * dAdtn - dT_dC * dCdtn 
    dTdtf = - dT_dA * dAdtf + dT_dB * dBdtf
    
    
    
    dAdcos = -(tn - tf) * A * kappa
    dBdcos = -E * kappa * (depth - tf)
    dCdcos = F * kappa * (depth - tn)
    dTdcos = - dT_dA * dAdcos + dT_dB * dBdcos + dT_dC * dCdcos
    
    
 
    return dT_ddepth, dTdtn, dTdtf, dTdcos

# backward function for compute_depth
def analytic_grad_depth(p_world, scale, q_rot, viewmat, pixf, focal_x, focal_y, dTddepth, dTdtn, dTdtf, dTdcos,T,V2G=None):
    dTdx = torch.zeros_like(p_world)
    dTdn = torch.zeros_like(normal_o)
    p_world_1 = p_world.clone().detach().requires_grad_(True)
    normal_o_1 = normal_o.clone().detach().requires_grad_(True)
    scale = scale.clone().detach().requires_grad_(True)
    if V2G is None:
        G2W = torch.eye(4)
        G2W[:3,:3] = q_rot
        G2W[:3,3] = p_world_1
        G2V = viewmat @ G2W
        V2G = torch.linalg.inv(G2V)
    else:
        G2V = torch.linalg.inv(V2G)
    

    ray_origin1 = V2G[:3, 3]
    ray_direction_0 = torch.tensor([(pixf[0] - W/2)/focal_x, (pixf[1] - H/2)/focal_y, 1.0])
    ray_direction_0 = ray_direction_0 / torch.linalg.norm(ray_direction_0)
    ray_direction = (V2G[:3, :3] @ ray_direction_0.unsqueeze(-1)).squeeze(-1)

    normal = V2G[:3, :3] @ normal_o_1
    # breakpoint()
    
    # normal = normal.clone().detach().requires_grad_(True)
    # ray_direction = ray_direction.detach()

    
    cos = ray_direction.dot(normal).abs() 
    dcosdn = ray_direction 
    dcosdr = normal  - ray_direction.dot(normal) * ray_direction 
     
    if ray_direction.dot(normal) < 0:
        dcosdn = -dcosdn
        dcosdr = -dcosdr
    t_star = (- ray_origin1).dot(normal) / normal.dot(ray_direction)
    ddepthdn = (- ray_origin1) / normal.dot(ray_direction) - (- ray_origin1).dot(normal) / (normal.dot(ray_direction)**2) * ray_direction
    ddepth_do = - normal / normal.dot(ray_direction)
    ddepth_dr = -(- ray_origin1).dot(normal) / (normal.dot(ray_direction)**2) * normal
    # print("ddepth_dn",ddepth_dn,ddepthdn)

    dTdno = dTddepth * ddepthdn + dTdcos * dcosdn

    dTdn[0] = dTdno[0] * V2G[0,0] + dTdno[1] * V2G[1,0] + dTdno[2] * V2G[2,0]
    dTdn[1] = dTdno[0] * V2G[0,1] + dTdno[1] * V2G[1,1] + dTdno[2] * V2G[2,1]
    dTdn[2] = dTdno[0] * V2G[0,2] + dTdno[1] * V2G[1,2] + dTdno[2] * V2G[2,2]
    
    ray_origin1_scaled = ray_origin1 / 3 / scale
    ray_direction_scaled = ray_direction / 3 / scale
    
    
    
    A = ray_direction_scaled.dot(ray_direction_scaled)
    B = 2 * (ray_origin1_scaled.dot(ray_direction_scaled))
    C = ray_origin1_scaled.dot(ray_origin1_scaled) - 1
    delta = B*B - 4*A*C
    sd = delta.sqrt()
    tn = (-B - sd) / (2*A)
    tf = (-B + sd) / (2*A)
    # print(tn,tf)
    tnogrg = (ray_origin1_scaled + tn * ray_direction_scaled).dot(ray_direction_scaled)
    tfogrg = (ray_origin1_scaled + tf * ray_direction_scaled).dot(ray_direction_scaled)
    dtndrg = - tn / tnogrg * (ray_origin1_scaled + tn * ray_direction_scaled)
    dtfdrg = - tf / tfogrg * (ray_origin1_scaled + tf * ray_direction_scaled)
    dtndog = - (ray_origin1_scaled + tn * ray_direction_scaled) / tnogrg
    dtfdog = - (ray_origin1_scaled + tf * ray_direction_scaled) / tfogrg
    dTdog = dTdtn * dtndog + dTdtf * dtfdog
    dTdrg = dTdtn * dtndrg + dTdtf * dtfdrg
    # dtndr = - tn /  
    
    dTdVG = torch.zeros_like(V2G)
    dTdo = dTdog / 3 / scale + dTddepth * ddepth_do
    dTdr = dTdrg / 3 / scale + dTddepth * ddepth_dr + dTdcos * dcosdr
    dTdVG[:3,3] = dTdo
    
    dTdVG[0,:3] = dTdr[0] * ray_direction_0 + dTdno[0] * normal_o_1
    dTdVG[1,:3] = dTdr[1] * ray_direction_0 + dTdno[1] * normal_o_1
    dTdVG[2,:3] = dTdr[2] * ray_direction_0 + dTdno[2] * normal_o_1
    # dTdVG[:3,0] += dTdn[0] * normal_o_1
    # dTdVG[:3,1] += dTdn[1] * normal_o_1
    # dTdVG[:3,2] += dTdn[2] * normal_o_1
    dT_dGVR = dTdVG[:3,:3].T
    dT_dGVt = dTdVG[:3,3]
    
    dT_dGVR_from_t = torch.zeros_like(G2V[:3,:3])
    dT_dGVR_from_t[:3,0] = - G2V[:3,3] * dT_dGVt[0]
    dT_dGVR_from_t[:3,1] = - G2V[:3,3] * dT_dGVt[1]
    dT_dGVR_from_t[:3,0] = - G2V[:3,3] * dT_dGVt[2]
    dT_dGV_R = dT_dGVR + dT_dGVR_from_t
    dT_dGV_t = - dT_dGVt @ G2V[:3,:3].T
    dT_dGV = torch.zeros_like(G2V)
    dT_dGV[:3,:3] = dT_dGV_R
    dT_dGV[:3,3] = dT_dGV_t
    dT_dGW = viewmat.T @ dT_dGV
    dTdx = dT_dGW[:3,3]
    dogdscale = -ray_origin1 / 3 / scale**2
    drgdscale = -ray_direction / 3 / scale**2
    # dtnds = dtndog * dogdscale + dtndrg * drgdscale
    # dtfds = dtfdog * dogdscale + dtfdrg * drgdscale
    dTds = dTdrg * drgdscale + dTdog * dogdscale
    T = analytic_integral(tn,tf,torch.tensor([kappa]), cos,t_star)
    # dtn_ds
    # make_dot(T).render('test')

    # dT_dog = torch.autograd.grad(T,ray_origin1_scaled,retain_graph=True)[0]
    # print("dT_dog",dTdog)
    # # dT_drg = torch.autograd.grad(T,ray_direction_scaled,retain_graph=True)[0]
    # print("dT_drg",dTdrg)
    # print("dT_ds",dTds)
    # dtn_dog = torch.autograd.grad(tn,ray_origin1_scaled,retain_graph=True)[0]
    # dtf_dog = torch.autograd.grad(tf,ray_origin1_scaled,retain_graph=True)[0]
    # print("dtn_dog",dtn_dog,dtndog) 
    # print("dtf_dog",dtf_dog,dtfdog) 
    # dtn_drg = torch.autograd.grad(tn,ray_direction_scaled,retain_graph=True)[0]
    # dtf_drg = torch.autograd.grad(tf,ray_direction_scaled,retain_graph=True)[0]
    # print("dtn_drg",dtn_drg,dtndrg) 
    # print("dtf_drg",dtf_drg,dtfdrg) 
    # dogdscale_0 = torch.autograd.grad(ray_origin1_scaled[0],scale,retain_graph=True)[0]
    # print("dogdscale_0",dogdscale_0[0],dogdscale[0])
    
    # dogdscale_1 = torch.autograd.grad(ray_origin1_scaled[1],scale,retain_graph=True)[0]
    # print("dogdscale_1",dogdscale_1[1],dogdscale[1])
    # dogdscale_2 = torch.autograd.grad(ray_origin1_scaled[2],scale,retain_graph=True)[0]
    # print("dogdscale_2",dogdscale_2[2],dogdscale[2])
    
    # dog_dscale = dogdscale_0 + dogdscale_1 + dogdscale_2
    
    # drgdscale_0 = torch.autograd.grad(ray_direction_scaled[0],scale,retain_graph=True)[0]
    # print("drgdscale_0",drgdscale_0,drgdscale[0])
    
    # drgdscale_1 = torch.autograd.grad(ray_direction_scaled[1],scale,retain_graph=True)[0]
    # print("drgdscale_1",drgdscale_1,drgdscale[1])
    
    # drgdscale_2 = torch.autograd.grad(ray_direction_scaled[2],scale,retain_graph=True)[0]
    # print("drgdscale_2",drgdscale_2,drgdscale[2])


    # drg_dscale = drgdscale_0 + drgdscale_1 + drgdscale_2
    
    # dtnds = dtn_dog * dog_dscale + dtn_drg * drg_dscale
    # dtfds = dtf_dog * dog_dscale + dtf_drg * drg_dscale
    # dtn_ds = torch.autograd.grad(tn,scale,retain_graph=True)[0]
    # dtf_ds = torch.autograd.grad(tf,scale,retain_graph=True)[0]
    # print("dtnds",dtnds,dtn_ds)
    # print("dtfds",dtfds,dtf_ds)
    # dT_dtn = torch.autograd.grad(T,tn,retain_graph=True)[0]
    # dT_dtf = torch.autograd.grad(T,tf,retain_graph=True)[0]
    # print("dT_dtn",dT_dtn,dTdtn)
    # print("dT_dtf",dT_dtf,dTdtf)
    
    # dTds = dTdtn * dtnds + dtfds * dTdtf
    # dT_ds = torch.autograd.grad(T,scale,retain_graph=True)[0]
    # print("dT_ds",dT_ds,dTds)
    

    return dTdx, dTdn, dTds
    
    
# print(1 - analytic_integral(-0.045963 *3,0.045963 *3,torch.tensor([50]),1.0,0))
# p_world = p_world.double()
# normal_o = normal_o.double()
scale = scale.double() 
# q_rot = q_rot.double()
# viewmat = viewmat.double()
# pixf = pixf.double()
t_star,tf,tn,cos = compute_depth(p_world, scale, q_rot, viewmat, pixf, focal_x, focal_y,V2G)

T = analytic_integral(tn,
                        tf,
                        torch.tensor([kappa]), 
                        cos,t_star)
# ttn = analytic_integral(0,
#                         tn,
#                         torch.tensor([kappa]), 
#                         cos,t_star)
# ttf = analytic_integral(0,
#                         tf,
#                         torch.tensor([kappa]), 
#                         cos,t_star)
# intw = ttn - ttf

dTdtn = torch.autograd.grad(T,tn,retain_graph=True)[0]
dTdtf = torch.autograd.grad(T,tf,retain_graph=True)[0]
dTddepth = torch.autograd.grad(T,t_star,retain_graph=True)[0]
dTdcos = torch.autograd.grad(T,cos,retain_graph=True)[0]
# dTdx = torch.autograd.grad(T,p_world,retain_graph=True)[0]
dTdn = torch.autograd.grad(T,normal_o,retain_graph=True)[0] 
dTds = torch.autograd.grad(T,scale,retain_graph=True)[0]
# # print(dTds,dT_dscale) 
# # # dLdx = torch.zeros_like(p_world)
# tn = tn.double()
# tf = tf.double()
# t_star = t_star.double()
# cos = cos.double()

dT_ddepth, dT_dtn, dT_dtf, dT_dcos = analytic_diff_ln(tn,tf,torch.tensor([kappa]), cos,t_star)

# focal_x = focal_x.double()
# focal_y = focal_y.double()
# dT_ddepth = dT_ddepth.double()
# dT_dtn = dT_dtn.double()
# dT_dtf = dT_dtf.double()
# dT_dcos = dT_dcos.double()
dT_dx, dT_dn, dT_ds = analytic_grad_depth(p_world, scale, q_rot, viewmat, pixf, focal_x, focal_y, dT_ddepth, dT_dtn, dT_dtf, dT_dcos,T,V2G)


print(numeric_integral(tn,
                        tf,
                        torch.tensor([kappa]), 
                        cos,t_star),T)
print(dT_ddepth, dTddepth)
print("dT_dtn",dT_dtn, dTdtn)
print(dT_dtf, dTdtf)
print(dT_dcos, dTdcos)
# print(dT_dx, dTdx)
print(dT_dn, dTdn)
print(dT_ds, dTds)

# x = torch.rand(3,1).requires_grad_(True)
# a = 10 * x.exp()
# L = L.sum()
# dLda = torch.autograd.grad(L,a,retain_graph=True)[0]
# dLdp = torch.autograd.grad(L,p_world,retain_graph=True)[0]

# dL_dp = viewmat[:3,:3].T @ dLda
# print(dL_dp,dLdp) 
# tn = 0.01
# tf = 100
# print(autodiff(tn,tf,0.1,cos,t_star,p_world,ray_origin1,ray_direction,normal))
# print(analytic_diff(tn,tf,0.1,cos,t_star,normal,ray_direction,ray_origin1,p_world))
# print(numeric_integral(tn,tf,0.1,cos,t_star),analytic_integral(tn,tf,0.1,cos,t_star))
    

# pa = ray_o_local + tn * ray_d_local
# pb = ray_o_local + tf * ray_d_local
# print(torch.linalg.norm(pa[:2]), torch.linalg.norm(pb[:2]))
# print(cos)
# print(tn, (t_star-tn)*cos)
# print(tf, (tf-t_star)*cos)

# print(t_star)