# GUDF: UDF based on 2D Gaussian Splatting

## basic idea
For a single 2d Gaussian with its center $x$ and normal $n$, we assume a local UDF $f(p,x,n) = |(p-x)\cdot n|$ defined as the distance between the point $p$ and the 2d gaussian plane. According to [Neural UDF](https://arxiv.org/pdf/2211.14173), the opacity of the points $r(t)$ near the plane can be computed by:
$$
\sigma(t) = \kappa|\cos(\theta)|(1-\Phi_\kappa(f_s(r(t))))
$$ 
where $\kappa$ is the parameter of the Sigmoid function $\Phi_\kappa(x)$, $\theta$ is the angle between the normal $n$ and the view direction. $f_s$ is SDF. 
For UDF:
$$
\sigma(t) = \begin{cases}
    \kappa|\cos(\theta)|(1-\Phi_\kappa(f_u(r(t)))),  &t < t^* \\
    \kappa|\cos(\theta)|(1-\Phi_\kappa(-f_u(r(t)))), &t >= t^*
    \end{cases}

$$ 
The UDF $f(t)$ is given by:
$$
f_u(r(t)) = \begin{cases} |\cos(\theta)|(t^* - t) , &t < t^* \\
                             |\cos(\theta)|(t - t^*), & t >= t^*
                             \end{cases}
$$
Applying the f_u to density we get:
$$
\begin{align}
&\sigma(t) = \kappa|\cos(\theta)|(1-\Phi_\kappa(f(r(t)))) \\
&f(r(t)) = |\cos(\theta)|(t^* - t)
\end{align}

$$

where $t^*$ is intersect depth of the ray with the 2d gaussian plane.
The transmittance $T$ of the gaussian can be computed by:
$$
T = \exp(-\int_{t_n}^{t_f}\sigma(t)dt) 
$$
The opacity of the gaussian can be computed by $1-T$  

Then the final rendering equation is same as the 2D Gaussian Splatting.  
The advantages of this method are: 
1) it naturely derives UDF from 2D Gaussian, which is easy to extract the meshes. 
2) the computation of transmittance and opacity based on local UDF, which provides more accurate gradients of geometry compared to the raw 2D Gaussian Splatting.

## derivation
The transmittance and opacity of the GUDF has a closed-form solution. We first derive the transmittance $T$ of the 2d gaussian:
$$
\begin{align}
T &= \exp(-\int_{t_n}^{t_f}\sigma(t)dt) \\
  &=\exp(-\int_{t_n}^{t_f}\kappa|\cos(\theta)|(1 - \Phi_\kappa(f(r(t))))dt) \\
  &=\exp(-\kappa|\cos(\theta)|\int_{t_n}^{t_f}(1 - \Phi_\kappa(f(r(t))))dt) \\
  &=\exp(-\kappa|\cos(\theta)|(t_f - t_n - \int_{t_n}^{t_f}\Phi_\kappa(f(r(t))))dt) \\
  &=\exp(-\kappa|\cos(\theta)|(t_f - t_n) + \kappa|\cos(\theta)|\int_{t_n}^{t_f}\Phi_\kappa(f(r(t)))dt) \\
\end{align}
$$
令$A = \exp(-\kappa|\cos(\theta)|(t_f - t_n))$，则有:
$$
\begin{align}
T &= A\exp(\kappa|\cos(\theta)|\int_{t_n}^{t_f}\Phi_\kappa(f(r(t)))dt) \\
  &= A\exp(\kappa|\cos(\theta)|\int_{t_n}^{t_f} \frac{1}{1+e^{-\kappa f(r(t))}}dt) \\
\end{align}
$$
令$y = f(r(t)) = |\cos(\theta)|(t^* - t)$,分段考虑$t\in[t_n,t*]$和$t\in[t*,t_f]$则有:
$$
\begin{align}
    &dy = -|\cos(\theta)|dt & t\in[t_n,t*] 
\end{align}
$$
则公式(7)可以被写成:
$$
\begin{align*}
    T &= A\exp(\kappa|\cos(\theta)|(\int_{f(r(t_n))}^{f(r(t_f))} \frac{-1}{|\cos(\theta)|}\frac{1}{1+e^{-\kappa y}}dy))\\
    &= A\exp(-\kappa(\int_{f(r(t_n))}^{f(r(t_f))}\frac{1}{1+e^{-\kappa y}}dy))
\end{align*}
$$
又有sigmoid函数的不定积分为$\int\Phi_\kappa(x)dx = \frac{1}{\kappa}\ln(1+e^{\kappa x})$ = Y(x),则有:
$$
\begin{align}
    T &= A\exp(-\kappa(Y(f(r(t_f))) - Y(f(r(t_n)))))\\
    &= A\exp(\ln(1+e^{\kappa f(r(t_n))}) - \ln(1+e^{\kappa f(r(t_f))})) \\
    &= \frac{A(1+e^{\kappa f(r(t_n))})}{1+e^{\kappa f(r(t_f))}} \\
    T(t) &= \frac{A(1+e^{\kappa f(r(0))})}{1+e^{\kappa f(r(t))}} \\
    w(t) &= σ(t)T(t) = -T'(t) \\
    \int_{t_n}^{t_f}w(t)dt &= -\int_{t_n}^{t_f}T'(t)dt \\
    &= T(t_n) - T(t_f)
\end{align}
$$

let $B=(1+e^{\kappa f(r(t_f))}),C=(1+e^{\kappa f(r(t_n))})$ The backpropagation of the transmittance is:
$$
\begin{align}
&\frac{\partial T}{\partial \kappa} = \frac{C}{B}\frac{\partial A}{\partial \kappa} - \frac{AC}{B^2}\frac{\partial B}{\partial \kappa} + \frac{A}{B}\frac{\partial C}{\partial \kappa} \\
&\frac{\partial A}{\partial \kappa} = -\cos(\theta)(t_f - t_n)A \\
&\frac{\partial B}{\partial \kappa} = \exp(\kappa f(r(t_f)))f(r(t_f)) \\
&\frac{\partial C}{\partial \kappa} = \exp(\kappa f(r(t_n)))f(r(t_n)) \\
\end{align}
$$
and:
$$
\begin{align}
&\frac{\partial T}{\partial t^*} = (\frac{A}{B}\frac{\partial C}{\partial t^*} - \frac{AC}{B^2} \frac{\partial B}{\partial 
t^*}) \\
&\frac{\partial C}{\partial t^*} = \frac{\partial C}{\partial f}\frac{\partial f}{\partial  t^*}  \\
&\frac{\partial B}{\partial t^*} = \frac{\partial B}{\partial f}\frac{\partial f}{\partial t^*} \\
&\frac{\partial T}{\partial t_n} = \frac{C}{B}\frac{\partial A}{t_n} -\frac{AC}{B^2}\frac{\partial B}{\partial t_n} + \frac{A}{B}\frac{\partial C}{\partial t_n} \\
&\frac{\partial T}{\partial t_f} = \frac{C}{B}\frac{\partial A}{t_f} -\frac{AC}{B^2}\frac{\partial B}{\partial t_f} + \frac{A}{B}\frac{\partial C}{\partial t_f} \\

&\frac{\partial A}{\partial t_n} = \kappa|\cos(\theta)|A, \;
\frac{\partial A}{\partial t_f} = -\kappa|\cos(\theta)|A \\
&\frac{\partial B}{\partial t_n} = 0,\; \frac{\partial B}{\partial t_f} = -\kappa|\cos(\theta)|\exp(\kappa f(r(t_f))) \\
&\frac{\partial C}{\partial t_f} = 0,\; \frac{\partial C}{\partial t_n} = -\kappa|\cos(\theta)|\exp(\kappa f(r(t_n))) \\

\end{align}
$$
in order to change the center of the 2d gaussian, the $\partial t^*/\partial x$ should be computed. Suppose the center and the normal of the 2d gaussian are $x$ and $n$, the origin of camera is $o$ and the view direction is $v$, then the intersection depth $t^*$ can be computed by:
$$
\begin{align}
&t^* = \frac{(x - o)\cdot n}{v\cdot n} \\
&\frac{\partial t^*}{\partial x} = \frac{n}{v\cdot n}
\end{align}
$$
TODO: the normal could also be changed following:
$$
\begin{align}
&\frac{\partial T}{\partial n} = \frac{C}{B}\frac{\partial A}{\partial n} - \frac{AC}{B^2}\frac{\partial B}{\partial n} + \frac{A}{B}\frac{\partial C}{\partial n} \\
&\frac{\partial A}{\partial n} = -(t_f - t_n)A\frac{\partial |cos(\theta)|}{\partial n} \\
&\frac{\partial B}{\partial n} = \kappa\exp(\kappa f(r(t_f)))(|\cos(\theta)|\frac{\partial t^*}{\partial n} + (t^* - t_f)\frac{\partial |cos(\theta)|}{\partial n}) \\
&\frac{\partial C}{\partial n} = \kappa\exp(\kappa f(r(t_n)))(|\cos(\theta)|\frac{\partial t^*}{\partial n} + (t^* - t_n)\frac{\partial |cos(\theta)|}{\partial n}) \\
&\frac{\partial t^*}{\partial n} = \frac{x-o}{n\cdot v} - \frac{n\cdot(x-o)}{(n\cdot v)^2}v
\end{align}
$$
But how to backpropagate the gradients of normal to the 2d gaussian quaternion?

## Computation of $t_n,t_f$

### old solution
Suppose the ray is projected on the 3d gaussian plane, and the 2d Gaussian is rotated to orient the normal $n$ with z-axis, the equation of 2d ellipse is $x^2/a^2 + y^2/b^2 = 1$, where $a,b$ is 3sigma of the gaussian. The ray is given by $r(t) = o + t\cdot v$, where $o=(x_0,y_0),v=(c,d)$ then the intersection depth $t_n$ and $t_f$ can be computed by solving the equation:
$$
\begin{align}
    &(\frac{c^2}{a^2}+\frac{d^2}{b^2})t^2+(\frac{2cx_0}{a^2}+\frac{2dy_0}{b^2})t+(\frac{x_0^2}{a^2}+\frac{y_0^2}{b^2}-1) = 0 \\
    &t_f = \frac{-B+\sqrt{B^2-4AC}}{2A} \\ 
    &t_n = \frac{-B-\sqrt{B^2-4AC}}{2A} \\
\end{align}
$$
the gradients of $t_n,t_f$ are:
$$
\begin{align}

    &\frac{\partial t_f}{\partial B} = \frac{1}{2A}(-1+\frac{B}{\sqrt{B^2-4AC}}) \\
    &\frac{\partial t_f}{\partial C} = \frac{-1}{\sqrt{B^2-4AC}} \\
    &\frac{\partial B}{\partial x_0} = \frac{2c}{a^2},\; \frac{\partial B}{\partial y_0} = \frac{2d}{b^2},\;\frac{\partial C}{\partial x_0} = \frac{2x_0}{a^2},\;\frac{\partial C}{\partial y_0} = \frac{2y_0}{b^2} \\
    &\frac{\partial t_f}{\partial x_0} = [\frac{\partial t_f}{\partial B},\frac{\partial t_f}{\partial C}] \cdot [\frac{\partial B}{\partial x_0},\frac{\partial C}{\partial x_0}] \\
    &\frac{\partial t_f}{\partial y_0} = [\frac{\partial t_f}{\partial B},\frac{\partial t_f}{\partial C}] \cdot [\frac{\partial B}{\partial y_0},\frac{\partial C}{\partial y_0}] \\
    &\frac{\partial t_n}{\partial x_0} = [\frac{\partial t_n}{\partial B},\frac{\partial t_n}{\partial C}] \cdot [\frac{\partial B}{\partial x_0},\frac{\partial C}{\partial x_0}] \\
    &\frac{\partial t_n}{\partial y_0} = [\frac{\partial t_n}{\partial B},\frac{\partial t_n}{\partial C}] \cdot [\frac{\partial B}{\partial y_0},\frac{\partial C}{\partial y_0}] \\

\end{align}
$$

the gradient of $a,b$ are:
$$
\begin{align}
&\frac{\partial t_f}{\partial a} = [\frac{\partial t_f}{\partial A},\frac{\partial t_f}{\partial B},\frac{\partial t_f}{\partial C}] \cdot [\frac{\partial A}{\partial a},\frac{\partial B}{\partial a},\frac{\partial C}{\partial a}] \\
&\frac{\partial t_f}{\partial b} = [\frac{\partial t_f}{\partial A},\frac{\partial t_f}{\partial B},\frac{\partial t_f}{\partial C}] \cdot [\frac{\partial A}{\partial b},\frac{\partial B}{\partial b},\frac{\partial C}{\partial b}] \\
&\frac{\partial t_n}{\partial a} = [\frac{\partial t_n}{\partial A},\frac{\partial t_n}{\partial B},\frac{\partial t_n}{\partial C}] \cdot [\frac{\partial A}{\partial a},\frac{\partial B}{\partial a},\frac{\partial C}{\partial a}] \\
&\frac{\partial t_n}{\partial b} = [\frac{\partial t_n}{\partial A},\frac{\partial t_n}{\partial B},\frac{\partial t_n}{\partial C}] \cdot [\frac{\partial A}{\partial b},\frac{\partial B}{\partial b},\frac{\partial C}{\partial b}] \\
&\frac{\partial t_f}{\partial A} = \frac{1}{A}(\frac{C}{\sqrt{B^2-4AC}}) - \frac{-B+\sqrt{B^2-4AC}}{2A^2} \\
&\frac{\partial t_n}{\partial A} = \frac{1}{A}(\frac{-C}{\sqrt{B^2-4AC}}) - \frac{-B-\sqrt{B^2-4AC}}{2A^2}\\
&\frac{\partial A}{\partial a} = \frac{-2c}{a^3},\;\frac{\partial A}{\partial b} = \frac{-2d}{b^3},\;\frac{\partial B}{\partial a} = \frac{-4cx_0}{a^3},\;\frac{\partial B}{\partial b} = \frac{-4dy_0}{b^3} \\
&\frac{\partial C}{\partial a} = \frac{-2x_0^2}{a^3},\;\frac{\partial C}{\partial b} = \frac{-2y_0^2}{b^3} \\
\end{align} 
$$

### new solution
supposing 2DGS has the scale of normal direction, then the tn,tf is the intersection depth of the ray with the ellipse. tn and tf can be computed by fisrt project ray to local gaussian coordinate:
$$
\begin{align}
    &o_g = (R_k(o-p_k)) \odot s_k^{-1} \\
    &r_g = R_kr\odot s_k^{-1} \\
    &x_g = o_g + t^gr_g \\
\end{align}
$$
this result a ball centered at the origin, with radius $1$. Thus we can compute the intersection of ray and gaussian:
$$
\begin{align}
    &\|x_g\|^2 = 1 \\
    &(o_g + t^gr_g)\cdot(o_g + t^gr_g) = 1\\
    & o_g\cdot o_g + 2o_g\cdot r_g t^g + r_g\cdot r_g t^{g2} = 1 \\
    & A = r_g\cdot r_g,\; B =  2o_g\cdot r_g,\; C = o_g\cdot o_g - 1\\
    &t_f = \frac{-B+\sqrt{B^2-4AC}}{2A} \\ 
    &t_n = \frac{-B-\sqrt{B^2-4AC}}{2A} \\
\end{align}
$$
the gradients of $t_n,t_f$ follows implicit differentiation:
$$
\begin{align}
&2(o_g+t^gr_g)\cdot r_g \mathrm{d}t^g+2t^g(o_g+t^gr_g)\mathrm{d}r_g+2(o_g+t^gr_g)\mathrm{d}o_g = 0 \\
&\frac{\partial t^g}{\partial r_g} = -\frac{t^g(o_g+t^gr_g)}{(o_g+t^gr_g)\cdot r_g} \\
&\frac{\partial t^g}{\partial o_g} = -\frac{o_g+t^gr_g}{(o_g+t^gr_g)\cdot r_g}
\\
\end{align} 
$$
the gradient backprop to 3d gaussian parameter could be computed followed GOF, assuming a view2gaussian matrix, the derivatation of view2gaussian matrix is well defined in GOF.
## Computation of final rendering equation
It is enough to use the transmittance and opacity of the GUDF to compute the final rendering equation. However, the backpropagation
of Transmittance to gaussian scales is not trivial. Thus, we remain the gaussian as a coefficent of the transmittance in backpropagation, while disable it in forward pass:
$$
\begin{align}
C = \Sigma_{i=1}^N c_i\frac{G(x)}{\text{sg}[G(x)]} (1-T_i)\prod_{j=1}^i T_j
\end{align}
$$

When in backward pass, the only change is change the gradient of means3d:
$$
\begin{align}
     \frac{\partial L}{\partial x_{\text{3d}}} = 
     \frac{\partial L}{\partial\alpha}(\frac{\alpha}{G_i}\frac{\partial G_i }{\partial x_{\text{2d}}}\frac{\partial x_{\text{2d}}}{\partial x_{\text{3d}}} + \frac{\alpha}{T_i}\frac{\partial T_i}{\partial x_{\text{3d}}})
\end{align}
$$

## TODO: 
1. Assume the weight of each point is a gaussian distribution:
   $$
   \sigma(t)\exp(-\int_{0}^{t}\sigma(s)ds) = G(t,t^*,\Sigma)= -\frac{dT}{dt}(t)   
   $$

   where $T$ is the transmittance of the 2d gaussian. The transmittance can be computed by:
   $$
   \text{CDF}(t) = \frac{1}{2}\left[1+\text{erf}\left(\frac{t-t^*}{\sqrt{2}\sigma}\right)\right] = \exp(-\int_0^t\sigma(s)ds)
   $$
   where $\text{erf} = \frac{2}{\sqrt{\pi}}\int_{0}^{x}e^{-t^2}\mathrm{d}t$ 
   $$
   \begin{align}
   T = \exp(-\int_{t_n}^{t_f}\sigma(t)dt) =\exp(-(\int_0^{t_f}\sigma(t)dt-\int_{0}^{t_n}\sigma(t)dt)) = \frac{\exp(-\int_0^{t_f}\sigma(t)dt)}{\exp(-\int_{0}^{t_n}\sigma(t)dt)} 
   \end{align}
   $$
2. Suppose the SDF function is defined as a set of basis functions $B_x(x),B_y(y),B_z(z)$, assume the center is $(c_x,c_y,c_z)$ and the point is $(o_x + t d_x,o_y + t d_y,o_z + t d_z)$, then the SDF can be computed by:
$$
    f(t) = B_x(o_x + t d_x - c_x)B_y(o_y + t d_y - c_y) B_z(o_z + t d_z - c_z)
$$
Assuming the basis functions are polynomials of degree $n$:
$$
\begin{align}
&B_x(x) = \sum_{i=0}^{n}a_0^ix^i,\; B_y(y) = \sum_{i=0}^{n}a_1^iy^i,\; B_z(z) = \sum_{i=0}^{n}a_2^iz^i  \\
&h = f(t), \; \frac{\partial h}{\partial t} = 
\end{align}
$$

