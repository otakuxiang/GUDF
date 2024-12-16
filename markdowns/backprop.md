# Backpropagation

## backpropagation T
given the forward equation:

$$
\begin{aligned}
T = T(t_f) = \frac{1+\exp(-\kappa|\cos\theta|(t^*-t_n))}{1 + \exp(-\kappa|\cos\theta|(t^*-t_f))}
\end{aligned}
$$
let $B=1+\exp(-\kappa|\cos\theta|(t^*-t_f)),C=1 + \exp(-\kappa|\cos\theta|(t^*-t_n))$ The backpropagation of the transmittance is:
### kappa
Due to nan problem, we compute backpropagation in log space.
$$
\begin{align}
T &= \frac{C}{B} \\
\frac{\partial T}{\partial \kappa} &= -\frac{C}{B^2}\frac{\partial B}{\partial \kappa} + \frac{1}{B}\frac{\partial C}{\partial \kappa} \\
&= - \exp(\ln C - 2\ln B + \ln\frac{\partial B}{\partial \kappa}) + \exp(-\ln B + \ln\frac{\partial C}{\partial \kappa}) \\
\frac{\partial B}{\partial \kappa} &= -\exp(-\kappa|\cos\theta|(t^*-t_f))|\cos\theta|(t^*-t_f) \\
\frac{\partial C}{\partial \kappa} &= -\exp(-\kappa|\cos\theta|(t^*-t_n))|\cos\theta|(t^*-t_n) \\
\end{align}
$$
note that $\frac{\partial B}{\partial \kappa},\frac{\partial C}{\partial \kappa}$ may be negtive , so we compute signs of them $s_{f}=\text{sign}(t^* - t_f),s_{n} = \text{sign}(t^* - t_n)$:.
$$
\begin{align}
\frac{\partial T}{\partial \kappa} &= s_{f}\frac{C}{B^2}|\frac{\partial B}{\partial \kappa}| - s_{n}\frac{1}{B}|\frac{\partial C}{\partial \kappa}| &\\
 &= s_{f}\exp(\ln C-2\ln B+\ln|\frac{\partial B}{\partial \kappa}|) - s_{n}\exp(-\ln B+\ln|\frac{\partial C}{\partial \kappa}|) &\\
\ln|\frac{\partial B}{\partial \kappa}| &= -\kappa|\cos\theta|(t^*-t_f) + \ln|\cos\theta| + \ln|t^*-t_f| &\\
\ln|\frac{\partial C}{\partial \kappa}| &= -\kappa|\cos\theta|(t^*-t_n) + \ln|\cos\theta| + \ln|t^*-t_n| &\\
\end{align}

$$

### interaction depth
$$
\begin{align}

&\frac{\partial T}{\partial t^*} = -\frac{C}{B^2}\frac{\partial B}{\partial t^*} + \frac{1}{B}\frac{\partial C}{\partial t^*} \\
&\frac{\partial B}{\partial t^*} = -\exp(-\kappa|\cos\theta|(t^*-t_f))\kappa|\cos\theta| \\
&\frac{\partial C}{\partial t^*} =  -\exp(-\kappa|\cos\theta|(t^*-t_n))\kappa|\cos\theta| \\
\end{align}
$$
similarly, in log space:
$$
\begin{align}
\frac{\partial T}{\partial t^*} &= \frac{C}{B^2}|\frac{\partial B}{\partial t^*}| - \frac{1}{B}|\frac{\partial C}{\partial t^*}| &\\
&= \exp(\ln C-2\ln B+\ln|\frac{\partial B}{\partial t^*}|) - \exp(\ln B+\ln|\frac{\partial C}{\partial t^*}|) &\\
\ln|\frac{\partial B}{\partial t^*}| &= -\kappa|\cos\theta|(t^*-t_f) + \ln(\kappa|\cos\theta|)  &\\
\ln|\frac{\partial C}{\partial t^*}| &= -\kappa|\cos\theta|(t^*-t_n) + \ln(\kappa|\cos\theta|)  &\\
\end{align}
$$
### interaction of nearer and farther surface
$$
\begin{align}
\frac{\partial T}{\partial t_f} &= -\frac{C}{B^2}\frac{\partial B}{\partial t_f} &\\
\frac{\partial B}{\partial t_f} &= \exp(-\kappa|\cos\theta|(t^*-t_f))\kappa|\cos\theta| &\\
\frac{\partial T}{\partial t_n} &= \frac{1}{B}\frac{\partial C}{\partial t_n} &\\
\frac{\partial C}{\partial t_n} &= \exp(-\kappa|\cos\theta|(t^*-t_n))\kappa|\cos\theta| &\\
\end{align}
$$
similarly, in log space:
$$
\begin{align}
\frac{\partial T}{\partial t_f} &= -\frac{C}{B^2}\frac{\partial B}{\partial t_f} = -\exp(\ln C - 2\ln B + \ln\frac{\partial B}{\partial t_f}) &\\
\ln\frac{\partial B}{\partial t_f} &= -\kappa|\cos\theta|(t^*-t_f) + \ln(\kappa|\cos\theta|) = \ln|\frac{\partial B}{\partial t^*}|&\\
\frac{\partial T}{\partial t_n} &= \frac{1}{B}\frac{\partial C}{\partial t_n} = \exp(-\ln B + \ln\frac{\partial C}{\partial t_n}) &\\
\ln\frac{\partial C}{\partial t_n} &= -\kappa|\cos\theta|(t^*-t_n) + \ln(\kappa|\cos\theta|) = \ln|\frac{\partial C}{\partial t^*}|&\\
\end{align}
$$

### cosine of angle of incidence
$$
\begin{align}
\frac{\partial T}{\partial \cos} &= -\frac{C}{B^2}\frac{\partial B}{\partial \cos} + \frac{1}{B}\frac{\partial C}{\partial \cos} &\\
\frac{\partial B}{\partial \cos} &= -\exp(-\kappa|\cos\theta|(t^*-t_f))\kappa(t^*-t_f) &\\
\frac{\partial C}{\partial \cos} &= -\exp(-\kappa|\cos\theta|(t^*-t_n))\kappa(t^*-t_n) &\\

\end{align}
$$
also, in log space:
$$
\begin{align}
\frac{\partial T}{\partial \cos} &= s_f\frac{C}{B^2}|\frac{\partial B}{\partial \cos}| - s_n\frac{1}{B}|\frac{\partial C}{\partial \cos}| &\\
&= s_f\exp(\ln C-2\ln B+\ln|\frac{\partial B}{\partial \cos}|) - s_n\exp(-\ln B+\ln|\frac{\partial C}{\partial \cos}|) &\\
\ln|\frac{\partial B}{\partial \cos}| &= -\kappa|\cos\theta|(t^*-t_f) + \ln\kappa + \ln|t^*-t_f| &\\
\ln|\frac{\partial C}{\partial \cos}| &= -\kappa|\cos\theta|(t^*-t_n) + \ln\kappa + \ln|t^*-t_n| &\\
\end{align}
$$
## backpropagation D
given the forward equation:
$$
\begin{equation}
d = \begin{cases}
            -T(t_f)t_f + t_n +  \frac{1 + \exp(\kappa|\cos\theta|(t_n-t^*))}{\kappa|\cos\theta|}(\kappa|\cos\theta|(t_f - t_n) + \ln(\frac{1 + e^{-\kappa f(r(t_n))}}{1 + e^{-\kappa f(r(t_f))}})), &e^{-\kappa f(r(t_n))} < 10^8 \\
            -T(t_f)t_f + t_n + (1-T(t_f)) / ({\kappa|\cos\theta|}), &e^{-\kappa f(r(t_n))} \geq 10^8 \\
    \end{cases}
\end{equation}
$$
Note that the last item $E = \int_{t_n}^{t_f}T(t)dt$, 

### kappa
$$
\begin{align}
\frac{\partial D}{\partial \kappa} &=  - t_f\frac{\partial T}{\partial \kappa} + \frac{\partial E}{\partial \kappa} &\\
\end{align}
$$
Note that $E$ is compose of three part: $F = \frac{C}{\kappa|\cos\theta|}, G =\kappa|\cos\theta|(t_f - t_n), H = \ln{C}-\ln{B}$.
$$
\begin{align}
\frac{\partial E}{\partial \kappa} &=  (G+H)\frac{\partial F}{\partial \kappa} + F(\frac{\partial G}{\partial \kappa} + \frac{\partial H}{\partial \kappa})  &\\
\frac{\partial F}{\partial \kappa} &= \frac{1}{\kappa|\cos\theta|}\frac{\partial C}{ \partial \kappa} - \frac{C}{\kappa^2|\cos\theta|} &\\
\frac{\partial G}{\partial \kappa} &= |\cos\theta|(t_f-t_n) &\\
\frac{\partial H}{\partial \kappa} &= \frac{1}{C}\frac{\partial C}{\partial \kappa} - \frac{1}{B}\frac{\partial B}{\partial \kappa} &\\

\end{align}
$$
$$
\begin{align}
    &\frac{\partial E}{\partial \kappa} = 
(-\kappa f(r(t_n))(e^{\kappa f(r(t_f))} + 1)(|\cos\theta|\kappa(t_f - t_n) - \log(e^{-\kappa f(r(t_f))} + 1) + \log(e^{-\kappa f(r(t_n))} + 1)) &\\
&+ |\cos\theta|\kappa((t^* - t_f)(e^{\kappa f(r(t_n))} + 1) - (t^* - t_n)(e^{\kappa f(r(t_f))} + 1) + (t_f - t_n)(e^{\kappa f(r(t_f))} + 1)(e^{\kappa f(r(t_n))} + 1)) &\\
&- (e^{\kappa f(r(t_f))} + 1)(e^{\kappa f(r(t_n))} + 1)(|\cos\theta|\kappa(t_f - t_n) - \log(e^{-\kappa f(r(t_f))} + 1) + \log(e^{-\kappa f(r(t_n))} + 1)))&\\
&e^{-\kappa f(r(t_n))}/(|\cos\theta|\kappa^2(e^{\kappa f(r(t_f))} + 1))
\end{align}



$$
Also, consider the nan problem, we compute the backpropagation in log space:
$$
\begin{align}
\frac{\partial H}{\partial \kappa} &= -s_n\frac{1}{C}|\frac{\partial C}{\partial \kappa}| + s_f\frac{1}{B}|\frac{\partial B}{\partial \kappa}| &\\
&= -s_n\exp(-\ln C + \ln|\frac{\partial C}{\partial \kappa}|) + s_f\exp(-\ln B + \ln|\frac{\partial B}{\partial \kappa}|) &\\
\frac{\partial F}{\partial \kappa} &= -s_n\frac{1}{\kappa|\cos\theta|}|\frac{\partial C}{ \partial \kappa}| - \frac{C}{\kappa^2|\cos\theta|} &\\
&= -s_n\exp(-\ln(\kappa|\cos\theta|) + \ln|\frac{\partial C}{\partial \kappa}|) - \exp(\ln C - \ln(\kappa^2|\cos\theta|))&\\
\frac{\partial E}{\partial \kappa} &=  (G+H)\frac{\partial F}{\partial \kappa} + \exp(\ln C - \ln(\kappa|\cos\theta|) + \ln(\frac{\partial G}{\partial \kappa} + \frac{\partial H}{\partial \kappa}))  &\\
\end{align}
$$
When use taylor expansion to approximate, the backpropagation of $E$ is:
$$
\begin{align}
\frac{\partial E}{\partial \kappa} &=  -\frac{1 - T}{\kappa^2|\cos\theta|} - \frac{1}{\kappa|\cos\theta|}\frac{\partial T}{\partial \kappa}  &\\
\end{align}
$$
### interaction depth
$$
\begin{align}
\frac{\partial D}{\partial t^*} &=  - t_f\frac{\partial T}{\partial t^*} + \frac{\partial E}{\partial t^*} &\\
\frac{\partial E}{\partial t^*} &= (G+H)\frac{\partial F}{\partial t^*} + F(\frac{\partial G}{\partial t^*} + \frac{\partial H}{\partial t^*}) &\\
\frac{\partial F}{\partial t^*} &= \frac{1}{\kappa|\cos\theta|}\frac{\partial C}{ \partial t^*},\;\; 
\frac{\partial G}{\partial t^*} = 0 &\\
\frac{\partial H}{\partial t^*} &= \frac{1}{C}\frac{\partial C}{\partial t^*} - \frac{1}{B}\frac{\partial B}{\partial t^*} &\\ 
\end{align}
$$
also, in log space:
$$
\begin{align}
\frac{\partial H}{\partial t^*} &= -\frac{1}{C}|\frac{\partial C}{\partial t^*}| + \frac{1}{B}|\frac{\partial B}{\partial t^*}| &\\ 
&= -\exp(- \ln C + \ln|\frac{\partial C}{\partial t^*}|) + \exp(-\ln B + \ln|\frac{\partial B}{\partial t^*}|) &\\
\frac{\partial F}{\partial t^*} &= -\frac{1}{\kappa|\cos\theta|}|\frac{\partial C}{ \partial t^*}| = -\exp(-\ln(\kappa|\cos\theta|) + \ln|\frac{\partial C}{\partial t^*}|) &\\
&= -\exp(-\kappa|\cos\theta|(t^*-t_n) + \ln(\kappa|\cos\theta|) - \ln(\kappa|\cos\theta|)) &\\
&= -\exp(-\kappa|\cos\theta|(t^*-t_n)) &\\
\frac{\partial E}{\partial t^*} &= (G+H)\frac{\partial F}{\partial t^*} + \exp(\ln C - \ln(\kappa^2|\cos\theta|) + \ln\frac{\partial H}{\partial t^*}) &\\
\end{align}
$$
Again, when use taylor expansion to approximate, the backpropagation of $E$ is:
$$
\begin{align}
    \frac{\partial E}{\partial t^*} &= - \frac{1}{\kappa|\cos\theta|}\frac{\partial T}{\partial t^*}
\end{align}
$$
### interaction of nearer and farther surface
$$
\begin{align}
\frac{\partial D}{\partial t_f} &=  - t_f\frac{\partial T}{\partial t_f} -T(t_f) + \frac{\partial E}{\partial t_f} &\\
\frac{\partial D}{\partial t_n} &=  - t_f\frac{\partial T}{\partial t_n} + \frac{\partial E}{\partial t_n} +1 &\\ 
\end{align}
$$
Notice that $E = \int_{t_n}^{t_f}T(t)dt$, so we can directly compute the backpropagation of $E$ by integrating equation:
$$
\begin{align}
&\frac{\partial E}{\partial t_f} = (\exp(|\cos\theta|\kappa t^*) + \exp(|\cos\theta|\kappa t_n))/(\exp(|\cos\theta|\kappa t^*) + \exp(|\cos\theta|\kappa t_f)) &\\
&\frac{\partial E}{\partial t_n} = (|\cos\theta|\kappa(t_f - t_n) + \ln T)e^{-\kappa f(r(t_n))} -1  &\\
&\frac{\partial D}{\partial t_f} =  - t_f\frac{\partial T}{\partial t_f} ,\;\;
\frac{\partial D}{\partial t_n} =  - t_f\frac{\partial T}{\partial t_n} &\\ 
\end{align}
$$
### cosine of angle of incidence
$$
\begin{align}
\frac{\partial D}{\partial \cos} &=  - t_f\frac{\partial T}{\partial \cos} + \frac{\partial E}{\partial \cos} &\\
\frac{\partial E}{\partial \cos} &=  (G+H)\frac{\partial F}{\partial \cos} + F(\frac{\partial G}{\partial \cos} + \frac{\partial H}{\partial \cos})  &\\
\frac{\partial F}{\partial \cos} &= \frac{1}{\kappa|\cos\theta|}\frac{\partial C}{ \partial \cos} - \frac{C}{\kappa\cos^2\theta} &\\
\frac{\partial G}{\partial \cos} &= \kappa(t_f-t_n) &\\
\frac{\partial H}{\partial \cos} &= \frac{1}{C}\frac{\partial C}{\partial \cos} - \frac{1}{B}\frac{\partial B}{\partial \cos}&\\
\end{align}
$$
also, in log space:
$$
\begin{align}
\frac{\partial H}{\partial \cos} &= -s_n\frac{1}{C}|\frac{\partial C}{\partial \cos}| + s_f\frac{1}{B}|\frac{\partial B}{\partial \cos}| &\\
&= -s_n\exp(-\ln C + \ln|\frac{\partial C}{\partial \cos}|) + s_f\exp(-\ln B + \ln|\frac{\partial B}{\partial \cos}|) &\\
\frac{\partial F}{\partial \cos} &= -s_n\frac{1}{\kappa|\cos\theta|}|\frac{\partial C}{ \partial \cos}| - \frac{C}{\kappa\cos^2\theta} &\\
&= -s_n\exp(-\ln(\kappa|\cos\theta|) + \ln|\frac{\partial C}{\partial \cos}|) - \exp(\ln C - \ln(\kappa\cos^2\theta))&\\
\frac{\partial E}{\partial \cos} &=  (G+H)\frac{\partial F}{\partial \cos} + \exp(\ln C - \ln(\kappa\cos^2\theta) + \ln(\frac{\partial G}{\partial \cos} + \frac{\partial H}{\partial \cos}))  &\\
\end{align}
$$
Again, when use taylor expansion to approximate, the backpropagation of $E$ is:
$$
\begin{align}
\frac{\partial E}{\partial \cos} &= -\frac{1 - T}{\kappa\cos^2\theta} - \frac{1}{\kappa|\cos\theta|}\frac{\partial T}{\partial \cos}  &\\
\end{align}
$$
