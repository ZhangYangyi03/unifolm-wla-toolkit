# Robot Action, State, and Statistics Processing Specification

[Chinese](robot_action_state_processing.md) | **English**

## Abstract

This specification maps data from different robot embodiments into a unified action space and a unified state space:

- each future action is represented by a 54-dimensional vector;
- each current state is represented by a 60-dimensional vector;
- Boolean masks identify the modules that are actually available;
- end-effector and base-pose actions are represented as SE(3) transforms relative to the current state;
- all other actions retain the control semantics defined by their source dataset;
- relative-pose actions use global Z-score normalization by default;
- ordinary actions and states use 1st/99th-percentile normalization by default;
- statistics from different tasks of the same robot embodiment are merged with equal task weights, after which left/right end-effector statistics may optionally be merged.

The terms **must**, **should**, and **must not** express requirements for reproducing this data-processing pipeline.

---

## 1. Notation, Data Types, and Conventions

### 1.1 Time and dimensional notation

| Symbol | Meaning |
|---|---|
| $t$ | Current observation time and reference time of the action chunk |
| `k` | Prediction-step index inside the action chunk; `k = 0, 1, ..., H - 1` |
| $H$ | Action-chunk length |
| $f_s$ | Source frame rate |
| $f_t$ | Target frame rate |
| $D_a=54$ | Unified action dimension |
| $D_s=60$ | Unified state dimension |
| `a[t,k]` | The `k`-th future action referenced to time `t` |
| `s[t]` | State at the current time |
| `m_a` | 54-dimensional action-validity mask |
| `m_s` | 60-dimensional state-validity mask |

The default configuration is:

```yaml
target_fps: 30
chunk_size: 30
norm_type: minmax_q
rel_norm_type: zscore
state_norm_type: minmax_q
gripper_norm_type: minmax_q
```

The default action chunk therefore contains 30 action points and covers approximately the interval `[0, 29/30]` seconds.

### 1.2 Numerical types

Unified actions, unified states, offsets, and scales should be stored as `float32`. Statistics may be computed and merged in `float64` to reduce numerical error, then converted to `float32` for use by the data pipeline.

Masks are Boolean arrays:

```math
\mathbf m^a\in\{0,1\}^{54},
\qquad
\mathbf m^s\in\{0,1\}^{60}.
```

### 1.3 Coordinate frames and units

#### 1.3.1 Robot coordinate-frame definition

The robot base frame is a right-handed coordinate frame denoted by $B$:

| Axis | Positive direction |
|---|---|
| $x$ | forward |
| $y$ | left |
| $z$ | up |

The axes satisfy:

```math
\mathbf e_x\times\mathbf e_y=\mathbf e_z.
```

The absolute left and right end-effector poses are denoted by $T^B_{E_L}$ and $T^B_{E_R}$, respectively, and are both expressed in the robot base frame $B$. Thus, both arms use the same pose-coordinate convention: $x$ points forward, $y$ points left, and $z$ points up. The right arm must not use a mirrored coordinate system; the sign of the same component must have the same physical meaning for both arms.

The instantaneous local orientation of an end effector is represented by the rotation matrix $R^B_E$ in its pose. The convention above defines the reference frame in which absolute end-effector poses are expressed; it does not require the moving local end-effector axes to remain parallel to the base axes.

#### 1.3.2 Units and numerical conventions

The processing pipeline does not perform automatic unit conversion. All datasets to be combined must first be converted to the following conventions:

- translation: meters;
- rotation angle: radians;
- linear velocity: meters per second;
- angular velocity: radians per second;
- quaternion order: `xyzw`;
- Euler-angle order: `xyz`;
- left/right joint order and gripper open/close direction must be consistent at the dataset level.

Any input that does not satisfy these conventions must be converted before statistics are computed or unified vectors are constructed.

---

## 2. End-to-End Processing Procedure

For every current time $t$, construct the model input and action target in the following order:

1. read the current state;
2. convert the current state to the unified 60-dimensional representation;
3. normalize the state module by module;
4. read a future action window;
5. compute relative end-effector and base-pose actions;
6. normalize each action module independently;
7. optionally binarize gripper actions;
8. resample action sequences from the source frame rate to the target frame rate;
9. write every action module into the unified 54-dimensional action vector;
10. generate action and state validity masks;
11. output the 54-dimensional action offset and scale required for denormalization.

The state path is:

```math
\text{raw state}
\longrightarrow
\text{state-format conversion}
\longrightarrow
\text{state normalization}
\longrightarrow
\mathbf s_t.
```

The action path is:

```math
(\text{current state},\text{future actions})
\longrightarrow
\text{relative actions}
\longrightarrow
\text{action normalization}
\longrightarrow
\text{resampling}
\longrightarrow
\mathbf A_t.
```

The unified action chunk is:

```math
\mathbf A_t=
\begin{bmatrix}
\mathbf a_{t,0}^{\mathsf T}\\
\mathbf a_{t,1}^{\mathsf T}\\
\vdots\\
\mathbf a_{t,H-1}^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{H\times54}.
```

---

## 3. Unified Action Representation

### 3.1 54-dimensional action layout

| Slice | Dim. | Module | Semantics | Canonical representation |
|---|---:|---|---|---|
| `[0:6]` | 6 | Left end effector | Future pose relative to the current left end-effector state | Relative xyz + rotation vector |
| `[6:7]` | 1 | Left gripper | Future gripper command | Raw scalar; optionally binarized after normalization |
| `[7:13]` | 6 | Left dexterous hand | Future hand-control command | First 6 raw action components |
| `[13:19]` | 6 | Right end effector | Future pose relative to the current right end-effector state | Relative xyz + rotation vector |
| `[19:20]` | 1 | Right gripper | Future gripper command | Raw scalar; optionally binarized after normalization |
| `[20:26]` | 6 | Right dexterous hand | Future hand-control command | First 6 raw action components |
| `[26:29]` | 3 | Waist | Future waist action | First 3 action components |
| `[29:32]` | 3 | Torso | Future torso action | First 3 action components |
| `[32:34]` | 2 | Base translation | Future base linear-velocity command | `vx, vy` |
| `[34:35]` | 1 | Base rotation | Future base yaw-rate command | `omega_z` |
| `[35:41]` | 6 | Base pose | Future pose relative to the current base state | Relative xyz + rotation vector |
| `[41:42]` | 1 | Height | Future height command | Scalar |
| `[42:48]` | 6 | Left leg | Future left-leg joint action | First 6 action components |
| `[48:54]` | 6 | Right leg | Future right-leg joint action | First 6 action components |

### 3.2 End-effector action components

Both left and right end-effector actions use the following six-dimensional ordering:

```math
\mathbf a^{ee}_{t,k}
=
[\Delta x,\Delta y,\Delta z,\phi_x,\phi_y,\phi_z]^{\mathsf T}.
```

The first three components are relative translation expressed in the current end-effector frame. The last three components form the relative rotation vector.

A rotation vector is defined as:

```math
\boldsymbol\phi=\theta\mathbf u,
```

where $\mathbf u$ is the unit rotation axis and $\theta$ is the rotation angle. Therefore:

```math
\|\boldsymbol\phi\|_2=\theta.
```

### 3.3 Non-pose actions

The following action modules are read directly from the future action sequence and do not undergo SE(3) relative-pose conversion:

- grippers;
- left and right dexterous hands;
- waist;
- torso;
- base velocity;
- height;
- left and right legs.

These fields may represent absolute targets, velocity commands, or precomputed increments, depending on the source dataset. Every dataset mapped to the same unified slot must use the same physical control semantics.

### 3.4 Action filling rules

Initialize:

```math
\mathbf a_{t,k}=\mathbf 0\in\mathbb R^{54},
\qquad
\mathbf m^a=\mathbf 0\in\{0,1\}^{54}.
```

For each module available on the current robot:

1. write the module value into its fixed slice;
2. set the corresponding mask slice to 1.

Unavailable modules remain zero and keep a zero mask.

If an input action module is wider than its destination slice, only the leading components are retained. If it is narrower than the destination slice, the input schema is invalid; action chunks are not automatically zero-padded. Every enabled action module must therefore provide at least the required number of dimensions.

---

## 4. Unified State Representation

### 4.1 60-dimensional state layout

| Slice | Dim. | Module | Semantics | Canonical representation |
|---|---:|---|---|---|
| `[0:9]` | 9 | Left end effector | Current absolute pose | xyz + rotation-6D |
| `[9:10]` | 1 | Left gripper | Current gripper state | Scalar |
| `[10:16]` | 6 | Left dexterous hand | Current hand state | 6 dimensions |
| `[16:25]` | 9 | Right end effector | Current absolute pose | xyz + rotation-6D |
| `[25:26]` | 1 | Right gripper | Current gripper state | Scalar |
| `[26:32]` | 6 | Right dexterous hand | Current hand state | 6 dimensions |
| `[32:35]` | 3 | Waist | Current waist-joint state | First 3 components |
| `[35:38]` | 3 | Torso | Current torso-joint state | First 3 components |
| `[38:40]` | 2 | Base translation velocity | Current `vx, vy` | 2 dimensions |
| `[40:41]` | 1 | Base angular velocity | Current `omega_z` | Reserved slot |
| `[41:47]` | 6 | Base inertial state | Gravity direction and angular velocity in the body frame | `gx, gy, gz, normalized wx, normalized wy, normalized wz` |
| `[47:48]` | 1 | Height | Current height | Reserved slot |
| `[48:54]` | 6 | Left leg | Current left-leg joint state | First 6 components |
| `[54:60]` | 6 | Right leg | Current right-leg joint state | First 6 components |

The state always represents the current absolute observation. It is not made relative to itself.

### 4.2 Rotation-6D for end-effector state

Let the rotation matrix be:

```math
R=
\begin{bmatrix}
R_{00}&R_{01}&R_{02}\\
R_{10}&R_{11}&R_{12}\\
R_{20}&R_{21}&R_{22}
\end{bmatrix}.
```

This specification constructs rotation-6D from the first two columns of $R$:

```math
\rho_6(R)=
[R_{00},R_{10},R_{20},R_{01},R_{11},R_{21}]^{\mathsf T}.
```

The absolute end-effector state is therefore:

```math
\mathbf s^{ee}_t=
[x,y,z,\rho_6(R_t)^{\mathsf T}]^{\mathsf T}
\in\mathbb R^9.
```

To reconstruct a rotation matrix from two arbitrary 3D vectors $\mathbf a_1$ and $\mathbf a_2$, apply Gram-Schmidt orthogonalization:

```math
\mathbf b_1=
\frac{\mathbf a_1}{\|\mathbf a_1\|_2},
```

```math
\widetilde{\mathbf b}_2
=
\mathbf a_2-(\mathbf b_1^{\mathsf T}\mathbf a_2)\mathbf b_1,
```

```math
\mathbf b_2=
\frac{\widetilde{\mathbf b}_2}{\|\widetilde{\mathbf b}_2\|_2},
\qquad
\mathbf b_3=\mathbf b_1\times\mathbf b_2,
```

```math
R=[\mathbf b_1,\mathbf b_2,\mathbf b_3].
```

All data-generation, training, and deployment components must use the same first-two-columns convention.

### 4.3 Base gravity-direction and angular-velocity state

State slice `[41:47]` does not contain an absolute base pose. It contains the gravity direction and angular velocity in body frame $B$:

```math
\boxed{
\mathbf s_t^{base}=
[
g_x^B,\,
g_y^B,\,
g_z^B,\,
\hat\omega_x^B,\,
\hat\omega_y^B,\,
\hat\omega_z^B
]^{\mathsf T}
}.
```

Here:

- `g_B = [gx, gy, gz]` is the unit gravity direction expressed in body frame $B$;
- `omega_B = [wx, wy, wz]` is the raw three-axis angular velocity in body frame $B$;
- `omega_hat_B` is the normalized three-axis angular velocity.

#### 4.3.1 Gravity-direction computation

Let $R_{WB}$ map vectors from body frame $B$ to the world frame. Define the unit gravity direction in the world frame as:

```math
\mathbf g^W=[0,0,-1]^{\mathsf T}.
```

The gravity direction expressed in body frame $B$ is:

```math
\mathbf g^B=R_{WB}^{\mathsf T}\mathbf g^W.
```

To remove numerical error from the input quaternion or rotation matrix, normalize it once more before writing it into the state:

```math
\mathbf g^B
\leftarrow
\frac{\mathbf g^B}{\max(\|\mathbf g^B\|_2,\epsilon)},
```

with the recommended value $\epsilon=10^{-8}$. The resulting vector should satisfy:

```math
\|\mathbf g^B\|_2=1.
```

The gravity direction encodes roll and pitch relative to gravity. It contains neither world-frame position nor absolute yaw about the gravity axis.

#### 4.3.2 Angular-velocity normalization

The raw body-frame angular velocity is:

```math
\boldsymbol\omega^B=
[\omega_x^B,\omega_y^B,\omega_z^B]^{\mathsf T}.
```

Normalize it component-wise using angular-velocity state statistics:

```math
\widehat{\boldsymbol\omega}^B
=
\frac{\boldsymbol\omega^B-\mathbf o_{\omega}}
{\mathbf c_{\omega}}.
```

The default is `state_norm_type=minmax_q`:

```math
\mathbf o_{\omega}
=
\frac{Q_{0.01}(\boldsymbol\omega^B)+Q_{0.99}(\boldsymbol\omega^B)}{2},
```

```math
\mathbf c_{\omega}
=
\frac{Q_{0.99}(\boldsymbol\omega^B)-Q_{0.01}(\boldsymbol\omega^B)}{2}.
```

If the input already provides normalized angular velocity, write it directly into `[44:47]` and do not normalize it again.

#### 4.3.3 Slot mapping

```math
[g_x^B,g_y^B,g_z^B]
\longrightarrow[41:44],
```

```math
[\hat\omega_x^B,\hat\omega_y^B,\hat\omega_z^B]
\longrightarrow[44:47].
```

This state block contains neither base xyz position nor a rotation vector. The base action in action slice `[35:41]` is unchanged and still represents relative base-pose xyz plus rotation vector.

### 4.4 State filling and masking

Initialize:

```math
\mathbf s_t=\mathbf 0\in\mathbb R^{60},
\qquad
\mathbf m^s=\mathbf 0\in\{0,1\}^{60}.
```

Write every available state module into its fixed slice and set the corresponding mask to 1. Unavailable modules remain zero. If a state module is narrower than its destination slice, pad its tail with zeros and still mark the full slice as valid. If it is wider, retain only the leading components that fit. State dimensions should therefore be validated when a dataset is integrated.

---

## 5. Pose-Format Conversion

Every relative-pose computation must first convert the input pose to an SE(3) homogeneous transform:

```math
T=
\begin{bmatrix}
R&\mathbf p\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
```

Here $\mathbf p=[x,y,z]^{\mathsf T}$.

### 5.1 xyz + RPY

The input order is:

```math
[x,y,z,r,p,y].
```

The last $y$ denotes yaw; below it is written as $\psi$ to avoid ambiguity. Define:

```math
R_x(r)=
\begin{bmatrix}
1&0&0\\
0&\cos r&-\sin r\\
0&\sin r&\cos r
\end{bmatrix},
```

```math
R_y(p)=
\begin{bmatrix}
\cos p&0&\sin p\\
0&1&0\\
-\sin p&0&\cos p
\end{bmatrix},
```

```math
R_z(\psi)=
\begin{bmatrix}
\cos\psi&-\sin\psi&0\\
\sin\psi&\cos\psi&0\\
0&0&1
\end{bmatrix}.
```

Use the fixed-axis `xyz` Euler-angle convention:

```math
R=R_z(\psi)R_y(p)R_x(r).
```

All angles are in radians.

### 5.2 xyz + quaternion

The input order is:

```math
[x,y,z,q_x,q_y,q_z,q_w].
```

First normalize the quaternion:

```math
\bar{\mathbf q}=
\frac{\mathbf q}{\|\mathbf q\|_2}.
```

Using the normalized quaternion $(q_x,q_y,q_z,q_w)$, compute:

```math
R=
\begin{bmatrix}
1-2(q_y^2+q_z^2) & 2(q_xq_y-q_zq_w) & 2(q_xq_z+q_yq_w)\\
2(q_xq_y+q_zq_w) & 1-2(q_x^2+q_z^2) & 2(q_yq_z-q_xq_w)\\
2(q_xq_z-q_yq_w) & 2(q_yq_z+q_xq_w) & 1-2(q_x^2+q_y^2)
\end{bmatrix}.
```

### 5.3 xyz + rotation vector

The input order is:

```math
[x,y,z,\phi_x,\phi_y,\phi_z].
```

Let:

```math
\boldsymbol\phi=[\phi_x,\phi_y,\phi_z]^{\mathsf T},
\qquad
\theta=\|\boldsymbol\phi\|_2.
```

For $\theta>0$, let $\mathbf u=\boldsymbol\phi/\theta$ and use Rodrigues' formula:

```math
R=
I+\sin\theta[\mathbf u]_{\times}
+(1-\cos\theta)[\mathbf u]_{\times}^2.
```

For very small $\theta$, use a numerically stable small-angle expansion or a robust SO(3) implementation.

---

## 6. Relative-Action Definition and Computation

### 6.1 Relative end-effector and base pose

Let the current state pose be:

```math
T_t=
\begin{bmatrix}
R_t&\mathbf p_t\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix},
```

and let the `k`-th future action target be:

```math
T_{t+k}=
\begin{bmatrix}
R_{t+k}&\mathbf p_{t+k}\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
```

The inverse current pose is:

```math
T_t^{-1}=
\begin{bmatrix}
R_t^{\mathsf T}&-R_t^{\mathsf T}\mathbf p_t\\
\mathbf 0^{\mathsf T}&1
\end{bmatrix}.
```

Define the relative action as:

```math
T^{rel}_{t,k}=T_t^{-1}T_{t+k}.
```

Expanding the product gives:

```math
R^{rel}_{t,k}=R_t^{\mathsf T}R_{t+k},
```

```math
\mathbf p^{rel}_{t,k}
=R_t^{\mathsf T}(\mathbf p_{t+k}-\mathbf p_t).
```

Thus, relative translation is expressed in the current end-effector or base frame, rather than being a direct world-frame position difference.

Convert the relative rotation matrix to a rotation vector:

```math
\boldsymbol\phi^{rel}_{t,k}
=\mathrm{Log}(R^{rel}_{t,k})^{\vee}.
```

The final relative action is:

```math
\mathbf a^{rel}_{t,k}
=
\begin{bmatrix}
\mathbf p^{rel}_{t,k}\\
\boldsymbol\phi^{rel}_{t,k}
\end{bmatrix}
\in\mathbb R^6.
```

### 6.2 Relative-action pseudocode

```text
function relative_pose(current_pose, future_poses, pose_format):
    T_current = pose_to_SE3(current_pose, pose_format)
    T_current_inverse = inverse_SE3(T_current)

    result = []
    for future_pose in future_poses:
        T_future = pose_to_SE3(future_pose, pose_format)
        T_relative = T_current_inverse @ T_future

        relative_xyz = T_relative[0:3, 3]
        relative_rotvec = SO3_log(T_relative[0:3, 0:3])
        result.append(concat(relative_xyz, relative_rotvec))

    return result
```

---

## 7. Global Relative-Action Statistics

Only global statistics are used. No separate statistics are maintained for individual positions inside an action chunk.

### 7.1 Global-statistics computation

For every valid current state in the dataset, compute the complete future relative-action chunk as defined in Section 6:

```math
\mathbf A^{(n)}
=
\begin{bmatrix}
\mathbf a^{rel}_{n,0}\\
\mathbf a^{rel}_{n,1}\\
\vdots\\
\mathbf a^{rel}_{n,H-1}
\end{bmatrix}
\in\mathbb R^{H\times d},
```

where $n=1,\ldots,N$ indexes samples, $H$ is the chunk length, and $d=6$ for an end-effector or base relative pose.

Flatten the sample and horizon dimensions into one matrix:

```math
X_{rel}
=
\mathrm{reshape}
\left(
\{\mathbf A^{(n)}\}_{n=1}^{N},
(NH,d)
\right).
```

Every relative-action point in every chunk is treated as an independent statistical sample, regardless of its horizon position.

For component $j$, the global mean is:

```math
\mu^{global}_j
=
\frac{1}{NH}
\sum_{n=1}^{N}
\sum_{k=0}^{H-1}
A^{(n)}_{k,j}.
```

The global population standard deviation is:

```math
\sigma^{global}_j
=
\sqrt{
\frac{1}{NH}
\sum_{n=1}^{N}
\sum_{k=0}^{H-1}
\left(A^{(n)}_{k,j}-\mu^{global}_j\right)^2
}.
```

The global extrema are:

```math
x^{global}_{min,j}=\min_{n,k}A^{(n)}_{k,j},
```

```math
x^{global}_{max,j}=\max_{n,k}A^{(n)}_{k,j}.
```

The global 1st and 99th percentiles are:

```math
Q^{global}_{0.01,j}
=
Q_{0.01}\left(\{A^{(n)}_{k,j}\}_{n,k}\right),
```

```math
Q^{global}_{0.99,j}
=
Q_{0.99}\left(\{A^{(n)}_{k,j}\}_{n,k}\right).
```

Every global statistic has shape `(d,)`.

Relative poses use Z-score normalization by default:

```math
\widehat{\mathbf a}^{rel}_{n,k}
=
\frac{\mathbf a^{rel}_{n,k}-\boldsymbol\mu^{global}}
{\boldsymbol\sigma^{global}}.
```

The same global mean and standard deviation are applied to every horizon position in the chunk.

### 7.2 Statistics format

Each relative-action module only needs the following global statistics:

```json
{
  "relative_action_key": {
    "global_max": [0.0],
    "global_min": [0.0],
    "global_q01": [0.0],
    "global_q99": [0.0],
    "global_mean": [0.0],
    "global_std": [0.0]
  }
}
```

For a six-dimensional relative pose, every array has length 6 and uses the following order:

```math
[\Delta x,\Delta y,\Delta z,
\phi_x,\phi_y,\phi_z].
```

### 7.3 Statistics-generation pseudocode

```text
function collect_global_relative_statistics(samples):
    all_relative_steps = []

    for sample in samples:
        relative_chunk = relative_pose(
            sample.current_pose,
            sample.future_pose_chunk,
            sample.pose_format,
        )

        for relative_action in relative_chunk:
            all_relative_steps.append(relative_action)

    X = stack(all_relative_steps)  # shape: (number_of_all_steps, action_dim)

    return {
        "global_max": max(X, axis=0),
        "global_min": min(X, axis=0),
        "global_q01": quantile(X, 0.01, axis=0),
        "global_q99": quantile(X, 0.99, axis=0),
        "global_mean": mean(X, axis=0),
        "global_std": population_std(X, axis=0)
    }
```

---

## 8. Ordinary State and Action Statistics

For fields that do not undergo online relative-pose conversion, stack all low-dimensional records into:

```math
X=
\begin{bmatrix}
\mathbf x_1^{\mathsf T}\\
\mathbf x_2^{\mathsf T}\\
\vdots\\
\mathbf x_M^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{M\times d}.
```

Compute component-wise:

```math
\boldsymbol\mu
=
\frac{1}{M}\sum_{i=1}^{M}\mathbf x_i,
```

```math
\boldsymbol\sigma
=
\sqrt{
\frac{1}{M}
\sum_{i=1}^{M}
(\mathbf x_i-\boldsymbol\mu)^2
},
```

and:

```math
\mathbf x_{min},\quad
\mathbf x_{max},\quad
Q_{0.01}(X),\quad
Q_{0.99}(X).
```

The standard deviation is the population standard deviation, with divisor $M$.

Every field in the ordinary statistics file must contain at least:

```json
{
  "feature_key": {
    "mean": [0.0],
    "std": [1.0],
    "min": [-1.0],
    "max": [1.0],
    "q01": [-0.9],
    "q99": [0.9]
  }
}
```

An optional `count` field may be stored for future sample-count-weighted merging.

---

## 9. Normalization Definitions

### 9.1 Unified affine form

All continuous quantities use:

```math
\widehat{\mathbf x}
=
\frac{\mathbf x-\mathbf o}{\mathbf c},
```

where division is component-wise. Denormalization is:

```math
\mathbf x
=
\widehat{\mathbf x}\odot\mathbf c+
\mathbf o.
```

Normalized values are not clipped. Values outside the statistical range may therefore be smaller than $-1$ or greater than $1$.

Every scale component is protected as follows:

```math
c_j=
\begin{cases}
1,&c_j<10^{-6},\\
c_j,&\text{otherwise}.
\end{cases}
```

If a module has no statistics, use identity normalization:

```math
\mathbf o=\mathbf 0,
\qquad
\mathbf c=\mathbf 1.
```

### 9.2 Quantile min-max normalization: `minmax_q`

Statistics are selected in this priority order:

1. `global_q01/global_q99`;
2. `q01/q99`;
3. `min/max`;
4. identity normalization.

Let the lower and upper bounds be $\mathbf l$ and $\mathbf h$. Then:

```math
\mathbf o=
\frac{\mathbf l+\mathbf h}{2},
```

```math
\mathbf c=
\frac{\mathbf h-\mathbf l}{2}.
```

Thus:

```math
\mathbf l\mapsto-1,
\qquad
\mathbf h\mapsto1.
```

### 9.3 Z-score normalization: `zscore`

Statistics are selected in this priority order:

1. `global_mean/global_std`;
2. `mean/std`;
3. identity normalization.

The parameters are:

```math
\mathbf o=\boldsymbol\mu,
\qquad
\mathbf c=\boldsymbol\sigma.
```

### 9.4 Extreme-value min-max normalization: `minmax`

Statistics are selected in this priority order:

1. `global_min/global_max`;
2. `min/max`;
3. identity normalization.

The parameters are:

```math
\mathbf o=
\frac{\mathbf x_{min}+\mathbf x_{max}}{2},
```

```math
\mathbf c=
\frac{\mathbf x_{max}-\mathbf x_{min}}{2}.
```

---

## 10. Normalization Used by Each Action Module

The default action-normalization configuration is:

```yaml
norm_type: minmax_q
rel_norm_type: zscore
gripper_norm_type: minmax_q
```

| Action module | Unified slice | Statistics source | Normalization type | Preferred statistics |
|---|---|---|---|---|
| Single-arm relative end-effector pose | `[0:6]` | Relative-action statistics | `rel_norm_type` | `global_mean/global_std` |
| Left relative end-effector pose | `[0:6]` | Relative-action statistics | `rel_norm_type` | `global_mean/global_std` |
| Right relative end-effector pose | `[13:19]` | Relative-action statistics | `rel_norm_type` | `global_mean/global_std` |
| Relative base pose | `[35:41]` | Relative-action statistics | `rel_norm_type` | `global_mean/global_std` |
| Single/left/right gripper | `[6:7]`, `[19:20]` | Ordinary action statistics | `gripper_norm_type` | `q01/q99` |
| Left/right dexterous hand | `[7:13]`, `[20:26]` | Ordinary action statistics | `gripper_norm_type` | `q01/q99` |
| Waist | `[26:29]` | Ordinary action statistics | `norm_type` | `q01/q99` |
| Torso | `[29:32]` | Ordinary action statistics | `norm_type` | `q01/q99` |
| Base `vx, vy` | `[32:34]` | Ordinary statistics of the complete `base_command` | `norm_type` | `q01/q99` of the mapped source components |
| Base `omega_z` | `[34:35]` | Ordinary statistics of the complete `base_command` | `norm_type` | Component mapped by `vyaw` or `vw` |
| Height command | `[41:42]` | Ordinary statistics of the complete `base_command` | `norm_type` | Component mapped by `height` |
| Left/right leg | `[42:54]` | Ordinary action statistics | `norm_type` | `q01/q99` |

Important details:

1. dexterous-hand actions use the gripper normalization type rather than the ordinary action normalization type;
2. relative end-effector and base poses must use relative-action statistics, not absolute-pose statistics;
3. statistics are generated for the complete source `base_command`, after which the relevant offset and scale components are selected according to the dimension map;
4. modules absent from the unified action space use offset 0 and scale 1.

### 10.1 `base_command` statistics mapping

Assume the source base command is:

```math
\mathbf b=[b_0,b_1,\ldots,b_{d-1}]^{\mathsf T},
```

with the dimension map:

```yaml
base_command_dims:
  vx: 0
  vy: 1
  vw: 2
  height: 3
```

The unified action normalization parameters are mapped as:

```math
o^a_{32}=o^b_0,
\qquad
c^a_{32}=c^b_0,
```

```math
o^a_{33}=o^b_1,
\qquad
c^a_{33}=c^b_1,
```

```math
o^a_{34}=o^b_2,
\qquad
c^a_{34}=c^b_2,
```

```math
o^a_{41}=o^b_3,
\qquad
c^a_{41}=c^b_3.
```

A `vyaw` field is handled in the same way as `vw`.

---

## 11. Normalization Used by Each State Module

The default state normalization type is:

```yaml
state_norm_type: minmax_q
```

| State module | Unified slice | Normalized? | Statistics source | Notes |
|---|---|---:|---|---|
| Left end-effector xyz | `[0:3]` | Yes | First 3 components of left absolute end-effector state statistics | Uses `state_norm_type` |
| Left end-effector rotation-6D | `[3:9]` | No | None | Preserve the geometric representation |
| Left gripper | `[9:10]` | Yes | Left-gripper state statistics | Uses `state_norm_type` |
| Left dexterous hand | `[10:16]` | Yes | Left-hand state statistics | Uses `state_norm_type` |
| Right end-effector xyz | `[16:19]` | Yes | First 3 components of right absolute end-effector state statistics | Uses `state_norm_type` |
| Right end-effector rotation-6D | `[19:25]` | No | None | Preserve the geometric representation |
| Right gripper | `[25:26]` | Yes | Right-gripper state statistics | Uses `state_norm_type` |
| Right dexterous hand | `[26:32]` | Yes | Right-hand state statistics | Uses `state_norm_type` |
| Waist | `[32:35]` | Yes | Waist-state statistics | Use at most the first 3 components |
| Torso | `[35:38]` | Yes | Torso-state statistics | Use at most the first 3 components |
| Body gravity direction | `[41:44]` | Unit-length normalization only | No dataset statistics | Must have unit norm |
| Body angular velocity | `[44:47]` | Yes | Three-axis angular-velocity state statistics | Store normalized angular velocity |
| Left leg | `[48:54]` | Yes | Left-leg state statistics | Use at most the first 6 components |
| Right leg | `[54:60]` | Yes | Right-leg state statistics | Use at most the first 6 components |
| Unused scalar base angular-velocity/height slots | `[40:41]`, `[47:48]` | No | None | Value 0 and mask 0 |

### 11.1 Why only xyz is normalized for end-effector pose state

An input absolute pose may use:

- xyz + RPY: 6 dimensions;
- xyz + quaternion: 7 dimensions;
- xyz + rotation vector: 6 dimensions.

The unified end-effector state uses xyz + rotation-6D, which has 9 dimensions. Applying raw rotation statistics directly to rotation-6D would mismatch both dimensionality and geometry. Therefore only the first three position components reuse the raw pose statistics:

```math
\widehat{\mathbf p}_t
=
\frac{\mathbf p_t-\mathbf o_{xyz}}
{\mathbf c_{xyz}},
```

while rotation-6D remains unchanged.

The base state does not use this pose-normalization rule. Slice `[41:47]` is constructed from gravity direction and normalized angular velocity as specified in Section 4.3.

---

## 12. Merging Statistics Across Tasks of the Same Robot Embodiment

### 12.1 Merge scope

Statistics are merged only within the **same robot embodiment**. Treat the statistics dictionary of each task as one independent group, then merge all tasks for that embodiment in one operation.

Let the task set be:

```math
\mathcal T=\{\tau_1,\tau_2,\ldots,\tau_M\}.
```

Tasks may participate in the same merge only if they have:

- the same robot embodiment;
- the same action and state module definitions;
- the same coordinate frames, axis directions, and units;
- the same physical meaning for identically named fields;
- the same action dimensions and component ordering.

Consequently:

- every task has the same weight;
- every task contributes one statistics group;
- the number of raw samples in a task does not change its merge weight;
- statistics from different robot embodiments must not be merged;
- different embodiments must maintain separate normalizers even if they use the same unified slots.

### 12.2 Mean merging and the equal-weight assumption

Equal group weighting does not assume that different tasks contain similar motions. It assumes that each task is selected with approximately equal probability during training. The equivalent sampling procedure is:

1. select one task uniformly;
2. sample a trajectory or training example from that task.

If task $i$ has action distribution $P_i(\mathbf x)$, the target training mixture is:

```math
P_{train}(\mathbf x)
=
\frac{1}{M}
\sum_{i=1}^{M}P_i(\mathbf x).
```

Under task-balanced sampling, equal task weighting matches the distribution seen by the model even when tasks contain different numbers of raw trajectories.

If training instead samples uniformly from all raw frames, equal group weighting only approximates sample-level pooling when tasks contribute similar numbers of valid trajectories, trajectory lengths, or action points. The relevant assumption concerns effective sample counts or task-selection probabilities, not similarity of motion content.

Let group $i$ have mean $\boldsymbol\mu_i$. Under the task-balanced assumption, the merged mean is:

```math
\boldsymbol\mu
=
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\mu_i.
```

### 12.3 Standard-deviation merging

Let group $i$ have population standard deviation $\boldsymbol\sigma_i$. The merged variance is:

```math
\boldsymbol\sigma^2
=
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\sigma_i^2
+
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\mu_i^2
-
\boldsymbol\mu^2.
```

Equivalently:

```math
\boldsymbol\sigma^2
=
\underbrace{
\frac{1}{M}\sum_{i=1}^{M}\boldsymbol\sigma_i^2
}_{\text{mean within-group variance}}
+
\underbrace{
\frac{1}{M}\sum_{i=1}^{M}
(\boldsymbol\mu_i-\boldsymbol\mu)^2
}_{\text{between-group variance of means}}.
```

Finally compute component-wise:

```math
\boldsymbol\sigma
=
\sqrt{\max(\boldsymbol\sigma^2,0)}.
```

If a standard deviation has no corresponding mean, fall back to:

```math
\boldsymbol\sigma
=
\sqrt{
\frac{1}{M}
\sum_{i=1}^{M}\boldsymbol\sigma_i^2
}.
```

### 12.4 Extrema merging

```math
\mathbf x_{min}
=
\min_i\mathbf x_{min}^{(i)},
```

```math
\mathbf x_{max}
=
\max_i\mathbf x_{max}^{(i)}.
```

All comparisons are component-wise.

### 12.5 Quantile merging

Use a conservative envelope:

```math
Q_{0.01}^{merged}
=
\min_i Q_{0.01}^{(i)},
```

```math
Q_{0.99}^{merged}
=
\max_i Q_{0.99}^{(i)}.
```

For other quantile fields, use the component-wise median across groups.

These values are not the true quantiles of the mixed raw-sample distribution. Exact mixed quantiles cannot generally be recovered from a few per-group quantiles. Exact computation requires rescanning raw samples or storing a mergeable distribution summary such as a histogram or t-digest.

### 12.6 Count merging

If ordinary statistics include `count`, merge it as:

```math
N=\sum_{i=1}^{M}n_i.
```

The default merged mean and standard deviation remain task-balanced and do not use `count` as a weight. `count` is therefore metadata only under the default scheme.

### 12.7 Shape compatibility

Only arrays with matching shapes may be merged for the same field and statistic:

- relative-action statistics must have the same action dimension;
- ordinary statistics are merged only when their shapes match the first valid statistic;
- if fewer than two ordinary-statistics arrays are shape-compatible, retain the first valid statistic;
- image statistics do not participate in low-dimensional action/state statistics merging.

### 12.8 Equal group weights versus sample weights

Equal group weighting is appropriate when task balance is the target. If the statistics must represent the empirical distribution of every raw sample, use sample-count weighting.

Let group $i$ contain $n_i$ samples:

```math
N=\sum_{i=1}^{M}n_i,
```

```math
\boldsymbol\mu_{weighted}
=
\frac{1}{N}
\sum_{i=1}^{M}n_i\boldsymbol\mu_i,
```

```math
\boldsymbol\sigma^2_{weighted}
=
\frac{1}{N}
\sum_{i=1}^{M}
 n_i\left(
 \boldsymbol\sigma_i^2+
 \boldsymbol\mu_i^2
 \right)
-
\boldsymbol\mu_{weighted}^2.
```

This weighted formula describes an alternative statistics policy and is not used by the default merge procedure.

---

## 13. Left/Right End-Effector Statistics Merging

### 13.1 Merge method

When shared left/right statistics are enabled, treat the left and right end effectors as two equally weighted groups.

Let the left and right mean vectors be:

```math
\boldsymbol{\mu}_L,
\qquad
\boldsymbol{\mu}_R.
```

The merged mean is:

```math
\boldsymbol{\mu}_{LR}
=
\frac{\boldsymbol{\mu}_L+\boldsymbol{\mu}_R}{2}.
```

The merged variance is:

```math
\boldsymbol{\sigma}_{LR}^2
=
\frac{\boldsymbol{\sigma}_L^2+
      \boldsymbol{\sigma}_R^2}{2}
+
\frac{\boldsymbol{\mu}_L^2+
      \boldsymbol{\mu}_R^2}{2}
-
\boldsymbol{\mu}_{LR}^2.
```

The merged quantile range is:

```math
Q_{0.01}^{LR}
=
\min(Q_{0.01}^{L},Q_{0.01}^{R}),
```

```math
Q_{0.99}^{LR}
=
\max(Q_{0.99}^{L},Q_{0.99}^{R}).
```

After merging, the left and right end effectors must use identical offsets and scales.

### 13.2 Merge order

Use the following order:

1. compute left and right statistics for every task;
2. merge all tasks belonging to the same robot embodiment;
3. merge the already task-merged left and right statistics;
4. assign the same final statistics to both left and right keys.

```math
\text{per-task statistics}
\longrightarrow
\text{same-embodiment cross-task merge}
\longrightarrow
\text{left/right merge}
\longrightarrow
\text{normalizer}.
```

### 13.3 Coordinate-consistency requirements

Left/right statistics merging does not automatically mirror axes, exchange components, or flip signs. Before merging, ensure that:

- left and right relative-translation xyz axes have the same meaning;
- left and right rotation vectors use the same right-handed convention;
- matching component indices represent the same physical direction;
- units are identical.

If right-arm coordinates must be mirrored into a left-arm canonical frame, first define a fixed transform:

```math
\widetilde{\mathbf a}_R=M\mathbf a_R,
```

and compute right-arm statistics from $\widetilde{\mathbf a}_R$. Matrix $M$ must include every required axis permutation and sign change for both translation and rotation-vector components.

### 13.4 Purpose

A shared left/right normalizer:

1. removes scale differences caused by unequal amounts of left- and right-arm data;
2. keeps the numerical space consistent under left/right swapping or symmetry augmentation;
3. allows a shared action head to learn a unified bimanual motion prior.

---

## 14. Gripper Binarization

Gripper binarization is applied after normalization. Let the normalized gripper sequence be:

```math
\widehat{g}_0,\widehat{g}_1,\ldots,
\widehat{g}_{H-1}.
```

Definite states are:

```math
\widehat{g}_k>0.9
\quad\Longrightarrow\quad
b_k=1,
```

```math
\widehat{g}_k<-0.9
\quad\Longrightarrow\quad
b_k=0.
```

Values in `[-0.9, 0.9]` are intermediate states. Scan backward from the end of the sequence and fill each intermediate value with the nearest definite future state.

Initialize the tail class as:

```math
b_{H-1}^{init}
=
\begin{cases}
1,&\widehat{g}_{H-1}>0,\\
0,&\widehat{g}_{H-1}\le 0.
\end{cases}
```

Pseudocode:

```text
carry = 1 if normalized_gripper[-1] > 0 else 0

for k from H-1 down to 0:
    if normalized_gripper[k] > 0.9:
        carry = 1
    else if normalized_gripper[k] < -0.9:
        carry = 0

    binary_gripper[k] = carry
```

The output belongs to `{0, 1}` and is no longer an ordinary symmetric continuous normalized value.

---

## 15. Action-Sequence Resampling

### 15.1 Linear resampling

For a source chunk containing $N_s$ points, define source timestamps as:

```math
t_i^{src}=
\frac{i}{f_s},
\qquad i=0,\ldots,N_s-1.
```

Define target timestamps as:

```math
t_j^{tgt}=
\frac{j}{f_t},
\qquad j=0,\ldots,f_t-1.
```

Interpolate every action component independently with piecewise-linear interpolation. If a target timestamp falls outside the source range, use the nearest endpoint value rather than linear extrapolation.

The default one-second linear mode reads $N_s=f_s$ points over:

```math
\left[0,\frac{f_s-1}{f_s}\right].
```

### 15.2 B-spline resampling

Smooth resampling reads two seconds of context, including the $t=0$ anchor:

```math
N_s=2f_s+1.
```

The source timestamps are:

```math
t_i^{src}=\frac{i}{f_s},
\qquad i=0,\ldots,2f_s.
```

The target remains the first one-second window:

```math
t_j^{tgt}=\frac{j}{f_t},
\qquad j=0,\ldots,f_t-1.
```

Use a cubic uniform B-spline with degree 3 and:

```math
K=\max\left(4,\left\lfloor\frac{N_s}{2}\right\rfloor+1\right)
```

basis functions. Let $B_{src}$ and $B_{tgt}$ be the basis matrices evaluated at source and target timestamps, and let:

```math
\lambda=10^{-9}.
```

Precompute the resampling matrix:

```math
W=
B_{tgt}
\left(B_{src}^{\mathsf T}B_{src}+\lambda I\right)^{-1}
B_{src}^{\mathsf T}.
```

For every action component:

```math
A_{tgt}=WA_{src}.
```

Bitwise reproduction requires identical uniform B-spline knot definitions, boundary conditions, basis ordering, and floating-point precision.

### 15.3 Execution order

The action-processing order is fixed:

```math
\text{relative-pose conversion}
\longrightarrow
\text{normalization}
\longrightarrow
\text{optional gripper binarization}
\longrightarrow
\text{resampling}
\longrightarrow
\text{unified-slot mapping}.
```

Only the current state frame is used; states are not temporally resampled.

---

## 16. Reproduction Checklist

A conforming implementation must satisfy all of the following:

1. actions have exactly 54 dimensions and states have exactly 60 dimensions;
2. end-effector state uses absolute xyz + rotation-6D;
3. rotation-6D uses the first two columns of the rotation matrix;
4. end-effector and base-pose actions use $T_t^{-1}T_{t+k}$;
5. relative poses are represented as xyz + rotation vector;
6. relative-action statistics are global statistics obtained by flattening the sample and horizon dimensions;
7. relative-action normalizers use `global_*` statistics;
8. relative poses use Z-score normalization by default;
9. other actions and states use q01/q99 min-max normalization by default;
10. only xyz is normalized for end-effector pose state; rotation-6D remains unchanged;
11. base-state slice `[41:47]` contains body-frame gravity direction and normalized three-axis angular velocity;
12. dexterous-hand actions use the gripper normalization type;
13. statistics are merged only across different tasks of the same robot embodiment, with equal task weights;
14. merged q01/q99 use an envelope rather than true mixed-distribution quantiles;
15. merged left/right statistics must produce identical offsets and scales for both arms;
16. left/right coordinate semantics must be aligned before statistics are merged;
17. unavailable slots retain value 0, offset 0, and scale 1, and are excluded by their masks.
