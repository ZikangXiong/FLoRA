---
heroText: FLoRA
tagline: Supplementary Material
---

## Action Predicates

### Accelerate Normal
$$p_{acc_n} = tanh(k(T - \min_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : initialized with 2.0 [m/s²], range [1.0, 3.0]
- $t_0, t_1$ : start and end time of the plan

This predicate identifies normal acceleration behavior. It looks for sustained positive acceleration that doesn't exceed comfortable levels for passengers.

### Accelerate Hard
$$p_{acc_h} = tanh(k(T - \min_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : initialized with 4.0 [m/s²], range [3.0, 5.0]
- $t_0, t_1$ : start and end time of the plan

Similar to normal acceleration but with higher threshold for more aggressive maneuvers like highway merging or passing.

### Decelerate Normal
$$p_{dec_n} = tanh(k(T + \max_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : initialized with 2.0 [m/s²], range [1.0, 3.0]
- $t_0, t_1$ : start and end time of the plan

This predicate identifies comfortable deceleration behavior. The maximum acceleration (minimum deceleration) is used to ensure smooth, controlled slowing.

### Decelerate Hard
$$p_{dec_h} = tanh(k(T + \max_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : initialized with 4.0 [m/s²], range [3.0, 5.0]
- $t_0, t_1$ : start and end time of the plan

Captures more intense braking maneuvers while still maintaining vehicle control.

### Fast Cruise
$$p_{cruise} = tanh(k(T - \max_{t \in [t_0,t_1]} |a_{long}(t)|))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : initialized with 0.5 [m/s²], range [0.3, 1.0]
- $t_0, t_1$ : start and end time of the plan

Identifies steady-state driving with minimal acceleration changes. Uses absolute acceleration to ensure both acceleration and deceleration are minimized.

### Slow Cruise
$$p_{cruise} = tanh(k(T - \max_{t \in [t_0,t_1]} |a_{long}(t)|))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : initialized with 0.5 [m/s²], range [0.3, 1.0]
- $t_0, t_1$ : start and end time of the plan

Similar to cruise fast but with lower threshold for slower driving speeds.

### Turn Left
$$p_{left} = tanh(k(T - \min_{t \in [t_0,t_1]} \dot{\psi}(t)))$$
- $\dot{\psi}(t)$ : yaw rate [rad/s]
- $T$ : initialized with 0.3 [rad/s], range [0.1, 0.5]

Captures left turns with adaptive threshold suitable for both gentle and sharp turns.

### Turn Right
$$p_{right} = tanh(k(T + \max_{t \in [t_0,t_1]} \dot{\psi}(t)))$$
- $\dot{\psi}(t)$ : yaw rate [rad/s]
- $T$ : initialized with 0.3 [rad/s], range [0.1, 0.5]

Mirror of left turn for right direction. Handles both gentle and sharp right turns.

### Start
$$p_{start} = tanh(k(T - (\max_{t \in [t_0,t_1]} v(t)) - \min_{t \in [t_0,t_1]} v(t)))$$
- $v(t)$ : velocity [m/s]
- $T$ : initialized with 3.5 [m/s], range [1.5, 6.0]

Characterizes vehicle start behavior by evaluating the range of initial velocities.

### Stop
$$p_{stop} = tanh(k(T - \max_{t \in [t_0,t_1]} v(t)))$$
- $v(t)$ : velocity [m/s], $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : initialized with 0.95, range [0.9, 1.0]

Evaluates stopping behavior considering the maximum velocity reached during the plan time horizon.

### Change Lane Left
$$p_{lcl} = tanh(k(T - \min(\frac{|d_{lat}(t_1)|}{w}, |\psi(t_1)|, \frac{|\dot{y}(t_1)|}{v_{lat}})))$$
- $d_{lat}(t)$ : lateral distance to lane center [m], $\psi(t)$ : heading [rad]
- $\dot{y}(t)$ : lateral velocity [m/s]
- $w$ : 3.5 [m], $v_{lat}$ : 0.7 [m/s]
- $T$ : initialized with 0.95, range [0.9, 1.0]

Characterizes left lane changes with adaptive thresholds for both slow and fast maneuvers.

### Change Lane Right
$$p_{lcr} = tanh(k(T - \min(\frac{|d_{lat}(t_1)|}{w}, |\psi(t_1)|, \frac{|\dot{y}(t_1)|}{v_{lat}})))$$
- $d_{lat}(t)$ : lateral distance to lane center [m], $\psi(t)$ : heading [rad]
- $\dot{y}(t)$ : lateral velocity [m/s]
- $w$ : 3.5 [m], $v_{lat}$ : 0.7 [m/s]
- $T$ : initialized with 0.95, range [0.9, 1.0]

Mirror of left lane change for right direction.

### Keep Lane
$$p_{kl} = tanh(k(T - \max_{t \in [t_0,t_1]} |y(t) - y(t_0)|))$$
- $y(t)$ : lateral position [m]
- $T$ : initialized with 0.2 [m], range [0.05, 0.4]

Evaluates lane-keeping behavior with adaptive threshold allowing both strict and relaxed tracking.

### Center in Lane
$$p_{center} = tanh(k(T - \max_{t \in [t_0,t_1]} |d_{lat}(t)|))$$
- $d_{lat}(t)$ : lateral distance to lane center [m]
- $T$ : initialized with 0.2 [m], range [0.1, 0.3]

Evaluates how well the vehicle maintains center position within the lane.

### Smooth Steering
$$p_{smooth} = tanh(k(T - \max_{t \in [t_0,t_1]} |\ddot{\psi}(t)|))$$
- $\ddot{\psi}(t)$ : yaw acceleration [rad/s²]
- $T$ : initialized with 0.3 [rad/s²], range [0.2, 0.4]

Measures smoothness of steering inputs by evaluating rate of yaw rate change.

### Follow Distance
$$p_{follow} = tanh(k(T - \max_{t \in [t_0,t_1]} |\frac{d_{long}(t)}{v(t)} - t_{desired}|))$$
- $d_{long}(t)$ : longitudinal distance to lead vehicle [m]
- $v(t)$ : velocity [m/s]
- $t_{desired}$ : 2.0 [s]
- $T$ : initialized with 0.5 [s], range [0.3, 0.7]

Evaluates maintenance of desired time headway to leading vehicle.

### Smooth Following
$$p_{smooth\_follow} = tanh(k(T - \max_{t \in [t_0,t_1]} |\frac{a_{long}(t)}{a_{lead}(t)} - 1|))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $a_{lead}(t)$ : lead vehicle acceleration [m/s²]
- $T$ : initialized with 0.3, range [0.2, 0.4]

Measures how smoothly the vehicle matches lead vehicle acceleration changes.

<!-- ## Condition Predicates

### Can Change Lane Left

### Can Change Lane Right

### Can Curise

### Can Decelerate Hard

### Can Decelerate Normal

### Can Start

### Can Stop

### Can Turn Left

### Can Turn Right

### Traffic Light Green

### Traffic Light Red

### VRU Crossing

### VRU In Path

### Other Vehicle In Lane

### Other Vehicle Cut In -->

## Dual-Purpose Predicates
All the following predicates are used as both action and condition predicates.

### In Drivable Area
$$p_{in\_drivable} = tanh(k(T - \max_{t \in [t_0,t_1]} d_{drivable}(t)))$$
- $d_{drivable}(t)$ : minimum distance to drivable area boundary [m]
- $T$ : initialized with 0.3 [m], range [0.2, 0.5]

Ensures vehicle maintains safe distance from boundaries of drivable area.

### Comfortable
$$Comfortable_{\boldsymbol{\theta}} := \tanh(\min_{\substack{i \in {a_{f}, a_{b}, a_{l}, a_{r}}}} {\theta_i - i(\tau)})$$

- $i(\tau)$ : maximum acceleration in plan $\tau$ for each direction (forward, backward, left, right).  
- $\boldsymbol{\theta} = [\theta_{a_{f}}, \theta_{a_{b}}, \theta_{a_{l}}, \theta_{a_{r}}]$ : threshold parameters for each direction  
    - $\theta_{a_{f}}$ : forward acceleration threshold, initial value is 1.0 [$m/s^2$], range [0.0, 5.0]  
    - $\theta_{a_{b}}$ : backward acceleration threshold, initial value is 1.0 [$m/s^2$], range [0.0, 5.0]  
    - $\theta_{a_{l}}$ : leftward acceleration threshold, initial value is 0.5 [$m/s^2$], range [0.0, 3.0]  
    - $\theta_{a_{r}}$ : rightward acceleration threshold, initial value is 0.5 [$m/s^2$], range [0.0, 3.0]

Evaluates motion comfort by checking if accelerations in all directions (forward, backward, left, right) remain below learned thresholds. The predicate is positive only when all accelerations are below their respective thresholds. All threshold parameters are differentiable and can be learned from demonstration data.

### Overtaking
$$p_{overtaking} = tanh(k(T - \min(\frac{v_{rel}(t)}{v_{min}}, \frac{d_{lat}(t_1)}{w}, \frac{d_{long}(t_0)}{d_{min}})))$$
- $v_{rel}(t)$ : relative velocity to overtaken vehicle [m/s]
- $d_{lat}(t)$ : lateral distance to overtaken vehicle [m]
- $d_{long}(t)$ : longitudinal distance to overtaken vehicle [m]
- $v_{min}$ : 2.0 [m/s], $w$ : 3.5 [m], $d_{min}$ : 10.0 [m]
- $T$ : initialized with 0.9, range [0.85, 0.95]

Characterizes overtaking maneuvers considering relative speed, lateral and longitudinal spacing.

### SafeTTC
$$p_{safe\_ttc} = tanh(k(T - \min_{i \in vehicles} \min_{t \in [t_0,t_1]} \frac{d_i(t)}{|v_{rel,i}(t)|+\epsilon}))$$
- $d_i(t)$ : distance to vehicle i [m]
- $v_{rel,i}(t)$ : relative velocity to vehicle i [m/s]
- $\epsilon$ : small constant to avoid division by zero (0.001)
- $T$ : initialized with 3.0 [s], range [2.0, 4.0]

Evaluates safety by measuring minimum time-to-collision with surrounding vehicles.

