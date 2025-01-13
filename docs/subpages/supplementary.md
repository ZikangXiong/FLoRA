# Action Predicates

Note: All T values specified are initial values that will be updated during the learning process to better match real-world driving behaviors.

## Acceleration Group
### Accelerate Normal
$$p_{acc_n}(s,a) = tanh(k(T - \min_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : 2.0 [m/s²], range [1.0, 3.0]

This predicate identifies normal acceleration behavior. It looks for sustained positive acceleration that doesn't exceed comfortable levels for passengers.

### Accelerate Hard
$$p_{acc_h}(s,a) = tanh(k(T - \min_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : 4.0 [m/s²], range [3.0, 5.0]

Similar to normal acceleration but with higher threshold for more aggressive maneuvers like highway merging or passing.

### Decelerate Normal
$$p_{dec_n}(s,a) = tanh(k(T + \max_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : 2.0 [m/s²], range [1.0, 3.0]

This predicate identifies comfortable deceleration behavior. The maximum acceleration (minimum deceleration) is used to ensure smooth, controlled slowing.

### Decelerate Hard
$$p_{dec_h}(s,a) = tanh(k(T + \max_{t \in [t_0,t_1]} a_{long}(t)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : 4.0 [m/s²], range [3.0, 5.0]

Captures more intense braking maneuvers while still maintaining vehicle control.

### Cruise
$$p_{cruise}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} |a_{long}(t)|))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $T$ : 0.5 [m/s²], range [0.3, 1.0]

Identifies steady-state driving with minimal acceleration changes. Uses absolute acceleration to ensure both acceleration and deceleration are minimized.

## Turning Group
### Turn Left
$$p_{left}(s,a) = tanh(k(T - \min_{t \in [t_0,t_1]} \dot{\psi}(t)))$$
- $\dot{\psi}(t)$ : yaw rate [rad/s]
- $T$ : 0.3 [rad/s], range [0.1, 0.5]

Captures left turns with adaptive threshold suitable for both gentle and sharp turns.

### Turn Right
$$p_{right}(s,a) = tanh(k(T + \max_{t \in [t_0,t_1]} \dot{\psi}(t)))$$
- $\dot{\psi}(t)$ : yaw rate [rad/s]
- $T$ : 0.3 [rad/s], range [0.1, 0.5]

Mirror of left turn for right direction. Handles both gentle and sharp right turns.

## Longitudinal Control Group
### Start
$$p_{start}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} v(t)))$$
- $v(t)$ : velocity [m/s]
- $T$ : 3.5 [m/s], range [1.5, 6.0]

Characterizes acceleration from stop with adaptive threshold for both slow and fast starts.

### Stop
$$p_{stop}(s,a) = tanh(k(T - \max(\min_{t \in [t_0,t_1]} v(t), -\min_{t \in [t_0,t_1]} a_{long}(t)/a_{max})))$$
- $v(t)$ : velocity [m/s], $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $a_{max}$ : 6.0 [m/s²]
- $T$ : 0.95, range [0.9, 1.0]

Evaluates stopping behavior considering both final velocity and deceleration rate.

## Lateral Control Group
### Change Lane Left
$$p_{lcl}(s,a) = tanh(k(T - \min(\frac{|d_{lat}(t_1)|}{w}, |\psi(t_1)|, \frac{|\dot{y}(t_1)|}{v_{lat}})))$$
- $d_{lat}(t)$ : lateral distance to lane center [m], $\psi(t)$ : heading [rad]
- $\dot{y}(t)$ : lateral velocity [m/s]
- $w$ : 3.5 [m], $v_{lat}$ : 0.7 [m/s]
- $T$ : 0.95, range [0.9, 1.0]

Characterizes left lane changes with adaptive thresholds for both slow and fast maneuvers.

### Change Lane Right
$$p_{lcr}(s,a) = tanh(k(T - \min(\frac{|d_{lat}(t_1)|}{w}, |\psi(t_1)|, \frac{|\dot{y}(t_1)|}{v_{lat}})))$$
- $d_{lat}(t)$ : lateral distance to lane center [m], $\psi(t)$ : heading [rad]
- $\dot{y}(t)$ : lateral velocity [m/s]
- $w$ : 3.5 [m], $v_{lat}$ : 0.7 [m/s]
- $T$ : 0.95, range [0.9, 1.0]

Mirror of left lane change for right direction.

### Keep Lane
$$p_{kl}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} |y(t) - y(t_0)|))$$
- $y(t)$ : lateral position [m]
- $T$ : 0.2 [m], range [0.05, 0.4]

Evaluates lane-keeping behavior with adaptive threshold allowing both strict and relaxed tracking.

## Lane Position Group
### Center in Lane
$$p_{center}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} |d_{lat}(t)|))$$
- $d_{lat}(t)$ : lateral distance to lane center [m]
- $T$ : 0.2 [m], range [0.1, 0.3]

Evaluates how well the vehicle maintains center position within the lane.

### Smooth Steering
$$p_{smooth}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} |\ddot{\psi}(t)|))$$
- $\ddot{\psi}(t)$ : yaw acceleration [rad/s²]
- $T$ : 0.3 [rad/s²], range [0.2, 0.4]

Measures smoothness of steering inputs by evaluating rate of yaw rate change.

## Vehicle Following Group
### Follow Distance
$$p_{follow}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} |\frac{d_{long}(t)}{v(t)} - t_{desired}|))$$
- $d_{long}(t)$ : longitudinal distance to lead vehicle [m]
- $v(t)$ : velocity [m/s]
- $t_{desired}$ : 2.0 [s]
- $T$ : 0.5 [s], range [0.3, 0.7]

Evaluates maintenance of desired time headway to leading vehicle.

### Smooth Following
$$p_{smooth\_follow}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} |\frac{a_{long}(t)}{a_{lead}(t)} - 1|))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $a_{lead}(t)$ : lead vehicle acceleration [m/s²]
- $T$ : 0.3, range [0.2, 0.4]

Measures how smoothly the vehicle matches lead vehicle acceleration changes.

## Dual-Purpose Group
All the following predicates are used as both action and condition predicates.

### In Drivable Area
$$p_{in\_drivable}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} d_{drivable}(t)))$$
- $d_{drivable}(t)$ : minimum distance to drivable area boundary [m]
- $T$ : 0.3 [m], range [0.2, 0.5]

Ensures vehicle maintains safe distance from boundaries of drivable area.

### Comfortable
$$p_{comfortable}(s,a) = tanh(k(T - \max_{t \in [t_0,t_1]} \max(|\frac{a_{long}(t)}{a_{max}}|, |\frac{a_{lat}(t)}{a_{lat\_max}}|, |\frac{j(t)}{j_{max}}|)))$$
- $a_{long}(t)$ : longitudinal acceleration [m/s²]
- $a_{lat}(t)$ : lateral acceleration [m/s²]
- $j(t)$ : jerk [m/s³]
- $a_{max}$ : 2.0 [m/s²], $a_{lat\_max}$ : 1.5 [m/s²], $j_{max}$ : 2.5 [m/s³]
- $T$ : 0.8, range [0.7, 0.9]

Evaluates overall motion comfort considering accelerations and jerk.

### Overtaking
$$p_{overtaking}(s,a) = tanh(k(T - \min(\frac{v_{rel}(t)}{v_{min}}, \frac{d_{lat}(t_1)}{w}, \frac{d_{long}(t_0)}{d_{min}})))$$
- $v_{rel}(t)$ : relative velocity to overtaken vehicle [m/s]
- $d_{lat}(t)$ : lateral distance to overtaken vehicle [m]
- $d_{long}(t)$ : longitudinal distance to overtaken vehicle [m]
- $v_{min}$ : 2.0 [m/s], $w$ : 3.5 [m], $d_{min}$ : 10.0 [m]
- $T$ : 0.9, range [0.85, 0.95]

Characterizes overtaking maneuvers considering relative speed, lateral and longitudinal spacing.

### SafeTTC
$$p_{safe\_ttc}(s,a) = tanh(k(T - \min_{i \in vehicles} \min_{t \in [t_0,t_1]} \frac{d_i(t)}{|v_{rel,i}(t)|+\epsilon}))$$
- $d_i(t)$ : distance to vehicle i [m]
- $v_{rel,i}(t)$ : relative velocity to vehicle i [m/s]
- $\epsilon$ : small constant to avoid division by zero (0.001)
- $T$ : 3.0 [s], range [2.0, 4.0]

Evaluates safety by measuring minimum time-to-collision with surrounding vehicles.