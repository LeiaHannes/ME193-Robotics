"""
Q-learning to drive a LEGO Education Double Motor in a straight line,
continuously, with no episodes. The robot drives forward the whole time
and learns while it goes. Press Ctrl+C to stop (Q table is saved).

Outline:
 1. Define hyperparameters, states, actions, reward
 2. Read the state (yaw error -> bin)
 3. Choose an action (epsilon-greedy)
 4. Execute the action (speed differential, motors never stop)
 5. Observe new yaw -> s', r
 6. Update Q
 7. Decay epsilon (every step, slowly, with a floor)
 8. Repeat from 3 using s'
"""

import os
import random
import time

import numpy as np
import legoeducation as le

# ============================================================
# 1. DEFINE THINGS
# ============================================================

# --- Connection (match your Connection Card) ---
CARD_COLOR = le.LEGO_COLOR_YELLOW
CARD_SERIAL = '0994'

# --- Hyperparameters ---
ALPHA = 0.2          # learning rate
GAMMA = 0.9          # discount factor
EPSILON = 1.0        # starting exploration rate
EPS_DECAY = 0.995    # per step: ~0.1 after ~460 steps, floor after ~600
EPS_MIN = 0.05       # exploration floor

# --- Reward ranges (degrees) ---
CENTER_RANGE = 2     # |error| <= 2  -> +1
CLOSE_RANGE = 6      # |error| <= 6  ->  0, beyond -> -1

# --- States: yaw error bins, aligned with reward ranges ---
BIN_EDGES = [-12, -6, -2, 2, 6, 12]      # 7 bins; bin 3 = center
N_STATES = len(BIN_EDGES) + 1

# --- Actions: speed differential (left = BASE + d, right = BASE - d) ---
BASE_SPEED = 50
ACTIONS = [-10, -5, 0, 5, 10]
N_ACTIONS = len(ACTIONS)

# --- Motor directions for "forward" on this build ---
LEFT_FWD = le.MOTOR_MOVE_DIRECTION_COUNTERCLOCKWISE
RIGHT_FWD = le.MOTOR_MOVE_DIRECTION_CLOCKWISE
STEP_TIME = 0.3      # seconds each action is held before observing

# --- Logging / saving ---
PRINT_EVERY = 10     # steps
SAVE_EVERY = 50      # steps
Q_FILE = 'q_table.npy'

if os.path.exists(Q_FILE):
    Q = np.load(Q_FILE)
    print(f"Loaded existing Q table from {Q_FILE}")
else:
    Q = np.zeros((N_STATES, N_ACTIONS))


def reward(yaw_error):
    e = abs(yaw_error)
    if e <= CENTER_RANGE:
        return 1
    elif e <= CLOSE_RANGE:
        return 0
    return -1


# ============================================================
# 2. READ THE STATE
# ============================================================

def reset_heading(dm):
    dm.imu_reset_yaw_axis(0)
    time.sleep(0.3)  # let the IMU settle


def get_yaw_error(dm):
    """Positive = drifted left, negative = drifted right."""
    yaw = dm.imu_device.yaw
    return (yaw + 180) % 360 - 180  # wrap to [-180, 180)


def get_state(yaw_error):
    return int(np.digitize(yaw_error, BIN_EDGES))


# ============================================================
# 3. CHOOSE AN ACTION
# ============================================================

def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(N_ACTIONS)            # explore
    row = Q[state]
    best = np.flatnonzero(row == row.max())           # break ties randomly
    return int(random.choice(best))                   # exploit


# ============================================================
# 4. EXECUTE THE ACTION
# ============================================================

_current_action = None

def apply_action(dm, a):
    """Set wheel speeds. Only sends a command when the action changes,
    so the motors keep running smoothly instead of restarting each step."""
    global _current_action
    if a != _current_action:
        d = ACTIONS[a]
        dm.motor_run(direction=LEFT_FWD, motor=le.MOTOR_LEFT, speed=BASE_SPEED + d)
        dm.motor_run(direction=RIGHT_FWD, motor=le.MOTOR_RIGHT, speed=BASE_SPEED - d)
        _current_action = a
    time.sleep(STEP_TIME)


def stop_motors(dm):
    try:
        dm.motor_stop(motor=le.MOTOR_LEFT)
        dm.motor_stop(motor=le.MOTOR_RIGHT)
    except TypeError:
        dm.motor_stop()


# ============================================================
# 5. OBSERVE THE RESULT
# ============================================================

def observe(dm):
    err = get_yaw_error(dm)
    return err, get_state(err), reward(err)


# ============================================================
# 6. UPDATE Q
# ============================================================

def update_q(s, a, r, s_next):
    q_hat = np.max(Q[s_next])                         # best Q in new row
    Q[s, a] = Q[s, a] + ALPHA * (r + GAMMA * q_hat - Q[s, a])


# ============================================================
# CONTINUOUS LOOP (steps 3-8)
# ============================================================

def print_policy():
    labels = ['<-12', '-12..-6', '-6..-2', 'center', '2..6', '6..12', '>12']
    print("\nLearned policy (yaw error bin -> differential):")
    for s in range(N_STATES):
        if np.all(Q[s] == 0):
            choice = '(unvisited)'
        else:
            choice = f"{ACTIONS[int(np.argmax(Q[s]))]:+d}"
        print(f"  {labels[s]:>8}: {choice}")
    print("\nQ table:\n", np.round(Q, 2))


def main():
    global EPSILON

    dm = le.DoubleMotor()
    dm.connect(card_color=CARD_COLOR, card_serial=CARD_SERIAL)
    if not dm.connected:
        print('Error connecting to Double Motor.')
        exit(1)

    step = 0
    recent_r = []

    try:
        reset_heading(dm)                             # this heading = "straight" for the whole run
        err, s, _ = observe(dm)                       # 2

        while True:
            a = choose_action(s, EPSILON)             # 3
            apply_action(dm, a)                       # 4
            err, s_next, r = observe(dm)              # 5
            update_q(s, a, r, s_next)                 # 6
            EPSILON = max(EPS_MIN, EPSILON * EPS_DECAY)  # 7
            s = s_next                                # 8

            step += 1
            recent_r.append(r)
            if step % PRINT_EVERY == 0:
                avg = sum(recent_r) / len(recent_r)
                print(f"step {step:5d}  eps={EPSILON:.3f}  err={err:+6.1f}  "
                      f"avg reward (last {len(recent_r)}): {avg:+.2f}")
                recent_r = []
            if step % SAVE_EVERY == 0:
                np.save(Q_FILE, Q)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        stop_motors(dm)
        np.save(Q_FILE, Q)
        print_policy()
        dm.disconnect()


if __name__ == '__main__':
    main()