# G1 Whole-Body Teleoperation

This repository provides two methods to control the **G1** humanoid robot (29 DOFs) using [AMO](https://github.com/OpenTeleVision/AMO/tree/main) in MuJoCo:

1. **Upper body inverse kinematics** using [Mink](https://github.com/kevinzakka/mink).
2. **VR teleoperation** using [XLeVR](https://github.com/Vector-Wangel/XLeRobot).

> **Note:** For VR teleoperation, a **Meta Quest 3 or Quest 3S** is required.

---

## Installation

```bash
conda create -n gbot python=3.10
conda activate gbot
pip install -r requirements.txt
```

---

## Usage

### 1. Mocap Control for G1 End-Effector

From the project root directory:

```bash
python 01_g1_29_2f85_mocap_amo.py
```

* You can now drag the mocap cube to control the movement of the G1's end-effector.

---

### 2. VR Control for Full-Body Teleoperation

From the project root directory:

```bash
python 02_g1_29_2f85_VR_amo.py
```

* Open your VR headset browser and navigate to the HTTPS address shown in the terminal (e.g., `https://<your-ip>:8443`).
* Use your VR controllers to move the G1.

  * **Grip button:** Hold to enable VR teleoperation.
  * **Trigger:** Hold to grasp.

#### VR Controller Mapping

| Input                | Forward      | Backward      | Left          | Right          |
| -------------------- | ------------ | ------------- | ------------- | -------------- |
| **Left Thumbstick**  | Height +     | Height −      | Yaw turn left | Yaw turn right |
| **Right Thumbstick** | Walk forward | Walk backward | Walk left     | Walk right     |



---

## Acknowledgements

* [**AMO**](https://github.com/OpenTeleVision/AMO) — Adaptive Motion Optimization for Hyper-Dexterous Humanoid Whole-Body Control.
* [**Mink**](https://github.com/kevinzakka/mink) — A library for differential inverse kinematics in Python, based on the MuJoCo physics engine.
* [**XLeRobot**](https://github.com/Vector-Wangel/XLeRobot) — Bringing Embodied AI to every family around the world.

