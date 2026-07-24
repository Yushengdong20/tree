"""grasp_object 腰部和底盘采样工具。"""

import math

import numpy as np
from kuavo_humanoid_sdk.common.three_link_torso_ik import ThreeLinkTorsoIk

from tree.utils.geometry import make_xz_pitch_transform


class GraspSampling:
    """生成当前腰部、腰部 IK 和底盘横移采样。"""

    def _current_torso_sample(self, fixed_knee_from_waist):
        _, pitch, _ = self._tf.transformations.euler_from_matrix(fixed_knee_from_waist)
        return {
            "label": (
                "当前腰部"
                f"(x={fixed_knee_from_waist[0, 3]:.3f}, z={fixed_knee_from_waist[2, 3]:.3f}, "
                f"pitch={pitch:.3f})"
            ),
            "enabled": False,
            "fixed_knee_from_waist": fixed_knee_from_waist,
            "knee_from_waist": fixed_knee_from_waist,
            "torso_pose": None,
            "x": float(fixed_knee_from_waist[0, 3]),
            "z": float(fixed_knee_from_waist[2, 3]),
            "pitch": float(pitch),
            "motion_cost": 0.0,
        }

    def _torso_ik_samples(self, current_fixed_knee_from_waist, current_torso_pose):
        current_x = float(current_fixed_knee_from_waist[0, 3])
        current_z = float(current_fixed_knee_from_waist[2, 3])
        samples = []
        pitch_samples = self._torso_pitch_sample_values()
        z_min, z_max, z_source = self._torso_sample_z_bounds()
        for x in self._sample_axis_values(self.torso_sample_x_min_m, self.torso_sample_x_max_m):
            for z in self._sample_axis_values(z_min, z_max):
                for pitch in pitch_samples:
                    if abs(pitch) > self.torso_pitch_abs_max_rad:
                        continue

                    ik_pose = ThreeLinkTorsoIk().solve_with_fixed_torso_pitch(
                        x,
                        z,
                        pitch,
                    )
                    if ik_pose is None:
                        continue

                    waist_x = float(ik_pose["end_world_x"])
                    waist_z = float(ik_pose["end_world_z"])
                    checked_pitch = float(ik_pose["torso_pitch"])
                    torso_pose = list(current_torso_pose)
                    torso_pose[0] = waist_x + self.knee_origin_x
                    torso_pose[2] = waist_z + self.knee_origin_z
                    torso_pose[4] = checked_pitch

                    # 关键步骤：只有通过三连杆 x/z/pitch 联合校验的姿态才进入抓取可达性搜索。
                    score = (
                        (waist_x - current_x) ** 2
                        + (waist_z - current_z) ** 2
                        + 0.25 * checked_pitch * checked_pitch
                    )
                    sample = {
                        "label": (
                            f"腰部采样(x={waist_x:.3f}, z={waist_z:.3f}, "
                            f"pitch={checked_pitch:.3f}, z_source={z_source})"
                        ),
                        "enabled": True,
                        "fixed_knee_from_waist": make_xz_pitch_transform(waist_x, waist_z, checked_pitch),
                        "torso_pose": torso_pose,
                        "x": waist_x,
                        "z": waist_z,
                        "pitch": checked_pitch,
                    }
                    sample["knee_from_waist"] = sample["fixed_knee_from_waist"]
                    sample["motion_cost"] = float(score)
                    samples.append((score, sample))

        samples.sort(key=lambda item: item[0])
        return [sample for _, sample in samples]

    def _torso_sample_z_bounds(self):
        """读取 torso 采样 z 范围，server 动态 blackboard 值优先于 JSON 配置。"""
        z_min = self.torso_sample_z_min_m
        z_max = self.torso_sample_z_max_m
        source = "config"

        if self.torso_sample_z_min_key and self.blackboard.exists(self.torso_sample_z_min_key):
            z_min = float(self.blackboard.get(self.torso_sample_z_min_key))
            source = "blackboard"
        if self.torso_sample_z_max_key and self.blackboard.exists(self.torso_sample_z_max_key):
            z_max = float(self.blackboard.get(self.torso_sample_z_max_key))
            source = "blackboard"

        if z_max < z_min:
            raise ValueError(
                "torso 动态采样高度范围非法: "
                f"z_min={z_min:.4f}, z_max={z_max:.4f}, source={source}"
            )
        return z_min, z_max, source

    def _chassis_lateral_torso_samples(self, torso_samples, current_base_from_map, poses_frame):
        """组合底盘横向位移和腰部采样，优先检查横移距离更小的组合。"""
        offsets = self._chassis_lateral_offsets()
        combined = []
        for offset_index, lateral_offset in enumerate(offsets):
            base_from_map = self._base_from_map_after_lateral_offset(
                current_base_from_map,
                lateral_offset,
            )
            for torso_index, torso_sample in enumerate(torso_samples):
                sample = dict(torso_sample)
                sample["base_from_map"] = base_from_map
                sample["chassis_lateral_offset_m"] = float(lateral_offset)
                sample["chassis_lateral_enabled"] = abs(float(lateral_offset)) > 1e-6
                sample["poses_frame"] = poses_frame
                sample["label"] = (
                    f"底盘横移(y={lateral_offset:.3f}) + {torso_sample['label']}"
                )
                # 关键步骤：使用字典序严格优先最小底盘横移，再比较 torso x/z 位移平方和。
                combined.append(
                    (
                        abs(float(lateral_offset)),
                        float(torso_sample.get("motion_cost", 0.0)),
                        offset_index,
                        torso_index,
                        sample,
                    )
                )
        combined.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return [sample for _, _, _, _, sample in combined]

    def _chassis_lateral_offsets(self):
        """生成底盘横向采样值，顺序为 0、-step、+step、-2step、+2step。"""
        values = self._sample_axis_values(
            self.chassis_lateral_sample_min_m,
            self.chassis_lateral_sample_max_m,
            step=self.chassis_lateral_sample_step_m,
        )
        values.sort(key=lambda value: (abs(float(value)), float(value)))
        return values

    @staticmethod
    def _base_from_map_after_lateral_offset(current_base_from_map, lateral_offset_m):
        """计算底盘横移后的 sample_base_link <- map 变换。"""
        base_from_sample_base = np.eye(4)
        base_from_sample_base[1, 3] = -float(lateral_offset_m)
        return base_from_sample_base @ current_base_from_map

    def _sample_axis_values(self, lower, upper, step=None):
        values = []
        value = float(lower)
        sample_step = float(self.torso_sample_step_m if step is None else step)
        while value <= upper + 1e-9:
            values.append(round(value, 10))
            value += sample_step
        if values and values[-1] < upper - 1e-9:
            values.append(float(upper))
        return values

    def _torso_pitch_sample_values(self):
        """按配置生成腰部 pitch 采样值，单位 rad。"""
        pitch_values_deg = self._sample_axis_values(
            self.torso_pitch_sample_min_deg,
            self.torso_pitch_sample_max_deg,
            step=self.torso_pitch_sample_step_deg,
        )
        # 关键步骤：配置使用角度便于调试，内部矩阵和 torso_pose 均使用弧度。
        return [math.radians(value) for value in pitch_values_deg]

    def _sample_batches(self, samples, batch_size):
        """按顺序把腰部采样拆成批次，批内并行检查，批间保留原始优先级。"""
        for start_index in range(0, len(samples), max(1, int(batch_size))):
            yield samples[start_index : start_index + max(1, int(batch_size))]
