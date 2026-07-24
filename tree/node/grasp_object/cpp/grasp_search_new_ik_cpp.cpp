#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kOffsetTheta4 =
    std::atan2(0.02, 0.2837) + std::atan2(0.02, 0.126 + 0.1075);
constexpr double kL34 = 0.284404166;  // sqrt(0.02^2 + 0.2837^2)
constexpr double kL45 = 0.234355478;  // sqrt(0.02^2 + (0.126 + 0.1075)^2)

const Eigen::Vector3d kL1LinkInBase(0.0, 0.2527, 0.242);
const Eigen::Vector3d kL1RightLinkInBase(0.0, -0.2527, 0.242);
constexpr int kNewIkQ7SampleCount = 21;
constexpr double kNewIkFkPositionTolerance = 1e-4;
constexpr double kNewIkFkAngleTolerance = 1e-3;

struct JointLimits {
  std::array<double, 7> lower;
  std::array<double, 7> upper;
};

struct CheckResult {
  bool valid = false;
  double pos_error = std::numeric_limits<double>::infinity();
  double angle_error = std::numeric_limits<double>::infinity();
};

struct SearchResult {
  bool found = false;
  int sample_index = -1;
  int arm_priority = 0;  // 0 表示右手，1 表示左手。
  int grasp_index = -1;
  bool rotate_z_180 = false;
  Eigen::Matrix4d grasp_pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d grasp_target_pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix4d waist_from_base = Eigen::Matrix4d::Identity();
  std::array<double, 6> grasp_target{};
  Eigen::Matrix4d pregrasp_target_pose = Eigen::Matrix4d::Identity();
  std::array<double, 6> pregrasp_target{};
  double pregrasp_offset_m = 0.0;
  double pos_error = std::numeric_limits<double>::infinity();
  double angle_error = std::numeric_limits<double>::infinity();
};

struct SampleRange {
  npy_intp begin = 0;
  npy_intp end = 0;
};

struct SearchContext {
  PyArrayObject* sample_fixed = nullptr;
  PyArrayObject* sample_base = nullptr;
  PyArrayObject* grasp_poses = nullptr;
  npy_intp sample_count = 0;
  npy_intp grasp_count = 0;
  bool use_map_poses = false;
  Eigen::Matrix4d fixed_knee_from_base = Eigen::Matrix4d::Identity();
  double grasp_offset_m = 0.0;
  std::string model_type;
  double pos_threshold = 0.01;
  double angle_threshold = 0.05;
  double pregrasp_offset_min_m = 0.12;
  double pregrasp_offset_max_m = 0.18;
  int pregrasp_offset_samples = 5;
  int arm_begin = 0;
  int arm_end = 2;
};

double Clip(double value, double lower, double upper) {
  return std::min(std::max(value, lower), upper);
}

double ClipToPi(double angle) {
  return std::fmod(angle + kPi, 2.0 * kPi) - kPi;
}

Eigen::Matrix3d RotX(double theta) {
  Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  rot(1, 1) = c;
  rot(1, 2) = -s;
  rot(2, 1) = s;
  rot(2, 2) = c;
  return rot;
}

Eigen::Matrix3d RotY(double theta) {
  Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  rot(0, 0) = c;
  rot(0, 2) = s;
  rot(2, 0) = -s;
  rot(2, 2) = c;
  return rot;
}

Eigen::Matrix3d RotZ(double theta) {
  Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
  const double c = std::cos(theta);
  const double s = std::sin(theta);
  rot(0, 0) = c;
  rot(0, 1) = -s;
  rot(1, 0) = s;
  rot(1, 1) = c;
  return rot;
}

Eigen::Matrix4d Transform(const Eigen::Matrix3d& rot,
                          const Eigen::Vector3d& pos = Eigen::Vector3d::Zero()) {
  Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
  transform.block<3, 3>(0, 0) = rot;
  transform.block<3, 1>(0, 3) = pos;
  return transform;
}

Eigen::Matrix4d MatrixAt(PyArrayObject* array, npy_intp index) {
  Eigen::Matrix4d matrix;
  const double* data = static_cast<const double*>(PyArray_GETPTR3(array, index, 0, 0));
  const npy_intp stride0 = PyArray_STRIDE(array, 1) / static_cast<npy_intp>(sizeof(double));
  const npy_intp stride1 = PyArray_STRIDE(array, 2) / static_cast<npy_intp>(sizeof(double));
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      matrix(r, c) = data[r * stride0 + c * stride1];
    }
  }
  return matrix;
}

Eigen::Matrix4d Matrix2d(PyArrayObject* array) {
  Eigen::Matrix4d matrix;
  const double* data = static_cast<const double*>(PyArray_GETPTR2(array, 0, 0));
  const npy_intp stride0 = PyArray_STRIDE(array, 0) / static_cast<npy_intp>(sizeof(double));
  const npy_intp stride1 = PyArray_STRIDE(array, 1) / static_cast<npy_intp>(sizeof(double));
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      matrix(r, c) = data[r * stride0 + c * stride1];
    }
  }
  return matrix;
}

PyObject* MatrixToNumpy(const Eigen::Matrix4d& matrix) {
  npy_intp dims[2] = {4, 4};
  PyObject* obj = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
  if (obj == nullptr) {
    return nullptr;
  }
  auto* array = reinterpret_cast<PyArrayObject*>(obj);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      *static_cast<double*>(PyArray_GETPTR2(array, r, c)) = matrix(r, c);
    }
  }
  return obj;
}

PyObject* Vector6ToList(const std::array<double, 6>& values) {
  PyObject* list = PyList_New(6);
  if (list == nullptr) {
    return nullptr;
  }
  for (int i = 0; i < 6; ++i) {
    PyList_SET_ITEM(list, i, PyFloat_FromDouble(values[i]));
  }
  return list;
}

std::array<double, 6> MatrixToXyzYpr(const Eigen::Matrix4d& pose) {
  const Eigen::Matrix3d rot = pose.block<3, 3>(0, 0);
  double roll = 0.0;
  double pitch = 0.0;
  double yaw = 0.0;
  // 关键步骤：复刻 tf.transformations.euler_from_matrix 默认 sxyz 顺序。
  if (std::abs(rot(2, 0)) < 1.0 - 1e-12) {
    pitch = std::asin(-rot(2, 0));
    roll = std::atan2(rot(2, 1), rot(2, 2));
    yaw = std::atan2(rot(1, 0), rot(0, 0));
  } else {
    pitch = rot(2, 0) <= -1.0 ? kPi / 2.0 : -kPi / 2.0;
    roll = std::atan2(-rot(0, 1), rot(1, 1));
    yaw = 0.0;
  }
  constexpr double rad_to_deg = 180.0 / kPi;
  return {pose(0, 3), pose(1, 3), pose(2, 3),
          yaw * rad_to_deg, pitch * rad_to_deg, roll * rad_to_deg};
}

JointLimits GetJointLimits(const std::string& model_type) {
  (void)model_type;
  JointLimits limits;
  // 关键步骤：new_ik 使用当前 Python NewIKAnalytical 的限位，不再按任务类型改 joint7。
  limits.lower = {-kPi, -0.349065850398866, -1.5707963267949,
                  -2.61799387799149, -1.5707963267949,
                  -1.30899693899575, -0.698131700797732};
  limits.upper = {1.5707963267949, 2.9670597283903604, 1.5707963267949,
                  0.0, 1.5707963267949, 0.698131700797732,
                  0.698131700797732};
  return limits;
}

Eigen::Matrix4d FkLeftArm(const std::array<double, 7>& theta) {
  const Eigen::Matrix4d x01 = Transform(RotY(theta[0]));
  const Eigen::Matrix4d x12 = Transform(RotX(theta[1]));
  const Eigen::Matrix4d x23 = Transform(RotZ(theta[2]));
  const Eigen::Matrix4d x34 = Transform(RotY(theta[3]), Eigen::Vector3d(0.02, 0.0, -0.2837));
  const Eigen::Matrix4d x45 = Transform(RotZ(theta[4]), Eigen::Vector3d(-0.02, 0.0, -(0.126 + 0.1075)));
  const Eigen::Matrix4d x56 = Transform(RotX(theta[5]));
  const Eigen::Matrix4d x67 = Transform(RotY(theta[6]), Eigen::Vector3d(0.0, 0.0, -0.021));
  return x01 * x12 * x23 * x34 * x45 * x56 * x67;
}

Eigen::Matrix4d FkRightArm(const std::array<double, 7>& theta) {
  std::array<double, 7> left_theta = {
      theta[0], -theta[1], -theta[2], theta[3], -theta[4], -theta[5], theta[6]};
  const Eigen::Matrix4d left_fk = FkLeftArm(left_theta);
  const Eigen::Matrix3d reflection_y = (Eigen::Vector3d(1.0, -1.0, 1.0)).asDiagonal();
  Eigen::Matrix4d right_fk = Eigen::Matrix4d::Identity();
  right_fk.block<3, 1>(0, 3) = reflection_y * left_fk.block<3, 1>(0, 3);
  right_fk.block<3, 3>(0, 0) = reflection_y * left_fk.block<3, 3>(0, 0) * reflection_y;
  return right_fk;
}

bool NormalizeVector(const Eigen::Vector3d& input, Eigen::Vector3d* output) {
  const double norm = input.norm();
  if (!std::isfinite(norm) || norm < 1e-10 || output == nullptr) {
    return false;
  }
  *output = input / norm;
  return true;
}

bool IsFiniteJointArray(const std::array<double, 7>& joint_angles) {
  for (double angle : joint_angles) {
    if (!std::isfinite(angle)) {
      return false;
    }
  }
  return true;
}

bool IsJointInLimits(const std::array<double, 7>& joint_angles,
                     const JointLimits& limits) {
  for (int index = 0; index < 7; ++index) {
    if (joint_angles[index] < limits.lower[index] ||
        joint_angles[index] > limits.upper[index]) {
      return false;
    }
  }
  return true;
}

bool IsNewIkFkValid(const std::array<double, 7>& joint_angles,
                    const Eigen::Matrix4d& shoulder_target,
                    const JointLimits& limits,
                    double* pos_error,
                    double* angle_error) {
  if (!IsFiniteJointArray(joint_angles) || !IsJointInLimits(joint_angles, limits)) {
    return false;
  }
  const Eigen::Matrix4d fk = FkLeftArm(joint_angles);
  const double local_pos_error =
      (fk.block<3, 1>(0, 3) - shoulder_target.block<3, 1>(0, 3)).norm();
  const Eigen::Matrix3d delta =
      fk.block<3, 3>(0, 0).transpose() * shoulder_target.block<3, 3>(0, 0);
  const double local_angle_error =
      std::acos(Clip((delta.trace() - 1.0) * 0.5, -1.0, 1.0));
  if (pos_error != nullptr) {
    *pos_error = local_pos_error;
  }
  if (angle_error != nullptr) {
    *angle_error = local_angle_error;
  }
  return std::isfinite(local_pos_error) && std::isfinite(local_angle_error) &&
         local_pos_error <= kNewIkFkPositionTolerance &&
         local_angle_error <= kNewIkFkAngleTolerance;
}

std::array<std::array<double, 7>, 4> NewIkLeftOnce(
    const std::array<double, 7>& reference_joint_angles,
    double q7,
    const Eigen::Matrix4d& shoulder_target,
    const JointLimits& limits) {
  std::array<std::array<double, 7>, 4> q_all;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  for (std::array<double, 7>& row : q_all) {
    row.fill(nan);
  }
  if (q7 <= limits.lower[6] || q7 >= limits.upper[6]) {
    return q_all;
  }
  for (std::array<double, 7>& row : q_all) {
    row[6] = q7;
  }

  const Eigen::Vector3d y_ee = shoulder_target.block<3, 1>(0, 1);
  const Eigen::Matrix3d r_ee = shoulder_target.block<3, 3>(0, 0);
  const Eigen::Vector3d p7 = shoulder_target.block<3, 1>(0, 3);
  const Eigen::Vector3d z_7_6(-std::sin(q7), 0.0, std::cos(q7));
  Eigen::Vector3d z6;
  if (!NormalizeVector(r_ee * z_7_6, &z6)) {
    return q_all;
  }
  const Eigen::Vector3d p6 = p7 + 0.021 * z6;
  const Eigen::Vector3d p2 = Eigen::Vector3d::Zero();
  const Eigen::Vector3d v26 = p6 - p2;
  const double ll26 = v26.squaredNorm();
  const double l26 = std::sqrt(ll26);
  if (kL34 + kL45 < l26 || kL34 + l26 < kL45 || l26 + kL45 < kL34 ||
      l26 < 1e-10) {
    return q_all;
  }

  const double theta246 = std::acos(Clip((kL34 * kL34 + kL45 * kL45 - ll26) /
                                            (2.0 * kL34 * kL45),
                                        -1.0, 1.0));
  const double theta_h46 = std::atan((0.126 + 0.1075) / 0.02);
  const double theta_342 = std::atan(0.2837 / 0.02);
  const double theta_46h = std::atan(0.02 / (0.126 + 0.1075));
  const double q4 = theta246 + theta_h46 + theta_342 - 2.0 * kPi;
  if (q4 <= limits.lower[3] || q4 >= limits.upper[3]) {
    return q_all;
  }
  for (std::array<double, 7>& row : q_all) {
    row[3] = q4;
  }

  const double theta462 = std::acos(Clip((ll26 + kL45 * kL45 - kL34 * kL34) /
                                            (2.0 * l26 * kL45),
                                        -1.0, 1.0));
  const double theta26h = theta_46h + theta462;
  const double d26 = -l26 * std::cos(theta26h);

  Eigen::Vector3d x6_axis;
  if (!NormalizeVector(y_ee.cross(z6), &x6_axis)) {
    return q_all;
  }
  Eigen::Matrix3d r6;
  r6.col(0) = x6_axis;
  r6.col(1) = y_ee;
  r6.col(2) = z6;
  const Eigen::Vector3d v6_62 = r6.transpose() * v26;
  const double denom = std::sqrt(v6_62.z() * v6_62.z() + v6_62.y() * v6_62.y());
  if (denom < 1e-10) {
    return q_all;
  }
  const double phi6 = std::atan2(v6_62.z(), v6_62.y());
  const double theta6 = std::asin(Clip(d26 / denom, -1.0, 1.0));
  std::array<double, 2> q6 = {kPi - theta6 - phi6, theta6 - phi6};
  for (double& candidate_q6 : q6) {
    if (candidate_q6 <= limits.lower[5]) {
      candidate_q6 += 2.0 * kPi;
    } else if (candidate_q6 >= limits.upper[5]) {
      candidate_q6 -= 2.0 * kPi;
    }
  }

  for (int index = 0; index < 2; ++index) {
    if (q6[index] >= limits.lower[5] && q6[index] <= limits.upper[5]) {
      q_all[2 * index][5] = q6[index];
      q_all[2 * index + 1][5] = q6[index];
    }
  }

  const double theta_p26 = 1.5 * kPi - theta462 - theta246 - theta_342;
  const double theta_p = kPi - theta_p26 - theta26h;
  const double sin_theta_p = std::sin(theta_p);
  if (std::abs(sin_theta_p) < 1e-10) {
    return q_all;
  }
  const double lp6 = l26 * std::sin(theta_p26) / sin_theta_p;

  std::array<Eigen::Vector3d, 4> z5_all;
  std::array<Eigen::Vector3d, 4> v2p_all;
  std::array<bool, 4> branch_ready = {false, false, false, false};

  for (int index = 0; index < 2; ++index) {
    if (!std::isfinite(q6[index])) {
      continue;
    }
    const Eigen::Vector3d z_6_5(0.0, std::sin(q6[index]), std::cos(q6[index]));
    const Eigen::Vector3d z5 = r6 * z_6_5;
    const Eigen::Vector3d v2p = v26 + lp6 * z5;
    const double l2p = v2p.norm();
    if (l2p < 1e-10) {
      continue;
    }

    z5_all[2 * index] = z5;
    z5_all[2 * index + 1] = z5;
    v2p_all[2 * index] = v2p;
    v2p_all[2 * index + 1] = v2p;
    branch_ready[2 * index] = true;
    branch_ready[2 * index + 1] = true;

    if (std::abs(v2p.y() / l2p) > 0.999) {
      q_all[2 * index][0] = reference_joint_angles[0];
      q_all[2 * index][1] = 0.0;
      q_all[2 * index + 1][0] = reference_joint_angles[0];
      q_all[2 * index + 1][1] = 0.0;
    } else {
      const double sin_q2 = Clip(v2p.y() / l2p, -1.0, 1.0);
      const double q2_principal = std::asin(sin_q2);
      q_all[2 * index][0] = std::atan2(-v2p.x(), -v2p.z());
      q_all[2 * index][1] = q2_principal;
      q_all[2 * index + 1][0] = std::atan2(v2p.x(), v2p.z());
      q_all[2 * index + 1][1] =
          q2_principal >= 0.0 ? kPi - q2_principal : -kPi - q2_principal;
    }
  }

  for (int index = 0; index < 4; ++index) {
    if (!branch_ready[index] || q_all[index][0] <= limits.lower[0] ||
        q_all[index][0] >= limits.upper[0] ||
        q_all[index][1] <= limits.lower[1] ||
        q_all[index][1] >= limits.upper[1]) {
      continue;
    }

    Eigen::Vector3d z3;
    if (!NormalizeVector(-v2p_all[index], &z3)) {
      continue;
    }
    Eigen::Vector3d y3;
    if (!NormalizeVector(v26.cross(v2p_all[index]), &y3)) {
      continue;
    }
    const Eigen::Vector3d x3 = y3.cross(z3);

    const double c1 = std::cos(q_all[index][0]);
    const double s1 = std::sin(q_all[index][0]);
    Eigen::Matrix3d r1;
    r1 << c1, 0.0, s1,
          0.0, 1.0, 0.0,
          -s1, 0.0, c1;
    const double c2 = std::cos(q_all[index][1]);
    const double s2 = std::sin(q_all[index][1]);
    Eigen::Matrix3d r12;
    r12 << 1.0, 0.0, 0.0,
           0.0, c2, -s2,
           0.0, s2, c2;
    const Eigen::Vector3d x_2_3 = (r1 * r12).transpose() * x3;
    q_all[index][2] = std::atan2(x_2_3.y(), x_2_3.x());
    if (q_all[index][2] <= limits.lower[2] || q_all[index][2] >= limits.upper[2]) {
      q_all[index][2] = nan;
      continue;
    }

    const Eigen::Vector3d vh4 =
        p2 + 0.2837 * z3 + 0.02 * x3 - p6 + (0.126 + 0.1075) * z5_all[index];
    const double c6 = std::cos(q_all[index][5]);
    const double s6 = std::sin(q_all[index][5]);
    Eigen::Matrix3d r56;
    r56 << 1.0, 0.0, 0.0,
           0.0, c6, -s6,
           0.0, s6, c6;
    const Eigen::Matrix3d r5 = r6 * r56.transpose();
    const Eigen::Vector3d v_5_h4 = r5.transpose() * vh4;
    q_all[index][4] = -std::atan2(v_5_h4.y(), v_5_h4.x());
    if (q_all[index][4] <= limits.lower[4] || q_all[index][4] >= limits.upper[4]) {
      q_all[index][4] = nan;
      continue;
    }
  }

  return q_all;
}

std::array<double, 7> IkLeftArm(const Eigen::Matrix4d& shoulder_target) {
  const JointLimits limits = GetJointLimits("grasp");
  std::array<double, 7> reference_joint_angles{};
  std::array<double, 7> best_joint_angles;
  best_joint_angles.fill(std::numeric_limits<double>::quiet_NaN());

  // 关键步骤：C++ new_ik 搜索只判断可达性，使用均匀 q7 采样覆盖解析分支。
  for (int sample_index = 0; sample_index < kNewIkQ7SampleCount; ++sample_index) {
    const double ratio =
        static_cast<double>(sample_index + 1) / static_cast<double>(kNewIkQ7SampleCount + 1);
    const double q7 = limits.lower[6] + ratio * (limits.upper[6] - limits.lower[6]);
    const std::array<std::array<double, 7>, 4> candidates =
        NewIkLeftOnce(reference_joint_angles, q7, shoulder_target, limits);
    for (const std::array<double, 7>& candidate : candidates) {
      if (IsNewIkFkValid(candidate, shoulder_target, limits, nullptr, nullptr)) {
        return candidate;
      }
    }
  }
  return best_joint_angles;
}

std::array<double, 7> ComputeRightIk(const Eigen::Vector3d& target_pos,
                                     const Eigen::Matrix3d& target_rot,
                                     const std::string& model_type) {
  const Eigen::Matrix3d reflection_y = (Eigen::Vector3d(1.0, -1.0, 1.0)).asDiagonal();
  Eigen::Matrix3d left_rot = reflection_y * target_rot * reflection_y;
  Eigen::Vector3d left_pos = target_pos;
  left_pos.y() *= -1.0;

  Eigen::Matrix4d x07 = Eigen::Matrix4d::Identity();
  x07.block<3, 3>(0, 0) = left_rot;
  x07.block<3, 1>(0, 3) = left_pos - kL1LinkInBase;
  std::array<double, 7> left_theta = IkLeftArm(x07);

  const JointLimits limits = GetJointLimits(model_type);
  for (int i = 0; i < 7; ++i) {
    left_theta[i] = Clip(ClipToPi(left_theta[i]), limits.lower[i], limits.upper[i]);
  }

  return {left_theta[0], -left_theta[1], -left_theta[2], left_theta[3],
          -left_theta[4], -left_theta[5], left_theta[6]};
}

std::array<double, 7> ComputeLeftIk(const Eigen::Vector3d& target_pos,
                                    const Eigen::Matrix3d& target_rot,
                                    const std::string& model_type) {
  Eigen::Matrix4d x07 = Eigen::Matrix4d::Identity();
  x07.block<3, 3>(0, 0) = target_rot;
  x07.block<3, 1>(0, 3) = target_pos - kL1LinkInBase;
  std::array<double, 7> joint_angles = IkLeftArm(x07);

  const JointLimits limits = GetJointLimits(model_type);
  for (int index = 0; index < 7; ++index) {
    joint_angles[index] =
        Clip(ClipToPi(joint_angles[index]), limits.lower[index], limits.upper[index]);
  }
  return joint_angles;
}

CheckResult CheckRightPose(const Eigen::Matrix4d& target_pose,
                           const std::string& model_type,
                           double pos_threshold,
                           double angle_threshold) {
  CheckResult result;
  try {
    const Eigen::Vector3d target_pos = target_pose.block<3, 1>(0, 3);
    const Eigen::Matrix3d target_rot = target_pose.block<3, 3>(0, 0);
    const std::array<double, 7> joint_angles =
        ComputeRightIk(target_pos, target_rot, model_type);
    for (double angle : joint_angles) {
      if (!std::isfinite(angle)) {
        return result;
      }
    }
    const Eigen::Matrix4d fk = FkRightArm(joint_angles);
    const Eigen::Vector3d fk_pos = fk.block<3, 1>(0, 3) + kL1RightLinkInBase;
    result.pos_error = (fk_pos - target_pos).norm();
    const Eigen::Matrix3d delta = target_rot.transpose() * fk.block<3, 3>(0, 0);
    const double trace_value = delta.trace();
    result.angle_error = std::acos(Clip((trace_value - 1.0) * 0.5, -1.0, 1.0));
    result.valid =
        std::isfinite(result.pos_error) && std::isfinite(result.angle_error) &&
        result.pos_error <= pos_threshold && result.angle_error <= angle_threshold;
  } catch (...) {
    result.valid = false;
  }
  return result;
}

CheckResult CheckLeftPose(const Eigen::Matrix4d& target_pose,
                          const std::string& model_type,
                          double pos_threshold,
                          double angle_threshold) {
  CheckResult result;
  try {
    const Eigen::Vector3d target_pos = target_pose.block<3, 1>(0, 3);
    const Eigen::Matrix3d target_rot = target_pose.block<3, 3>(0, 0);
    const std::array<double, 7> joint_angles =
        ComputeLeftIk(target_pos, target_rot, model_type);
    for (double angle : joint_angles) {
      if (!std::isfinite(angle)) {
        return result;
      }
    }
    const Eigen::Matrix4d fk = FkLeftArm(joint_angles);
    const Eigen::Vector3d fk_pos = fk.block<3, 1>(0, 3) + kL1LinkInBase;
    result.pos_error = (fk_pos - target_pos).norm();
    const Eigen::Matrix3d delta = target_rot.transpose() * fk.block<3, 3>(0, 0);
    result.angle_error =
        std::acos(Clip((delta.trace() - 1.0) * 0.5, -1.0, 1.0));
    result.valid =
        std::isfinite(result.pos_error) && std::isfinite(result.angle_error) &&
        result.pos_error <= pos_threshold && result.angle_error <= angle_threshold;
  } catch (...) {
    result.valid = false;
  }
  return result;
}

CheckResult CheckPoseForArm(int arm_priority,
                            const Eigen::Matrix4d& target_pose,
                            const std::string& model_type,
                            double pos_threshold,
                            double angle_threshold) {
  return arm_priority == 0
             ? CheckRightPose(target_pose, model_type, pos_threshold, angle_threshold)
             : CheckLeftPose(target_pose, model_type, pos_threshold, angle_threshold);
}

std::pair<std::array<double, 6>, Eigen::Matrix4d> BuildRightGraspTarget(
    const Eigen::Matrix4d& grasp_pose,
    const Eigen::Matrix4d& target_from_source,
    double offset_m,
    bool rotate_z_180) {
  Eigen::Matrix4d offset = Eigen::Matrix4d::Identity();
  offset(2, 3) = -offset_m;
  Eigen::Matrix4d grasp_to_sdk = Eigen::Matrix4d::Identity();
  grasp_to_sdk(1, 1) = -1.0;
  grasp_to_sdk(2, 2) = -1.0;
  Eigen::Matrix4d target_pose = target_from_source * grasp_pose * offset * grasp_to_sdk;
  if (rotate_z_180) {
    Eigen::Matrix4d rotate = Eigen::Matrix4d::Identity();
    rotate(0, 0) = -1.0;
    rotate(1, 1) = -1.0;
    target_pose = target_pose * rotate;
  }

  const Eigen::Vector3d x_axis_offset = 0.01 * target_pose.block<3, 1>(0, 0);
  const Eigen::Vector3d positive_translation =
      target_pose.block<3, 1>(0, 3) + x_axis_offset;
  const Eigen::Vector3d negative_translation =
      target_pose.block<3, 1>(0, 3) - x_axis_offset;
  if (positive_translation.norm() < negative_translation.norm()) {
    target_pose.block<3, 1>(0, 3) = positive_translation;
  } else {
    target_pose.block<3, 1>(0, 3) = negative_translation;
  }
  return {MatrixToXyzYpr(target_pose), target_pose};
}

std::vector<double> BuildOrderedPregraspOffsets(double lower,
                                                double upper,
                                                int sample_count) {
  if (sample_count <= 0 || upper < lower) {
    throw std::runtime_error("Invalid pregrasp offset configuration");
  }
  std::vector<double> offsets;
  offsets.reserve(static_cast<size_t>(sample_count));
  if (sample_count == 1) {
    offsets.push_back(lower);
    return offsets;
  }
  std::vector<double> ascending;
  ascending.reserve(static_cast<size_t>(sample_count));
  const double step = (upper - lower) / static_cast<double>(sample_count - 1);
  for (int index = 0; index < sample_count; ++index) {
    ascending.push_back(lower + step * static_cast<double>(index));
  }
  const int middle = sample_count / 2;
  offsets.push_back(ascending[static_cast<size_t>(middle)]);
  for (int distance = 1; distance < sample_count; ++distance) {
    const int upper_index = middle + distance;
    const int lower_index = middle - distance;
    if (upper_index < sample_count) {
      offsets.push_back(ascending[static_cast<size_t>(upper_index)]);
    }
    if (lower_index >= 0) {
      offsets.push_back(ascending[static_cast<size_t>(lower_index)]);
    }
  }
  return offsets;
}

Eigen::Matrix4d BuildPregraspPose(const Eigen::Matrix4d& grasp_target_pose,
                                  double grasp_offset_m,
                                  double pregrasp_offset_m) {
  Eigen::Matrix4d pregrasp_pose = grasp_target_pose;
  pregrasp_pose.block<3, 1>(0, 3) +=
      (pregrasp_offset_m - grasp_offset_m) * grasp_target_pose.block<3, 1>(0, 2);
  return pregrasp_pose;
}

long long ResultOrderKey(const SearchResult& result, npy_intp grasp_count) {
  if (!result.found) {
    return std::numeric_limits<long long>::max();
  }
  const long long sample = static_cast<long long>(result.sample_index);
  const long long arm = static_cast<long long>(result.arm_priority);
  const long long grasp = static_cast<long long>(result.grasp_index);
  const long long rotate = result.rotate_z_180 ? 1LL : 0LL;
  return (((sample * 2LL + arm) * static_cast<long long>(grasp_count) + grasp) * 2LL) +
         rotate;
}

bool IsBetterResult(const SearchResult& candidate,
                    const SearchResult& current,
                    npy_intp grasp_count) {
  if (!candidate.found) {
    return false;
  }
  if (!current.found) {
    return true;
  }
  return ResultOrderKey(candidate, grasp_count) < ResultOrderKey(current, grasp_count);
}

SearchResult SearchSampleRange(const SearchContext& context,
                               npy_intp sample_begin,
                               npy_intp sample_end,
                               const std::atomic<long long>* best_order_key) {
  SearchResult result;
  for (npy_intp sample_index = sample_begin; sample_index < sample_end; ++sample_index) {
    const long long sample_first_key =
        (static_cast<long long>(sample_index) * 2LL *
         static_cast<long long>(context.grasp_count)) * 2LL;
    if (best_order_key != nullptr && sample_first_key > best_order_key->load()) {
      break;
    }

    const Eigen::Matrix4d fixed_knee_from_waist =
        MatrixAt(context.sample_fixed, sample_index);
    const Eigen::Matrix4d waist_from_base =
        fixed_knee_from_waist.inverse() * context.fixed_knee_from_base;
    const Eigen::Matrix4d base_from_map = MatrixAt(context.sample_base, sample_index);

    // 关键步骤：可按业务阶段限制单侧手臂，both 模式仍保持右手优先。
    for (int arm_priority = context.arm_begin; arm_priority < context.arm_end;
         ++arm_priority) {
      for (npy_intp grasp_index = 0; grasp_index < context.grasp_count; ++grasp_index) {
        Eigen::Matrix4d source_grasp_pose = MatrixAt(context.grasp_poses, grasp_index);
        Eigen::Matrix4d base_grasp_pose =
            context.use_map_poses ? base_from_map * source_grasp_pose : source_grasp_pose;

        for (int rotate_index = 0; rotate_index < 2; ++rotate_index) {
        const bool rotate_z_180 = rotate_index == 1;
        const long long candidate_key =
            ((static_cast<long long>(sample_index) * 2LL +
              static_cast<long long>(arm_priority)) *
                 static_cast<long long>(context.grasp_count) +
             static_cast<long long>(grasp_index)) *
                2LL +
            static_cast<long long>(rotate_index);
        if (best_order_key != nullptr && candidate_key > best_order_key->load()) {
          return result;
        }

        const std::pair<std::array<double, 6>, Eigen::Matrix4d> target =
            BuildRightGraspTarget(
                base_grasp_pose, waist_from_base, context.grasp_offset_m, rotate_z_180);
        const CheckResult check = CheckPoseForArm(
            arm_priority, target.second, context.model_type, context.pos_threshold,
            context.angle_threshold);
        if (!check.valid) {
          continue;
        }

        bool pregrasp_found = false;
        Eigen::Matrix4d pregrasp_pose = Eigen::Matrix4d::Identity();
        double selected_pregrasp_offset_m = 0.0;
        const std::vector<double> pregrasp_offsets = BuildOrderedPregraspOffsets(
            context.pregrasp_offset_min_m, context.pregrasp_offset_max_m,
            context.pregrasp_offset_samples);
        for (double pregrasp_offset_m : pregrasp_offsets) {
          const Eigen::Matrix4d candidate_pregrasp_pose = BuildPregraspPose(
              target.second, context.grasp_offset_m, pregrasp_offset_m);
          const CheckResult pregrasp_check = CheckPoseForArm(
              arm_priority, candidate_pregrasp_pose, context.model_type, 0.03, 0.50);
          if (!pregrasp_check.valid) {
            continue;
          }
          pregrasp_found = true;
          pregrasp_pose = candidate_pregrasp_pose;
          selected_pregrasp_offset_m = pregrasp_offset_m;
          break;
        }
        if (!pregrasp_found) {
          continue;
        }

        result.found = true;
        result.sample_index = static_cast<int>(sample_index);
        result.arm_priority = arm_priority;
        result.grasp_index = static_cast<int>(grasp_index);
        result.rotate_z_180 = rotate_z_180;
        result.grasp_pose = base_grasp_pose;
        result.grasp_target_pose = target.second;
        result.waist_from_base = waist_from_base;
        result.grasp_target = target.first;
        result.pregrasp_target_pose = pregrasp_pose;
        result.pregrasp_target = MatrixToXyzYpr(pregrasp_pose);
        result.pregrasp_offset_m = selected_pregrasp_offset_m;
        result.pos_error = check.pos_error;
        result.angle_error = check.angle_error;
        return result;
        }
      }
    }
  }
  return result;
}

void StoreBestResult(const SearchContext& context,
                     const SearchResult& candidate,
                     SearchResult* best_result,
                     std::atomic<long long>* best_order_key,
                     std::mutex* best_mutex) {
  if (!candidate.found) {
    return;
  }

  std::lock_guard<std::mutex> lock(*best_mutex);
  if (!IsBetterResult(candidate, *best_result, context.grasp_count)) {
    return;
  }
  *best_result = candidate;
  best_order_key->store(ResultOrderKey(candidate, context.grasp_count));
}

void SearchWorker(const SearchContext* context,
                  const std::vector<SampleRange>* ranges,
                  std::atomic<size_t>* next_range_index,
                  SearchResult* best_result,
                  std::atomic<long long>* best_order_key,
                  std::mutex* best_mutex) {
  while (true) {
    const size_t range_index = next_range_index->fetch_add(1);
    if (range_index >= ranges->size()) {
      return;
    }

    const SampleRange& range = (*ranges)[range_index];
    const long long range_first_key =
        (static_cast<long long>(range.begin) * static_cast<long long>(context->grasp_count)) * 2LL;
    if (range_first_key > best_order_key->load()) {
      return;
    }

    // 关键步骤：每个线程内部仍按原始 sample/candidate/rotate 顺序找本段第一个解。
    const SearchResult local_result =
        SearchSampleRange(*context, range.begin, range.end, best_order_key);
    StoreBestResult(*context, local_result, best_result, best_order_key, best_mutex);
  }
}

std::vector<SampleRange> BuildSampleRanges(npy_intp sample_count, int batch_size) {
  std::vector<SampleRange> ranges;
  if (sample_count <= 0) {
    return ranges;
  }
  if (batch_size <= 0) {
    batch_size = static_cast<int>(sample_count);
  }
  for (npy_intp begin = 0; begin < sample_count; begin += batch_size) {
    SampleRange range;
    range.begin = begin;
    range.end = std::min<npy_intp>(begin + batch_size, sample_count);
    ranges.push_back(range);
  }
  return ranges;
}

SearchResult SearchAllSamples(const SearchContext& context, int worker_count, int batch_size) {
  if (context.sample_count <= 0 || context.grasp_count <= 0) {
    return SearchResult();
  }
  if (worker_count <= 1 || context.sample_count <= 1) {
    return SearchSampleRange(context, 0, context.sample_count, nullptr);
  }

  const std::vector<SampleRange> ranges = BuildSampleRanges(context.sample_count, batch_size);
  if (ranges.empty()) {
    return SearchResult();
  }

  worker_count = std::min<int>(worker_count, static_cast<int>(ranges.size()));
  SearchResult best_result;
  std::atomic<long long> best_order_key(std::numeric_limits<long long>::max());
  std::atomic<size_t> next_range_index(0);
  std::mutex best_mutex;
  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(worker_count));
  for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
    threads.emplace_back(SearchWorker, &context, &ranges, &next_range_index,
                         &best_result, &best_order_key, &best_mutex);
  }
  for (std::thread& thread : threads) {
    thread.join();
  }
  return best_result;
}

bool SetDictItemSteal(PyObject* result, const char* key, PyObject* value) {
  if (value == nullptr) {
    return false;
  }
  const int rc = PyDict_SetItemString(result, key, value);
  Py_DECREF(value);
  return rc == 0;
}

PyObject* BuildResult(int sample_index,
                      int arm_priority,
                      int grasp_index,
                      bool rotate_z_180,
                      const Eigen::Matrix4d& grasp_pose,
                      const Eigen::Matrix4d& grasp_target_pose,
                      const Eigen::Matrix4d& waist_from_base,
                      const std::array<double, 6>& grasp_target,
                      const Eigen::Matrix4d& pregrasp_target_pose,
                      const std::array<double, 6>& pregrasp_target,
                      double pregrasp_offset_m,
                      double pos_error,
                      double angle_error) {
  PyObject* result = PyDict_New();
  if (result == nullptr) {
    return nullptr;
  }

  if (!SetDictItemSteal(result, "sample_index", PyLong_FromLong(sample_index)) ||
      !SetDictItemSteal(result, "arm_side",
                        PyUnicode_FromString(arm_priority == 0 ? "right" : "left")) ||
      !SetDictItemSteal(result, "grasp_index", PyLong_FromLong(grasp_index)) ||
      !SetDictItemSteal(result, "rotate_z_180", PyBool_FromLong(rotate_z_180 ? 1 : 0)) ||
      !SetDictItemSteal(result, "grasp_pose", MatrixToNumpy(grasp_pose)) ||
      !SetDictItemSteal(result, "grasp_target_pose", MatrixToNumpy(grasp_target_pose)) ||
      !SetDictItemSteal(result, "waist_from_base", MatrixToNumpy(waist_from_base)) ||
      !SetDictItemSteal(result, "grasp_target", Vector6ToList(grasp_target)) ||
      !SetDictItemSteal(result, "pregrasp_target_pose", MatrixToNumpy(pregrasp_target_pose)) ||
      !SetDictItemSteal(result, "pregrasp_target", Vector6ToList(pregrasp_target)) ||
      !SetDictItemSteal(result, "pregrasp_offset_m", PyFloat_FromDouble(pregrasp_offset_m)) ||
      !SetDictItemSteal(result, "pos_error", PyFloat_FromDouble(pos_error)) ||
      !SetDictItemSteal(result, "angle_error", PyFloat_FromDouble(angle_error))) {
    Py_DECREF(result);
    return nullptr;
  }
  return result;
}

void ValidateArray(PyArrayObject* array, int ndim, const npy_intp* dims, const char* name) {
  if (PyArray_NDIM(array) != ndim) {
    throw std::runtime_error(std::string(name) + " ndim mismatch");
  }
  for (int i = 0; i < ndim; ++i) {
    if (dims[i] >= 0 && PyArray_DIM(array, i) != dims[i]) {
      throw std::runtime_error(std::string(name) + " shape mismatch");
    }
  }
}

PyObject* FindGraspTarget(PyObject*, PyObject* args, PyObject* kwargs) {
  PyObject* sample_fixed_obj = nullptr;
  PyObject* sample_base_obj = nullptr;
  PyObject* sample_enabled_obj = nullptr;
  PyObject* sample_chassis_obj = nullptr;
  PyObject* grasp_poses_obj = nullptr;
  const char* poses_frame = nullptr;
  PyObject* fixed_knee_obj = nullptr;
  double grasp_offset_m = 0.0;
  const char* model_type_c = nullptr;
  double pos_threshold = 0.01;
  double angle_threshold = 0.05;
  double pregrasp_offset_min_m = 0.12;
  double pregrasp_offset_max_m = 0.18;
  int pregrasp_offset_samples = 5;
  int cpp_search_workers = 1;
  int cpp_search_batch_size = 0;
  const char* allowed_arm_side_c = "both";

  static const char* keywords[] = {
      "sample_fixed_knee_from_waist",
      "sample_base_from_map",
      "sample_enabled",
      "sample_chassis_lateral_enabled",
      "grasp_poses",
      "poses_frame",
      "fixed_knee_from_base",
      "grasp_offset_m",
      "model_type",
      "pos_threshold",
      "angle_threshold",
      "pregrasp_offset_min_m",
      "pregrasp_offset_max_m",
      "pregrasp_offset_samples",
      "cpp_search_workers",
      "cpp_search_batch_size",
      "allowed_arm_side",
      nullptr,
  };
  if (!PyArg_ParseTupleAndKeywords(
          args, kwargs, "OOOOOsOdsdd|ddiiis", const_cast<char**>(keywords),
          &sample_fixed_obj, &sample_base_obj, &sample_enabled_obj, &sample_chassis_obj,
          &grasp_poses_obj, &poses_frame, &fixed_knee_obj, &grasp_offset_m,
          &model_type_c, &pos_threshold, &angle_threshold,
          &pregrasp_offset_min_m, &pregrasp_offset_max_m, &pregrasp_offset_samples,
          &cpp_search_workers,
          &cpp_search_batch_size,
          &allowed_arm_side_c)) {
    return nullptr;
  }

  PyArrayObject* sample_fixed = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(sample_fixed_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* sample_base = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(sample_base_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* sample_enabled = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(sample_enabled_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* sample_chassis = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(sample_chassis_obj, NPY_BOOL, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* grasp_poses = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(grasp_poses_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));
  PyArrayObject* fixed_knee = reinterpret_cast<PyArrayObject*>(
      PyArray_FROM_OTF(fixed_knee_obj, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY));

  if (sample_fixed == nullptr || sample_base == nullptr || sample_enabled == nullptr ||
      sample_chassis == nullptr || grasp_poses == nullptr || fixed_knee == nullptr) {
    Py_XDECREF(sample_fixed);
    Py_XDECREF(sample_base);
    Py_XDECREF(sample_enabled);
    Py_XDECREF(sample_chassis);
    Py_XDECREF(grasp_poses);
    Py_XDECREF(fixed_knee);
    return nullptr;
  }

  PyObject* output = nullptr;
  try {
    const npy_intp sample_count = PyArray_DIM(sample_fixed, 0);
    const npy_intp grasp_count = PyArray_DIM(grasp_poses, 0);
    const npy_intp sample_dims[3] = {sample_count, 4, 4};
    const npy_intp sample_bool_dims[1] = {sample_count};
    const npy_intp grasp_dims[3] = {grasp_count, 4, 4};
    const npy_intp fixed_dims[2] = {4, 4};
    ValidateArray(sample_fixed, 3, sample_dims, "sample_fixed_knee_from_waist");
    ValidateArray(sample_base, 3, sample_dims, "sample_base_from_map");
    ValidateArray(sample_enabled, 1, sample_bool_dims, "sample_enabled");
    ValidateArray(sample_chassis, 1, sample_bool_dims, "sample_chassis_lateral_enabled");
    ValidateArray(grasp_poses, 3, grasp_dims, "grasp_poses");
    ValidateArray(fixed_knee, 2, fixed_dims, "fixed_knee_from_base");

    const bool use_map_poses = std::string(poses_frame) == "map";
    const std::string model_type(model_type_c);
    const Eigen::Matrix4d fixed_knee_from_base = Matrix2d(fixed_knee);

    SearchContext context;
    context.sample_fixed = sample_fixed;
    context.sample_base = sample_base;
    context.grasp_poses = grasp_poses;
    context.sample_count = sample_count;
    context.grasp_count = grasp_count;
    context.use_map_poses = use_map_poses;
    context.fixed_knee_from_base = fixed_knee_from_base;
    context.grasp_offset_m = grasp_offset_m;
    context.model_type = model_type;
    context.pos_threshold = pos_threshold;
    context.angle_threshold = angle_threshold;
    context.pregrasp_offset_min_m = pregrasp_offset_min_m;
    context.pregrasp_offset_max_m = pregrasp_offset_max_m;
    context.pregrasp_offset_samples = pregrasp_offset_samples;
    const std::string allowed_arm_side(allowed_arm_side_c);
    if (allowed_arm_side == "right") {
      context.arm_begin = 0;
      context.arm_end = 1;
    } else if (allowed_arm_side == "left") {
      context.arm_begin = 1;
      context.arm_end = 2;
    } else if (allowed_arm_side != "both" && !allowed_arm_side.empty()) {
      throw std::runtime_error("allowed_arm_side must be right, left or both");
    }

    // 关键步骤：多线程只做纯 C++ 搜索，Python 对象构造保持在主线程。
    const SearchResult search_result =
        SearchAllSamples(context, cpp_search_workers, cpp_search_batch_size);
    if (search_result.found) {
      output = BuildResult(search_result.sample_index, search_result.arm_priority,
                           search_result.grasp_index,
                           search_result.rotate_z_180, search_result.grasp_pose,
                           search_result.grasp_target_pose, search_result.waist_from_base,
                           search_result.grasp_target, search_result.pregrasp_target_pose,
                           search_result.pregrasp_target, search_result.pregrasp_offset_m,
                           search_result.pos_error,
                           search_result.angle_error);
    } else {
      Py_INCREF(Py_None);
      output = Py_None;
    }
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
    output = nullptr;
  }

  Py_DECREF(sample_fixed);
  Py_DECREF(sample_base);
  Py_DECREF(sample_enabled);
  Py_DECREF(sample_chassis);
  Py_DECREF(grasp_poses);
  Py_DECREF(fixed_knee);
  return output;
}

PyMethodDef Methods[] = {
    {"find_grasp_target", reinterpret_cast<PyCFunction>(FindGraspTarget),
     METH_VARARGS | METH_KEYWORDS,
     "Find the minimum-motion valid grasp target for both arms."},
    {"find_right_grasp_target", reinterpret_cast<PyCFunction>(FindGraspTarget),
     METH_VARARGS | METH_KEYWORDS,
     "Backward-compatible alias of find_grasp_target."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef Module = {
    PyModuleDef_HEAD_INIT,
    "_grasp_search_new_ik_cpp",
    "C++ dual-arm grasp search kernel.",
    -1,
    Methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__grasp_search_new_ik_cpp() {
  import_array();
  return PyModule_Create(&Module);
}
