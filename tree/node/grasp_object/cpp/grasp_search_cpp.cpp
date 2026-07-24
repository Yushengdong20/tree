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

const Eigen::Vector3d kL1LinkInBase(0.0, 0.253, 0.242);
const Eigen::Vector3d kL1RightLinkInBase(0.0, -0.253, 0.242);

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

double SafeAtan2(double y, double x) {
  return (x * x + y * y) > 1e-6 ? std::atan2(y, x) : 0.0;
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
  JointLimits limits;
  limits.lower = {-kPi, -0.349065850398866, -1.5707963267949,
                  -2.61799387799149, -1.5707963267949,
                  -1.30899693899575, -1.9198621771937625};
  limits.upper = {1.5707963267949, 2.0943951023932, 1.5707963267949,
                  0.0, 1.5707963267949, 0.698131700797732,
                  1.0471975511965976};
  if (model_type == "grasp") {
    limits.lower[6] = -0.658131700797732;
    limits.upper[6] = 0.658131700797732;
  } else if (model_type != "move_box") {
    throw std::runtime_error("Unsupported model_type: " + model_type);
  }
  return limits;
}

Eigen::Vector3d EulerZxyFromMatrix(const Eigen::Matrix3d& matrix) {
  // 与 scipy Rotation.as_euler("ZXY") 对齐的常规分支。
  const double x = std::asin(Clip(matrix(2, 1), -1.0, 1.0));
  const double cos_x = std::cos(x);
  double z = 0.0;
  double y = 0.0;
  if (std::abs(cos_x) > 1e-12) {
    z = std::atan2(-matrix(0, 1), matrix(1, 1));
    y = std::atan2(-matrix(2, 0), matrix(2, 2));
  } else {
    z = std::atan2(matrix(1, 0), matrix(0, 0));
    y = 0.0;
  }
  return Eigen::Vector3d(z, x, y);
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

std::array<double, 7> IkLeftArm(const Eigen::Matrix4d& x07) {
  Eigen::Matrix4d x07_tmp = x07;
  Eigen::Vector3d err = Eigen::Vector3d::Zero();
  double err_norm = std::numeric_limits<double>::infinity();
  std::array<double, 7> theta{};

  const Eigen::Matrix3d base_rotation = RotY(0.6 * kPi / 4.0) * RotZ(kPi / 4.0);
  const Eigen::Matrix3d prefix_rotation = RotZ(-kPi / 4.0) * RotY(-0.6 * kPi / 4.0);

  for (int iter = 0; iter < 10; ++iter) {
    x07_tmp.block<3, 1>(0, 3) -= err;
    const Eigen::Vector3d rotated_position = base_rotation * x07_tmp.block<3, 1>(0, 3);
    const double length = x07_tmp.block<3, 1>(0, 3).norm();
    const double x07p = rotated_position.x();
    const double y07p = rotated_position.y();
    const double z07p = rotated_position.z();

    const double theta4_virtual =
        kPi - std::acos(Clip((kL34 * kL34 + kL45 * kL45 - length * length) /
                                (2.0 * kL34 * kL45),
                            -1.0, 1.0));
    const double x37 = kL45 * std::sin(theta4_virtual);
    const double z37 = -(kL34 + kL45 * std::cos(theta4_virtual));
    const double theta3_virtual =
        SafeAtan2(x07p, std::hypot(y07p, z07p)) - SafeAtan2(x37, -z37);
    const double theta_sum =
        theta3_virtual + theta4_virtual + std::atan2(0.02, 0.126 + 0.1075);
    const Eigen::Matrix3d r24 = RotY(-theta_sum);
    const double theta2_virtual = SafeAtan2(y07p, -z07p);
    Eigen::Matrix3d r04 = RotX(theta2_virtual) * r24;
    r04 = prefix_rotation * r04;
    const Eigen::Matrix3d r47 = r04.transpose() * x07_tmp.block<3, 3>(0, 0);
    const Eigen::Vector3d zxy = EulerZxyFromMatrix(r47);

    const double theta4 = -(theta4_virtual + kOffsetTheta4);
    const Eigen::Matrix3d r34 = RotY(theta4);
    const Eigen::Matrix3d r03 = r04 * r34.transpose();
    double theta1 = SafeAtan2(r03(0, 2), r03(2, 2));
    if (theta1 > 0.75 * kPi) {
      theta1 -= 2.0 * kPi;
    }
    const double theta2 = std::asin(Clip(-r03(1, 2), -1.0, 1.0));
    const Eigen::Matrix3d r01 = RotY(theta1);
    const Eigen::Matrix3d r12 = RotX(theta2);
    const Eigen::Matrix3d r23 = r12.transpose() * r01.transpose() * r03;
    const double theta3 = SafeAtan2(r23(1, 0), r23(0, 0));

    theta = {theta1, theta2, theta3, theta4, zxy[0], zxy[1], zxy[2]};
    const Eigen::Vector3d fk_pos = FkLeftArm(theta).block<3, 1>(0, 3);
    const Eigen::Vector3d next_err = fk_pos - x07.block<3, 1>(0, 3);
    const double next_norm = next_err.norm();
    const bool active_next = std::abs(err_norm - next_norm) >= 1e-6;
    err = next_err;
    err_norm = next_norm;
    if (!active_next) {
      break;
    }
  }
  return theta;
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
    "_grasp_search_cpp",
    "C++ dual-arm grasp search kernel.",
    -1,
    Methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__grasp_search_cpp() {
  import_array();
  return PyModule_Create(&Module);
}
