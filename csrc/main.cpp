#include <math.h>
#include <iostream>
#include <chrono>
#include "Eigen/Eigen"

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

using namespace std;

namespace os::gigo::d3 {

  const double pi = 3.1415926535897932385;

  struct CamParamMini {
    double fx, fy, cx, cy;
    double pose_tx, pose_ty, pose_tz;
    double qx, qy, qz, qw;
    double yaw, pitch, roll;
    int image_width, image_height;
  };

  inline Eigen::Matrix3d R_from_quaternion(double x, double y, double z, double w) {
    Eigen::Quaterniond q;
    q.x() = x;
    q.y() = y;
    q.z() = z;
    q.w() = w;
    return q.normalized().toRotationMatrix();
  }

  inline Eigen::Matrix3d R_matrix(const double yaw, const double pitch, const double roll) {
    auto y = yaw / 180 * pi;
    auto p = pitch / 180 * pi;
    auto r = roll / 180 * pi;
    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0, sin(y), cos(y), 0, 0, 0, 1;
    Eigen::Matrix3d Ry;
    Ry << cos(p), 0, sin(p), 0, 1, 0, -sin(p), 0, cos(p);
    Eigen::Matrix3d Rx;
    Rx << 1, 0, 0, 0, cos(r), -sin(r), 0, sin(r), cos(r);
    return Rz * Ry * Rx;
  }


  inline Eigen::Matrix3d K_matrix(const CamParamMini &p) {
    Eigen::Matrix3d K;
    K << p.fx, 0, p.cx, 0, p.fy, p.cy, 0, 0, 1;
    return K;
  }

  // Project 3D points to image by pinhole camera model.
  std::vector<cv::Point2f>
  PersMapping(Eigen::Matrix<double, 3, 15> p3d, Eigen::Matrix3d P, Eigen::Matrix<double, 3, 1> T) {
    std::vector<cv::Point2f> pts;
    auto p3cam = p3d.colwise() - T; // 3 x 15  - 3 x 1 -> 3 x 15
    auto p2d = P * p3cam; // 3 x 3 @ 3x15 -> 3x15
    for (int i = 0; i < 15; ++i) {
      auto u = p2d(0, i);
      auto v = p2d(1, i);
      auto z = p2d(2, i);
      pts.push_back({static_cast<float >(u / z), static_cast<float >(v / z)});
    }
    return pts;
  }

  
  cv::Mat
  calc_vcam_matrix(const CamParamMini &src_cam_params, const CamParamMini &dst_cam_params, double height = -0.393) {
    // 4 points are enough. Here we choose 15 points.
    Eigen::Matrix<double, 15, 3> pts3d;
    pts3d << 10, 0, height, 7 + 2., -1, height,
      80, 0, height, 80, -8, height, 80, 8, height,
      30, 0, height, 30, -8, height, 30, 8, height,
      10, 0, height, 10, -8, height, 10, 8, height,
      10, -4, height, 10, 4, height,
      12, -2, height, 12, 2, height;
    auto pts3d_T = pts3d.transpose();
    auto Rs = R_from_quaternion(src_cam_params.qx, src_cam_params.qy, src_cam_params.qz, src_cam_params.qw);
    auto Rd = R_from_quaternion(dst_cam_params.qx, dst_cam_params.qy, dst_cam_params.qz, dst_cam_params.qw);
    Eigen::Matrix<double, 3, 1> Ts;
    Ts << src_cam_params.pose_tx, src_cam_params.pose_ty, src_cam_params.pose_tz;
    Eigen::Matrix<double, 3, 1> Td;
    Td << dst_cam_params.pose_tx, dst_cam_params.pose_ty, dst_cam_params.pose_tz;

    auto Ks = K_matrix(src_cam_params);
    auto Kd = K_matrix(dst_cam_params);

    auto Ps = (Rs * Ks.inverse()).inverse();
    auto Pd = (Rd * Kd.inverse()).inverse();

    auto pts_src = PersMapping(pts3d_T, Ps, Ts);
    auto pts_dst = PersMapping(pts3d_T, Pd, Td);

    auto h = cv::findHomography(pts_src, pts_dst);

    return h;
  }

  class VirtualCamera {
  public:
    VirtualCamera() = delete;

    VirtualCamera(const CamParamMini std_cam) {
      this->std_cam_ = std_cam;
      this->std_size = cv::Size(std_cam.image_width, std_cam.image_height);
    }

    void set_src_cam(const CamParamMini cam, double height = -0.393) {
      this->h_ = calc_vcam_matrix(cam , this->std_cam_, height = height);
    }

    cv::Mat transform(cv::Mat original_img) {
      cv::Mat output;
      cv::warpPerspective(original_img, output, this->h_, this->std_size, 0);
      return output;
    }

  private:
    CamParamMini std_cam_;
    cv::Mat h_;
    cv::Size std_size;
  };
}

int main() {
  float stride = 3.75;
  float std_stride = 1.875;
  
  // Uncomment this line if you need single-thread.
  // cv::setNumThreads(1);


  // Settings of Virtual Camera
  os::gigo::d3::CamParamMini std_cam
    {1948.3936767578125 / std_stride, 1957.3106689453125 / std_stride, 962.94580078125 /  std_stride, 559.2073974609375 / std_stride,
     2.1791250705718994, 0.0012771482579410076, 1.0726985931396484,
     -0.4864780008792877, 0.4880296289920807, -0.5088465809822083, 0.5159858465194702,
     1.5563105, -3.1311595, 1.6163771,
     static_cast<int>(3840 / 2 / std_stride), static_cast<int>(2160 / 2/ std_stride)
    };

  // Settings of Real Camera
  os::gigo::d3::CamParamMini src_cam{
    3807.4036 / stride, 3822.3098 / stride, 1903.5006 / stride, 1147.8899 / stride,
    2.187498, -0.009653946, 1.015064,
    -0.48510775, 0.4893328, -0.5197188, 0.5050903,
    1.5516857, -3.1316676, 1.621268,
    static_cast<int>(3840 / stride), static_cast<int>(2160 / stride)
  };

  cout << "Loading image" <<endl;
  auto src_img = cv::imread("a.jpg");
    
  using timer = chrono::high_resolution_clock;
  
  cv::resize(src_img, src_img, cv::Size(3840 / stride, 2160 / stride));
  cv::Size dst_size{src_cam.image_width, src_cam.image_height};

  cv::Mat output;

  // Establishing Virtual Camera
  os::gigo::d3::VirtualCamera vc(std_cam);
  vc.set_src_cam(src_cam);

  
  auto times = 1000;
  
  // Warm up
  cout << "Warm Up...." <<endl;
  
  auto s = timer::now();
  for (auto i = 0; i < times; ++i) {
    output = vc.transform(src_img);
  }
  auto e = timer::now();
  cout << std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() << "ms @" << times << "times" << endl;
  cout << "Warm Up.... DONE" <<endl;

  s = timer::now();
  for (auto i = 0; i < times; ++i) {
    output = vc.transform(src_img);
  }
  e = timer::now();
  auto int_ms = std::chrono::duration_cast<std::chrono::milliseconds>(e - s);
  cout << int_ms.count() << "ms @" << times << "times"<< endl;
  cv::imwrite("vcam.jpg", output);
  return 0;
}
