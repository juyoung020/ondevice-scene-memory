// 패치 GraftMap · YawOnly · TryGraft 판단 계산 대조(기준 빌드 플래그). 출력: 기록마다 float 들
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sophus/se3.hpp>
#include <Eigen/Geometry>
static Sophus::SE3f YawOnly(const Sophus::SE3f &S)
{
    const Eigen::Matrix3f R = S.rotationMatrix();
    const float yaw = atan2(R(1, 0), R(0, 0));
    return Sophus::SE3f(Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()).toRotationMatrix(), S.translation());
}
static float rnd() { return (float)rand() / RAND_MAX * 2.f - 1.f; }
static Sophus::SE3f rse3(float ang) {
    Eigen::Quaternionf q(1.f, ang * rnd(), ang * rnd(), ang * rnd()); q.normalize();
    return Sophus::SE3f(q, Eigen::Vector3f(3 * rnd(), 3 * rnd(), 3 * rnd()));
}
static void wq(FILE *f, const Sophus::SE3f &T) {
    Eigen::Quaternionf q = T.unit_quaternion(); float v[7] = {q.x(), q.y(), q.z(), q.w(), T.translation()[0], T.translation()[1], T.translation()[2]};
    fwrite(v, 4, 7, f);
}
int main(int argc, char **argv) {
    FILE *f = fopen(argv[1], "wb"); int n = atoi(argv[2]); srand(7);
    for (int i = 0; i < n; i++) {
        float ang = (i % 3 == 0) ? 0.001f : (i % 3 == 1 ? 0.05f : 1.0f);
        Sophus::SE3f S0 = rse3(ang * 3), P = rse3(1.f), A = rse3(ang), B = rse3(ang);
        if (i % 5 == 0) B = A * rse3(0.0005f);
        Eigen::Vector3f x(5 * rnd(), 5 * rnd(), 5 * rnd()), nv(rnd(), rnd(), rnd()); nv.normalize();
        wq(f, S0); wq(f, P); wq(f, A); wq(f, B); fwrite(x.data(), 4, 3, f); fwrite(nv.data(), 4, 3, f);
        // 결과
        Sophus::SE3f S = YawOnly(S0);
        wq(f, S);
        wq(f, P * S.inverse());
        Eigen::Vector3f y = S * x, m = S.so3() * nv; fwrite(y.data(), 4, 3, f); fwrite(m.data(), 4, 3, f);
        Sophus::SE3f D = A.inverse() * B;
        float dt = D.translation().norm(), da = D.so3().log().norm() * 180.f / M_PI;
        fwrite(&dt, 4, 1, f); fwrite(&da, 4, 1, f);
    }
    fclose(f);
}
