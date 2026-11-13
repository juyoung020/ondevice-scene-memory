// ICM-42688-P 연결 확인: I2C 로 칩을 찾아 가속도·자이로를 읽고, 가만히 둔 동안의 중력 방향과 자이로 치우침을 찍는다.
//   imu_probe [/dev/i2c-1] [초=5]
// 빌드: g++ -O2 -std=c++17 -o imu_probe imu_probe.cpp
// 배선(Pi 5 핀 번호): VCC → 1(3.3 V), GND → 6, SDA → 3, SCL → 5. I2C 켜기: sudo dtparam i2c_arm=on (재부팅 없이 적용)
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <thread>

static int fd = -1;

static bool wr(uint8_t reg, uint8_t v) {
    uint8_t b[2] = {reg, v};
    return write(fd, b, 2) == 2;
}
static bool rd(uint8_t reg, uint8_t* out, int n) {
    return write(fd, &reg, 1) == 1 && read(fd, out, n) == n;
}

int main(int argc, char** argv) {
    const char* dev = argc > 1 ? argv[1] : "/dev/i2c-1";
    double secs = argc > 2 ? std::atof(argv[2]) : 5.0;
    fd = open(dev, O_RDWR);
    if (fd < 0) {
        std::perror(dev);
        return 1;
    }
    int addr = 0;
    for (int a : {0x68, 0x69}) {
        uint8_t who = 0;
        if (ioctl(fd, I2C_SLAVE, a) == 0 && rd(0x75, &who, 1)) {
            std::printf("주소 0x%02x: WHO_AM_I 0x%02x%s\n", a, who, who == 0x47 ? " (ICM-42688-P)" : "");
            if (who == 0x47) addr = a;
        }
    }
    if (!addr) {
        std::fprintf(stderr, "ICM-42688-P 를 찾지 못함(0x68·0x69). 배선과 `i2cdetect -y 1` 을 확인\n");
        return 2;
    }
    ioctl(fd, I2C_SLAVE, addr);
    wr(0x11, 0x01); // 소프트 리셋
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    wr(0x4F, 0x07); // 자이로 ±2000 °/s, 200 Hz
    wr(0x50, 0x07); // 가속도 ±16 g, 200 Hz
    wr(0x4E, 0x0F); // 자이로·가속도 켬(저잡음)
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    const double as = 16.0 * 9.80665 / 32768.0, gs = 2000.0 / 32768.0;
    double sa[3] = {0, 0, 0}, sg[3] = {0, 0, 0};
    long n = 0, fail = 0;
    auto t0 = std::chrono::steady_clock::now(), next = t0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < secs) {
        uint8_t b[12];
        if (rd(0x1F, b, 12)) {
            double a[3], g[3];
            for (int i = 0; i < 3; ++i) {
                a[i] = static_cast<int16_t>(b[2 * i] << 8 | b[2 * i + 1]) * as;
                g[i] = static_cast<int16_t>(b[6 + 2 * i] << 8 | b[7 + 2 * i]) * gs;
                sa[i] += a[i];
                sg[i] += g[i];
            }
            if (n % 100 == 0)
                std::printf("가속도 %7.2f %7.2f %7.2f m/s²  자이로 %7.2f %7.2f %7.2f °/s\n", a[0], a[1], a[2], g[0], g[1], g[2]);
            ++n;
        } else {
            ++fail;
        }
        next += std::chrono::milliseconds(5);
        std::this_thread::sleep_until(next);
    }
    if (n == 0) {
        std::fprintf(stderr, "읽기 실패 %ld 번\n", fail);
        return 3;
    }
    double ax = sa[0] / n, ay = sa[1] / n, az = sa[2] / n, norm = std::sqrt(ax * ax + ay * ay + az * az);
    std::printf("표본 %ld 개(%.0f Hz), 읽기 실패 %ld\n", n, n / secs, fail);
    std::printf("평균 가속도 %.3f %.3f %.3f, 크기 %.3f m/s²(가만히 두면 9.81)\n", ax, ay, az, norm);
    std::printf("중력 방향(단위) %.3f %.3f %.3f\n", ax / norm, ay / norm, az / norm);
    std::printf("자이로 치우침 %.3f %.3f %.3f °/s\n", sg[0] / n, sg[1] / n, sg[2] / n);
    return 0;
}
