#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include <H5Cpp.h>
#include <fftw3.h>
#include <ctime> 
#include <cstdlib>
#include <tiffio.h>
#include <queue>

#ifdef _MSC_VER
#pragma warning(disable: 4018)  // 有符号/无符号不匹配
#pragma warning(disable: 4244)  // double 转 float
#pragma warning(disable: 4267)  // size_t 转 int
#pragma warning(disable: 4305)  // 截断
#pragma warning(disable: 4996)  // 安全函数
#pragma warning(disable: 6385)  // 缓冲区溢出误报
#pragma warning(disable: 6386)  // 缓冲区溢出误报
#endif

namespace fs = std::filesystem;
using namespace H5;
using namespace std;

const float PI = 3.14159265358979323846f;
using Complex = complex<float>;

const int TEST_WIDTH = 5096;
const int TEST_HEIGHT = 5096;

struct BandConfig {
    string name;
    float colorEnhance;
    float gamma;
    float speckleFilter;
    string description;
};

const BandConfig X_BAND = {
    .name = "X",
    .colorEnhance = 3.0f,
    .gamma = 0.8f,
    .speckleFilter = 0.3f,
    .description = "X波段 (3cm) - 标准配置"
};

const BandConfig KU_BAND = {
    .name = "Ku",
    .colorEnhance = 3.5f,
    .gamma = 0.7f,
    .speckleFilter = 0.5f,
    .description = "Ku波段 (2cm) - 小目标增强配置"
};

struct RGBImage {
    unsigned char* r;
    unsigned char* g;
    unsigned char* b;
    int width;
    int height;
    int size;

    RGBImage(int w, int h) : width(w), height(h) {
        size = w * h;
        r = new unsigned char[size]();
        g = new unsigned char[size]();
        b = new unsigned char[size]();
    }

    ~RGBImage() {
        delete[] r;
        delete[] g;
        delete[] b;
    }

    void savePPM(const string& filename) {
        ofstream file(filename, ios::binary);
        if (!file.is_open()) return;
        file << "P6\n" << width << " " << height << "\n255\n";
        for (int i = 0; i < size; i++) {
            file << r[i] << g[i] << b[i];
        }
        file.close();
        cout << "已保存: " << filename << " (" << width << "x" << height << ")" << endl;
    }

    void saveTIFF(const string& filename) {
        TIFF* out = TIFFOpen(filename.c_str(), "w");
        if (!out) return;

        // 设置必要的 TIFF 标签
        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 3);      // RGB 三通道
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(out, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
        TIFFSetField(out, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW); // 推荐 LZW 无损压缩

        // 组装临时行缓冲区进行写入
        unsigned char* line = new unsigned char[width * 3];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int idx = y * width + x;
                line[x * 3 + 0] = r[idx];
                line[x * 3 + 1] = g[idx];
                line[x * 3 + 2] = b[idx];
            }
            TIFFWriteScanline(out, line, y);
        }

        delete[] line;
        TIFFClose(out);
        cout << "已保存 TIFF (彩色): " << filename << endl;
    }
};

struct GrayImage {
    unsigned char* data;
    int width;
    int height;
    int size;

    GrayImage(int w, int h) : width(w), height(h) {
        size = w * h;
        data = new unsigned char[size]();
    }

    ~GrayImage() {
        delete[] data;
    }

    void savePGM(const string& filename) {
        ofstream file(filename, ios::binary);
        if (!file.is_open()) return;
        file << "P5\n" << width << " " << height << "\n255\n";
        file.write(reinterpret_cast<char*>(data), size);
        file.close();
        cout << "已保存灰度图: " << filename << " (" << width << "x" << height << ")" << endl;
    }

    void saveTIFF(const string& filename) {
        TIFF* out = TIFFOpen(filename.c_str(), "w");
        if (!out) return;

        TIFFSetField(out, TIFFTAG_IMAGEWIDTH, width);
        TIFFSetField(out, TIFFTAG_IMAGELENGTH, height);
        TIFFSetField(out, TIFFTAG_SAMPLESPERPIXEL, 1);      // 灰度单通道
        TIFFSetField(out, TIFFTAG_BITSPERSAMPLE, 8);
        TIFFSetField(out, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_MINISBLACK);
        TIFFSetField(out, TIFFTAG_COMPRESSION, COMPRESSION_LZW);

        for (int y = 0; y < height; y++) {
            TIFFWriteScanline(out, &data[y * width], y);
        }

        TIFFClose(out);
        cout << "已保存 TIFF (灰度): " << filename << endl;
    }
};

// [修复辅助函数] 二维频谱中心化移位
void fftshift2D(fftwf_complex* data, int width, int height) {
    fftwf_complex* temp = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * width * height);
    int halfW = width / 2;
    int halfH = height / 2;
    for (int y = 0; y < height; y++) {
        int shiftY = (y + halfH) % height;
        for (int x = 0; x < width; x++) {
            int shiftX = (x + halfW) % width;
            temp[shiftY * width + shiftX][0] = data[y * width + x][0];
            temp[shiftY * width + shiftX][1] = data[y * width + x][1];
        }
    }
    memcpy(data, temp, sizeof(fftwf_complex) * width * height);
    fftwf_free(temp);
}

GrayImage* createGrayImage(Complex* data, int width, int height, float gamma) {
    int size = width * height;
    vector<float> mag(size);
    double sum = 0, sq_sum = 0;

    // 1. 计算幅值并统计
    for (int i = 0; i < size; i++) {
        mag[i] = abs(data[i]);
        sum += mag[i];
        sq_sum += (double)mag[i] * mag[i];
    }

    // 2. 计算统计量 (均值和标准差)
    float mean = sum / size;
    float std = sqrt(sq_sum / size - (double)mean * mean);

    // 3. 确定映射区间：[Mean - 1*Std, Mean + 3*Std]
    // 这种方式能自动滤除干扰极值，保留 95% 以上的地物细节
    float min_val = max(0.0f, mean - 1.0f * std);
    float max_val = mean + 3.0f * std;

    GrayImage* output = new GrayImage(width, height);
    for (int i = 0; i < size; i++) {
        // 归一化到 0-1
        float norm = (mag[i] - min_val) / (max_val - min_val + 1e-6f);
        norm = max(0.0f, min(1.0f, norm));

        // 4. 应用 Gamma 校正 (提亮暗部)
        output->data[i] = (unsigned char)(pow(norm, gamma) * 255);
    }
    return output;
}

RGBImage* processCSI(Complex* input, int width, int height, const BandConfig& config) {
    cout << "\n正在执行 CSI 处理（凸包填充+曳尾光晕+亮白核心）..." << endl;
    int totalPixels = width * height;

    // 1. 先生成灰度底图
    cout << "1/5 生成灰度底图..." << endl;
    GrayImage* baseGray = createGrayImage(input, width, height, config.gamma * 1.2f);

    // 2. 子孔径分解
    cout << "2/5 分配FFT内存..." << endl;
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
    fftwf_complex* spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);

    for (int i = 0; i < totalPixels; i++) {
        in[i][0] = input[i].real();
        in[i][1] = input[i].imag();
    }

    cout << "3/5 FFT变换..." << endl;
    fftwf_plan p_for = fftwf_plan_dft_2d(height, width, in, spec, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p_for);
    fftshift2D(spec, width, height);

    // 多视提取
    const int LOOKS = 3;

    auto extractMultiLookSub = [&](int shift_type) -> vector<float> {
        vector<float> channelMag(totalPixels, 0.0f);

        float centers[3];
        if (shift_type == 0) {
            centers[0] = 0.14f; centers[1] = 0.20f; centers[2] = 0.26f;
        }
        else if (shift_type == 1) {
            centers[0] = 0.44f; centers[1] = 0.50f; centers[2] = 0.56f;
        }
        else {
            centers[0] = 0.74f; centers[1] = 0.80f; centers[2] = 0.86f;
        }

        float width_band = 0.18f;

        for (int look = 0; look < LOOKS; look++) {
            fftwf_complex* sub_spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
            memset(sub_spec, 0, sizeof(fftwf_complex) * totalPixels);

            float center = centers[look];

            for (int y = 0; y < height; y++) {
                float weight = 0;
                float relative_y = (float)y / height;
                float dist = fabs(relative_y - center);
                if (dist < width_band / 2) {
                    weight = 0.5f * (1.0f + cos(2.0f * PI * dist / width_band));
                }

                for (int x = 0; x < width; x++) {
                    int idx = y * width + x;
                    sub_spec[idx][0] = spec[idx][0] * weight;
                    sub_spec[idx][1] = spec[idx][1] * weight;
                }
            }

            fftwf_complex* out_img = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
            fftwf_plan p_back = fftwf_plan_dft_2d(height, width, sub_spec, out_img, FFTW_BACKWARD, FFTW_ESTIMATE);
            fftwf_execute(p_back);

            float norm_factor = 1.0f / (width * height);
            for (int i = 0; i < totalPixels; i++) {
                float mag = sqrt(out_img[i][0] * out_img[i][0] + out_img[i][1] * out_img[i][1]) * norm_factor;
                channelMag[i] += mag;
            }

            fftwf_destroy_plan(p_back);
            fftwf_free(sub_spec);
            fftwf_free(out_img);
        }

        return channelMag;
        };

    cout << "4/5 多视提取子孔径..." << endl;
    vector<float> magR = extractMultiLookSub(0);
    vector<float> magG = extractMultiLookSub(1);
    vector<float> magB = extractMultiLookSub(2);

    // 归一化
    auto normalizeChannel = [&](const vector<float>& mag) -> vector<float> {
        double sum = 0, sq_sum = 0;
        for (float v : mag) { sum += v; sq_sum += v * v; }
        float mean = sum / totalPixels;
        float std = sqrt(sq_sum / totalPixels - mean * mean);
        float minV = mean - 0.5f * std;
        float maxV = mean + 4.0f * std;

        vector<float> norm(totalPixels);
        for (int i = 0; i < totalPixels; i++) {
            float val = (mag[i] - minV) / (maxV - minV + 1e-6f);
            norm[i] = max(0.0f, min(1.0f, val));
        }
        return norm;
        };

    vector<float> normR = normalizeChannel(magR);
    vector<float> normG = normalizeChannel(magG);
    vector<float> normB = normalizeChannel(magB);

    // 计算各向异性和强度
    vector<float> anisotropy(totalPixels);
    vector<float> intensity(totalPixels);
    vector<float> originalHue(totalPixels, 0.0f);

    for (int i = 0; i < totalPixels; i++) {
        float r = normR[i], g = normG[i], b = normB[i];
        float max_c = max({ r, g, b });
        float mean_c = (r + g + b) / 3.0f;
        intensity[i] = mean_c;

        float var = ((r - mean_c) * (r - mean_c) + (g - mean_c) * (g - mean_c) + (b - mean_c) * (b - mean_c)) / 3.0f;
        float cv = sqrt(var) / (mean_c + 1e-6f);
        anisotropy[i] = min(1.0f, cv * 1.5f);

        if (max_c == r) originalHue[i] = 0.0f;
        else if (max_c == g) originalHue[i] = 1.0f;
        else originalHue[i] = 2.0f;
    }

    // 识别强目标
    vector<float> targetStrength(totalPixels);
    float min_intensity = 0.18f;
    float min_anisotropy = 0.45f;

    for (int i = 0; i < totalPixels; i++) {
        if (intensity[i] > min_intensity && anisotropy[i] > min_anisotropy) {
            targetStrength[i] = anisotropy[i] * intensity[i];
        }
        else {
            targetStrength[i] = 0.0f;
        }
    }

    // 计算阈值
    vector<float> sortedStrength = targetStrength;
    sort(sortedStrength.begin(), sortedStrength.end(), greater<float>());
    float strongTargetThresh = 0.0f;
    int strong_idx = totalPixels * 10 / 1000;
    if (strong_idx < totalPixels) strongTargetThresh = sortedStrength[strong_idx];
    strongTargetThresh = max(strongTargetThresh, 0.08f);

    cout << "   强度阈值: " << min_intensity << ", 各向异性阈值: " << min_anisotropy << endl;
    cout << "   强目标阈值: " << strongTargetThresh << endl;

    // 核心目标掩码
    vector<bool> coreTarget(totalPixels, false);
    int strongCount = 0;
    for (int i = 0; i < totalPixels; i++) {
        if (targetStrength[i] > strongTargetThresh) {
            coreTarget[i] = true;
            strongCount++;
        }
    }
    cout << "   识别核心目标: " << strongCount << endl;

    // ========== 凸包填充 ==========
    struct Point {
        int x, y;
        Point() : x(0), y(0) {}
        Point(int _x, int _y) : x(_x), y(_y) {}
    };

    vector<int> labelMap(totalPixels, -1);
    int labelCount = 0;
    vector<vector<Point> > regionPoints;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (coreTarget[idx] && labelMap[idx] == -1) {
                vector<Point> points;
                queue<pair<int, int> > q;
                q.push(make_pair(x, y));
                labelMap[idx] = labelCount;
                points.push_back(Point(x, y));

                while (!q.empty()) {
                    pair<int, int> front = q.front(); q.pop();
                    int cx = front.first;
                    int cy = front.second;
                    for (int dy = -1; dy <= 1; dy++) {
                        for (int dx = -1; dx <= 1; dx++) {
                            int nx = cx + dx, ny = cy + dy;
                            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                int nidx = ny * width + nx;
                                if (coreTarget[nidx] && labelMap[nidx] == -1) {
                                    labelMap[nidx] = labelCount;
                                    q.push(make_pair(nx, ny));
                                    points.push_back(Point(nx, ny));
                                }
                            }
                        }
                    }
                }
                regionPoints.push_back(points);
                labelCount++;
            }
        }
    }

    auto convexHull = [](vector<Point>& pts) -> vector<Point> {
        if (pts.size() <= 1) return pts;
        sort(pts.begin(), pts.end(), [](Point a, Point b) {
            return a.x < b.x || (a.x == b.x && a.y < b.y);
            });
        vector<Point> hull;
        for (int i = 0; i < (int)pts.size(); i++) {
            while (hull.size() >= 2) {
                Point& a = hull[hull.size() - 2];
                Point& b = hull.back();
                if ((b.x - a.x) * (pts[i].y - a.y) - (b.y - a.y) * (pts[i].x - a.x) <= 0)
                    hull.pop_back();
                else break;
            }
            hull.push_back(pts[i]);
        }
        int lower = hull.size();
        for (int i = pts.size() - 2; i >= 0; i--) {
            while (hull.size() > lower) {
                Point& a = hull[hull.size() - 2];
                Point& b = hull.back();
                if ((b.x - a.x) * (pts[i].y - a.y) - (b.y - a.y) * (pts[i].x - a.x) <= 0)
                    hull.pop_back();
                else break;
            }
            hull.push_back(pts[i]);
        }
        if (hull.size() > 1) hull.pop_back();
        return hull;
        };

    auto pointInPolygon = [](Point p, vector<Point>& poly) -> bool {
        bool inside = false;
        for (int i = 0, j = poly.size() - 1; i < (int)poly.size(); j = i++) {
            if (((poly[i].y > p.y) != (poly[j].y > p.y)) &&
                (p.x < (poly[j].x - poly[i].x) * (p.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x)) {
                inside = !inside;
            }
        }
        return inside;
        };

    vector<bool> convexFilled(totalPixels, false);
    int totalHullPixels = 0;
    vector<vector<Point> > convexHulls;  // 存储每个区域的凸包
    vector<int> hullCentersX, hullCentersY;  // 存储凸包中心

    for (int r = 0; r < (int)regionPoints.size(); r++) {
        vector<Point>& points = regionPoints[r];
        if (points.size() < 5) {
            for (int i = 0; i < (int)points.size(); i++) {
                convexFilled[points[i].y * width + points[i].x] = true;
            }
            convexHulls.push_back(vector<Point>());
            hullCentersX.push_back(0);
            hullCentersY.push_back(0);
            continue;
        }

        vector<Point> hull = convexHull(points);
        convexHulls.push_back(hull);
        if (hull.size() < 3) {
            hullCentersX.push_back(0);
            hullCentersY.push_back(0);
            continue;
        }

        // 计算凸包中心
        float centerX = 0, centerY = 0;
        for (int i = 0; i < (int)hull.size(); i++) {
            centerX += hull[i].x;
            centerY += hull[i].y;
        }
        hullCentersX.push_back((int)(centerX / hull.size()));
        hullCentersY.push_back((int)(centerY / hull.size()));

        // 填充凸包内部
        int minX = width, maxX = 0, minY = height, maxY = 0;
        for (int i = 0; i < (int)hull.size(); i++) {
            minX = min(minX, hull[i].x);
            maxX = max(maxX, hull[i].x);
            minY = min(minY, hull[i].y);
            maxY = max(maxY, hull[i].y);
        }

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                Point p(x, y);
                if (pointInPolygon(p, hull)) {
                    int idx = y * width + x;
                    convexFilled[idx] = true;
                    totalHullPixels++;
                }
            }
        }
    }

    for (int i = 0; i < totalPixels; i++) {
        if (coreTarget[i]) { convexFilled[i] = true;  totalHullPixels++; }

    }

    cout << "   凸包填充像素: " << totalHullPixels << endl;

    // ========== 生成亮白核心（凸包内部20%面积，形状与凸包相似） ==========
    vector<bool> whiteCore(totalPixels, false);
    float coreScale = 0.45f;  // 缩放因子 0.45 -> 面积约20% (0.45^2=0.2025)

    for (int r = 0; r < (int)convexHulls.size(); r++) {
        vector<Point>& hull = convexHulls[r];
        if (hull.size() < 3) continue;

        int centerX = hullCentersX[r];
        int centerY = hullCentersY[r];

        // 生成缩放后的凸包
        vector<Point> scaledHull;
        for (int i = 0; i < (int)hull.size(); i++) {
            int nx = centerX + (int)((hull[i].x - centerX) * coreScale);
            int ny = centerY + (int)((hull[i].y - centerY) * coreScale);
            scaledHull.push_back(Point(nx, ny));
        }

        // 填充缩放凸包内部
        int minX = width, maxX = 0, minY = height, maxY = 0;
        for (int i = 0; i < (int)scaledHull.size(); i++) {
            minX = min(minX, scaledHull[i].x);
            maxX = max(maxX, scaledHull[i].x);
            minY = min(minY, scaledHull[i].y);
            maxY = max(maxY, scaledHull[i].y);
        }

        for (int y = minY; y <= maxY; y++) {
            for (int x = minX; x <= maxX; x++) {
                Point p(x, y);
                if (pointInPolygon(p, scaledHull)) {
                    int idx = y * width + x;
                    if (convexFilled[idx]) {  // 只在原凸包内部有效
                        whiteCore[idx] = true;
                    }
                }
            }
        }
    }

    int whiteCoreCount = 0;
    for (int i = 0; i < totalPixels; i++) {
        if (whiteCore[i]) whiteCoreCount++;
    }
    cout << "   亮白核心像素: " << whiteCoreCount << endl;

    // ========== 曳尾光晕效果 ==========
    // 高亮度彗星核 5x5
    float cometKernel[5][5] = {
        {0.2f, 0.3f, 0.5f, 0.7f, 0.9f},
        {0.2f, 0.3f, 0.5f, 0.8f, 1.0f},
        {0.2f, 0.3f, 0.6f, 1.0f, 1.0f},
        {0.2f, 0.3f, 0.5f, 0.8f, 1.0f},
        {0.2f, 0.3f, 0.5f, 0.7f, 0.9f}
    };

    // 8个曳尾方向
    int directions[8][2] = {
        {1, 0}, {1, 1}, {0, 1}, {-1, 1},
        {-1, 0}, {-1, -1}, {0, -1}, {1, -1}
    };

    vector<float> starLayer(totalPixels, 0.0f);
    vector<float> starR(totalPixels, 0.0f);
    vector<float> starG(totalPixels, 0.0f);
    vector<float> starB(totalPixels, 0.0f);

    srand(42);

    // 1. 边缘核心点：带曳尾的彗星效果（避开亮白核心）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (coreTarget[idx] && !whiteCore[idx]) {
                int dirIdx = (x * 7 + y * 13) % 8;
                int dxDir = directions[dirIdx][0];
                int dyDir = directions[dirIdx][1];

                float strength = min(1.0f, targetStrength[idx] / strongTargetThresh);

                for (int dy = -2; dy <= 2; dy++) {
                    for (int dx = -2; dx <= 2; dx++) {
                        int nx = x + dx, ny = y + dy;
                        if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                            int kx = dx + 2;
                            int ky = dy + 2;

                            int dot = dx * dxDir + dy * dyDir;
                            float tailBoost = 1.0f;
                            if (dot > 0) tailBoost = 1.0f + dot * 0.25f;

                            float val = strength * cometKernel[ky][kx] * tailBoost;

                            int nidx = ny * width + nx;
                            if (val > starLayer[nidx]) {
                                starLayer[nidx] = val;

                                int seed = x * 65537 + y * 131071;
                                float hue_tilt = ((seed >> 8) & 0xFF) / 255.0f;
                                float hue = (originalHue[idx] + hue_tilt * 0.2f);
                                if (hue > 2.5f) hue -= 2.0f;

                                if (hue < 0.6f) {
                                    starR[nidx] = 1.0f;
                                    starG[nidx] = 0.20f + hue_tilt * 0.15f;
                                    starB[nidx] = 0.10f + hue_tilt * 0.10f;
                                }
                                else if (hue < 1.4f) {
                                    starR[nidx] = 0.15f + (1.0f - hue_tilt) * 0.10f;
                                    starG[nidx] = 1.0f;
                                    starB[nidx] = 0.20f + hue_tilt * 0.15f;
                                }
                                else if (hue < 2.2f) {
                                    starR[nidx] = 0.25f + hue_tilt * 0.15f;
                                    starG[nidx] = 0.15f + (1.0f - hue_tilt) * 0.10f;
                                    starB[nidx] = 1.0f;
                                }
                                else {
                                    starR[nidx] = 1.0f;
                                    starG[nidx] = 0.15f + hue_tilt * 0.15f;
                                    starB[nidx] = 0.80f + hue_tilt * 0.20f;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 2. 凸包内部：普通星星（避开亮白核心）
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            if (convexFilled[idx] && !coreTarget[idx] && !whiteCore[idx]) {
                if ((x % 6 == 3) && (y % 6 == 3)) {
                    int seed = x * 131071 + y * 65537;
                    float random = (seed % 100) / 100.0f;

                    if (random < 0.8f) {
                        for (int dy = -2; dy <= 2; dy++) {
                            for (int dx = -2; dx <= 2; dx++) {
                                int nx = x + dx, ny = y + dy;
                                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                                    int kx = dx + 2, ky = dy + 2;
                                    float strength = 0.6f + random * 0.3f;
                                    float val = strength * cometKernel[ky][kx] * 0.6f;

                                    int nidx = ny * width + nx;
                                    if (val > starLayer[nidx]) {
                                        starLayer[nidx] = val;

                                        int seed2 = x * 65537 + y * 131071;
                                        float hue_tilt = ((seed2 >> 8) & 0xFF) / 255.0f;
                                        float hue = (originalHue[idx] + hue_tilt * 0.2f);
                                        if (hue > 2.5f) hue -= 2.0f;

                                        if (hue < 0.6f) {
                                            starR[nidx] = 0.85f;
                                            starG[nidx] = 0.25f + hue_tilt * 0.15f;
                                            starB[nidx] = 0.15f + hue_tilt * 0.10f;
                                        }
                                        else if (hue < 1.4f) {
                                            starR[nidx] = 0.20f + (1.0f - hue_tilt) * 0.10f;
                                            starG[nidx] = 0.85f;
                                            starB[nidx] = 0.25f + hue_tilt * 0.15f;
                                        }
                                        else if (hue < 2.2f) {
                                            starR[nidx] = 0.30f + hue_tilt * 0.15f;
                                            starG[nidx] = 0.20f + (1.0f - hue_tilt) * 0.10f;
                                            starB[nidx] = 0.85f;
                                        }
                                        else {
                                            starR[nidx] = 0.85f;
                                            starG[nidx] = 0.20f + hue_tilt * 0.15f;
                                            starB[nidx] = 0.70f + hue_tilt * 0.20f;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    int starCount = 0;
    for (int i = 0; i < totalPixels; i++) {
        if (starLayer[i] > 0.01f) starCount++;
    }
    cout << "   星星像素: " << starCount << " (" << 100.0 * starCount / totalPixels << "%)" << endl;

    // ========== 合成最终图像 ==========
    cout << "5/5 合成图像..." << endl;
    RGBImage* output = new RGBImage(width, height);

    for (int i = 0; i < totalPixels; i++) {
        unsigned char gray_val = baseGray->data[i];
        unsigned char dark_gray = (unsigned char)(gray_val * 0.75f);

        // 优先显示亮白核心
        if (whiteCore[i]) {
            output->r[i] = 255;
            output->g[i] = 255;
            output->b[i] = 255;
        }
        else if (starLayer[i] > 0.05f) {
            float r = starR[i] * starLayer[i];
            float g = starG[i] * starLayer[i];
            float b = starB[i] * starLayer[i];

            r = min(1.0f, r * 1.2f);
            g = min(1.0f, g * 1.2f);
            b = min(1.0f, b * 1.2f);

            output->r[i] = (unsigned char)(min(1.0f, r) * 255);
            output->g[i] = (unsigned char)(min(1.0f, g) * 255);
            output->b[i] = (unsigned char)(min(1.0f, b) * 255);
        }
        else {
            output->r[i] = dark_gray;
            output->g[i] = dark_gray;
            output->b[i] = dark_gray;
        }
    }

    delete baseGray;
    fftwf_destroy_plan(p_for);
    fftwf_free(in);
    fftwf_free(spec);

    cout << "CSI处理完成！" << endl;
    return output;
}
// 读取HDF5数据集
short* readHDF5Data(const string& filename, const string& datasetName, int& width, int& height) {
    try {
        H5File file(filename, H5F_ACC_RDONLY);
        DataSet dataset = file.openDataSet(datasetName);
        DataSpace space = dataset.getSpace();

        // 获取维度
        hsize_t dims[2];
        space.getSimpleExtentDims(dims);

        height = dims[0];
        width = dims[1];
        int total = width * height;

        cout << datasetName << ": " << width << " x " << height << " = " << total << " 像素" << endl;

        // 读取数据
        short* data = new short[total];
        dataset.read(data, PredType::NATIVE_SHORT);

        file.close();
        return data;

    }
    catch (Exception& e) {
        cerr << "读取 " << datasetName << " 失败: " << e.getDetailMsg() << endl;
        return nullptr;
    }
}

// 选择波段
BandConfig selectBand() {
    cout << "\n请选择波段:" << endl;
    cout << "1. X波段 (3cm) - 标准配置" << endl;
    cout << "2. Ku波段 (2cm) - 小目标增强配置 (默认)" << endl;
    cout << "请输入选择 (1/2，直接回车默认Ku波段): ";

    string choice;
    getline(cin, choice);

    if (choice == "1") {
        cout << "已选择: X波段" << endl;
        return X_BAND;
    }
    else {
        cout << "已选择: Ku波段" << endl;
        return KU_BAND;
    }
}

int main() {
    // 开启循环，实现程序复用
    bool keepRunning = true;

    while (keepRunning) {
        try {
            cout << "\n========================================" << endl;
            cout << "   CSI彩色SAR图像处理 (支持X/Ku波段)" << endl;
            cout << "========================================" << endl;

            // 选择波段
            BandConfig config = selectBand();

            // 获取文件名
            string filename;
            cout << "\n请输入HDF5文件路径 (直接回车使用默认文件 1.h5, 输入 q 退出): ";
            getline(cin, filename);

            // 检查退出指令
            if (filename == "q" || filename == "Q") {
                break;
            }

            if (filename.empty()) {
                filename = "1.h5";
                cout << "使用默认文件: " << filename << endl;
            }

            if (!fs::exists(filename)) {
                cerr << "× 文件不存在: " << filename << endl;
                continue; // 返回菜单重新输入
            }

            // 关闭HDF5错误输出
            H5Eset_auto(H5E_DEFAULT, NULL, NULL);

            // 读取原始数据
            int fullWidth, fullHeight;
            short* realData = readHDF5Data(filename, "/s_i", fullWidth, fullHeight);
            if (!realData) continue;

            short* imagData = readHDF5Data(filename, "/s_q", fullWidth, fullHeight);
            if (!imagData) {
                delete[] realData;
                continue;
            }

            cout << "\n完整图像尺寸: " << fullWidth << " x " << fullHeight << endl;

            // 选择处理模式
            cout << "\n请选择处理模式:" << endl;
            cout << "1. 测试区域模式 (5096x5096) - 快速处理" << endl;
            cout << "2. 全图模式 - 处理整个图像 (可能需要大量内存)" << endl;
            cout << "请输入选择 (1/2，直接回车默认测试区域): ";

            string modeChoice;
            getline(cin, modeChoice);

            bool fullMode = (modeChoice == "2");
            Complex* data = nullptr;
            int processWidth, processHeight;

            if (fullMode) {
                // 全图模式逻辑
                cout << "\n已选择: 全图模式" << endl;
                processWidth = fullWidth;
                processHeight = fullHeight;

                data = new Complex[(size_t)fullWidth * fullHeight];

                cout << "正在准备全图数据..." << endl;
                for (int y = 0; y < fullHeight; y++) {
                    for (int x = 0; x < fullWidth; x++) {
                        size_t idx = (size_t)y * fullWidth + x;
                        data[idx] = Complex((float)realData[idx], (float)imagData[idx]);
                    }
                }
                cout << "全图内存分配: " << ((size_t)fullWidth * fullHeight * sizeof(Complex)) / (1024 * 1024) << " MB" << endl;
            }
            else {
                // 测试区域模式逻辑
                cout << "\n已选择: 测试区域模式" << endl;
                int testW = min(TEST_WIDTH, fullWidth);
                int testH = min(TEST_HEIGHT, fullHeight);
                int startX = 0, startY = 0;

                cout << "\n请选择测试区域位置:" << endl;
                cout << "1. 左上角\n2. 右上角\n3. 中间 (默认)\n4. 随机位置" << endl;
                cout << "请输入选择 (1/2/3/4，直接回车默认中间): ";

                string posChoice;
                getline(cin, posChoice);

                if (posChoice == "1") {
                    startX = 0; startY = 0;
                }
                else if (posChoice == "2") {
                    startX = fullWidth - testW; startY = 0;
                }
                else if (posChoice == "4") {
                    srand(time(nullptr));
                    int maxStartX = fullWidth - testW;
                    int maxStartY = fullHeight - testH;
                    if (maxStartX > 0 && maxStartY > 0) {
                        startX = rand() % (maxStartX + 1);
                        startY = rand() % (maxStartY + 1);
                    }
                    else {
                        startX = (fullWidth - testW) / 2;
                        startY = (fullHeight - testH) / 2;
                    }
                }
                else {
                    startX = (fullWidth - testW) / 2;
                    startY = (fullHeight - testH) / 2;
                }

                processWidth = testW;
                processHeight = testH;
                cout << "\n提取起始坐标: (" << startX << ", " << startY << ")" << endl;

                data = new Complex[(size_t)testW * testH];
                for (int y = 0; y < testH; y++) {
                    for (int x = 0; x < testW; x++) {
                        size_t srcIdx = static_cast<size_t>(startY + y) * fullWidth + (startX + x);
                        size_t dstIdx = static_cast<size_t>(y) * testW + x;
                        data[dstIdx] = Complex(static_cast<float>(realData[srcIdx]), static_cast<float>(imagData[srcIdx]));
                    }
                }

                // 生成测试区域灰度图对比
                cout << "\n正在生成测试区域灰度图..." << endl;
                GrayImage* grayTest = createGrayImage(data, testW, testH, config.gamma);
                string grayTestFile = "original_" + config.name + "_test_gray.pgm";
                grayTest->savePGM(grayTestFile);
                grayTest->saveTIFF("original_" + config.name + "_test.tiff");
                delete grayTest;
            }

            // 释放原始 short 数据内存
            delete[] realData;
            delete[] imagData;

            // 核心 CSI 处理逻辑
            RGBImage* result = nullptr;
            if (fullMode) {
                long long totalMemory = (long long)fullWidth * fullHeight * sizeof(Complex);
                if (totalMemory > 2LL * 1024 * 1024 * 1024) { // 2GB 预警
                    cout << "！ 警告：全图需要 " << totalMemory / (1024 * 1024) << " MB 内存。是否继续？(y/n): ";
                    string confirm;
                    getline(cin, confirm);
                    if (confirm != "y" && confirm != "Y") {
                        delete[] data;
                        continue;
                    }
                }
                result = processCSI(data, fullWidth, fullHeight, config);

                cout << "\n正在生成全图灰度图..." << endl;
                GrayImage* grayFull = createGrayImage(data, fullWidth, fullHeight, config.gamma);
                grayFull->savePGM("original_" + config.name + "_full_gray.pgm");
                grayFull->saveTIFF("original_" + config.name + "_full_gray.tiff");
                delete grayFull;
            }
            else {
                result = processCSI(data, processWidth, processHeight, config);
            }

            // 保存并清理结果
            string modeStr = fullMode ? "full" : "test";
            string outputFile = "csi_" + config.name + "_" + modeStr + ".ppm";
            if (result) {
                result->savePPM(outputFile);
                result->saveTIFF("csi_" + config.name + "_" + modeStr + ".tiff");
                delete result;
            }
            delete[] data;

            cout << "\n√ 处理完成！结果已保存至程序文件夹内！" << endl;

            // 询问是否继续处理下一个文件
            cout << "\n----------------------------------------" << endl;
            cout << "输入 'n' 退出程序，直接回车返回主菜单: ";
            string nextAction;
            getline(cin, nextAction);
            if (nextAction == "n" || nextAction == "N") {
                keepRunning = false;
            }

        }
        catch (bad_alloc& e) {
            cerr << "\n× 内存分配失败: " << e.what() << endl;
        }
        catch (exception& e) {
            cerr << "\n× 处理过程发生错误: " << e.what() << endl;
        }
    }

    cout << "\n程序已退出。" << endl;
    return 0;
}