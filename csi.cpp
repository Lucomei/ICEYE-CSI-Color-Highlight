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

namespace fs = std::filesystem;
using namespace H5;
using namespace std;

const float PI = 3.14159265358979323846f;
using Complex = complex<float>;

const int TEST_WIDTH = 2048;
const int TEST_HEIGHT = 2048;

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

void applySpeckleFilter(vector<float>& data, int width, int height, float strength) {
    if (strength <= 0) return;
    vector<float> temp = data;
    int kernelSize = 3;
    int half = kernelSize / 2;

    for (int y = half; y < height - half; y++) {
        for (int x = half; x < width - half; x++) {
            int idx = y * width + x;
            float sum = 0;
            for (int ky = -half; ky <= half; ky++) {
                for (int kx = -half; kx <= half; kx++) {
                    sum += temp[(y + ky) * width + (x + kx)];
                }
            }
            data[idx] = data[idx] * (1 - strength) + (sum / 9.0f) * strength;
        }
    }
}

RGBImage* processCSI(Complex* input, int width, int height, const BandConfig& config) {
    cout << "\n正在执行 CSI 处理（各向异性检测+发光高亮模式）..." << endl;
    int totalPixels = width * height;

    // 1. 先生成灰度底图
    cout << "1/5 生成灰度底图..." << endl;
    GrayImage* baseGray = createGrayImage(input, width, height, config.gamma * 1.3f);

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

    // 子孔径提取（使用更窄的频带，增强各向异性检测能力）
    auto extractSub = [&](int shift_type) -> vector<float> {
        fftwf_complex* sub_spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
        memset(sub_spec, 0, sizeof(fftwf_complex) * totalPixels);

        for (int y = 0; y < height; y++) {
            float weight = 0;
            float relative_y = (float)y / height;
            if (shift_type == 0) {
                float center = 0.2f;
                float width_band = 0.35f;  // 更窄的频带
                float dist = fabs(relative_y - center);
                if (dist < width_band / 2) {
                    weight = 0.5f * (1.0f + cos(2.0f * PI * dist / width_band));
                }
            }
            else if (shift_type == 1) {
                float center = 0.5f;
                float width_band = 0.35f;
                float dist = fabs(relative_y - center);
                if (dist < width_band / 2) {
                    weight = 0.5f * (1.0f + cos(2.0f * PI * dist / width_band));
                }
            }
            else {
                float center = 0.8f;
                float width_band = 0.35f;
                float dist = fabs(relative_y - center);
                if (dist < width_band / 2) {
                    weight = 0.5f * (1.0f + cos(2.0f * PI * dist / width_band));
                }
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

        vector<float> mag(totalPixels);
        float norm_factor = 1.0f / (width * height);
        for (int i = 0; i < totalPixels; i++) {
            mag[i] = sqrt(out_img[i][0] * out_img[i][0] + out_img[i][1] * out_img[i][1]) * norm_factor;
        }

        fftwf_destroy_plan(p_back);
        fftwf_free(sub_spec);
        fftwf_free(out_img);
        return mag;
        };

    cout << "4/5 提取子孔径..." << endl;
    vector<float> magR = extractSub(0);
    vector<float> magG = extractSub(1);
    vector<float> magB = extractSub(2);

    // 归一化彩色通道
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

    // ========== 关键：计算各向异性程度 ==========
    // 各向异性 = 三个子孔径响应的差异程度
    // 各向同性（自然地形）：三个值相近 -> anisotropy小
    // 各向异性（人造目标）：三个值差异大 -> anisotropy大
    vector<float> anisotropy(totalPixels);      // 各向异性程度（0-1）
    vector<float> intensity(totalPixels);       // 总强度
    vector<float> originalHue(totalPixels, 0.0f);

    for (int i = 0; i < totalPixels; i++) {
        float r = normR[i], g = normG[i], b = normB[i];
        float max_c = max({ r, g, b });
        float min_c = min({ r, g, b });
        float mean_c = (r + g + b) / 3.0f;

        intensity[i] = mean_c;

        // 计算变异系数（标准差/均值）作为各向异性度量
        float var = ((r - mean_c) * (r - mean_c) + (g - mean_c) * (g - mean_c) + (b - mean_c) * (b - mean_c)) / 3.0f;
        float cv = sqrt(var) / (mean_c + 1e-6f);

        // 各向异性程度：cv越大，各向异性越强
        // 自然地形（各向同性）的cv通常 < 0.3
        // 人造目标（各向异性）的cv通常 > 0.5
        anisotropy[i] = min(1.0f, cv * 1.5f);

        // 记录主色调倾向
        if (max_c == r) originalHue[i] = 0.0f;
        else if (max_c == g) originalHue[i] = 1.0f;
        else originalHue[i] = 2.0f;
    }

    // ========== 基于各向异性识别强目标 ==========
    // 人造目标条件：各向异性高 且 强度足够（排除弱噪声）
    vector<float> targetStrength(totalPixels);

    float min_intensity = 0.15f;   // 强度至少15%
    float min_anisotropy = 0.45f;  // 各向异性至少45%（排除自然地形）

    for (int i = 0; i < totalPixels; i++) {
        if (intensity[i] > min_intensity && anisotropy[i] > min_anisotropy) {
            // 目标强度 = 各向异性 × 强度
            targetStrength[i] = anisotropy[i] * intensity[i];
        }
        else {
            targetStrength[i] = 0.0f;
        }
    }

    // 计算强目标阈值（前2%）
    vector<float> sortedStrength = targetStrength;
    sort(sortedStrength.begin(), sortedStrength.end(), greater<float>());

    float strongTargetThresh = 0.0f;
    int strong_idx = totalPixels * 2 / 100;
    if (strong_idx < totalPixels) strongTargetThresh = sortedStrength[strong_idx];
    strongTargetThresh = max(strongTargetThresh, 0.08f);

    // 空间滤波：去除孤立噪点
    vector<float> filteredStrength = targetStrength;
    int half = 1;
    for (int y = half; y < height - half; y++) {
        for (int x = half; x < width - half; x++) {
            int idx = y * width + x;
            if (targetStrength[idx] > strongTargetThresh) {
                int neighbor_count = 0;
                for (int ky = -half; ky <= half; ky++) {
                    for (int kx = -half; kx <= half; kx++) {
                        int nidx = (y + ky) * width + (x + kx);
                        if (targetStrength[nidx] > strongTargetThresh) neighbor_count++;
                    }
                }
                if (neighbor_count < 3) {
                    filteredStrength[idx] = 0.0f;
                }
            }
        }
    }
    targetStrength = filteredStrength;

    // 重新计算最终阈值
    sortedStrength = targetStrength;
    sort(sortedStrength.begin(), sortedStrength.end(), greater<float>());
    strongTargetThresh = 0.0f;
    strong_idx = totalPixels * 1 / 100;  // 前1%
    if (strong_idx < totalPixels) strongTargetThresh = sortedStrength[strong_idx];
    strongTargetThresh = max(strongTargetThresh, 0.10f);

    cout << "   各向异性阈值: " << min_anisotropy << ", 强目标阈值: " << strongTargetThresh << endl;

    // 统计各向异性分布
    int highAnisoCount = 0;
    for (int i = 0; i < totalPixels; i++) {
        if (anisotropy[i] > 0.5f && intensity[i] > 0.15f) highAnisoCount++;
    }
    cout << "   高各向异性点: " << highAnisoCount << " (" << 100.0 * highAnisoCount / totalPixels << "%)" << endl;

    // ========== 发光高亮合成 ==========
    cout << "5/5 生成发光高亮效果..." << endl;
    RGBImage* output = new RGBImage(width, height);

    int strongCount = 0;

    for (int i = 0; i < totalPixels; i++) {
        unsigned char gray_val = baseGray->data[i];
        unsigned char dark_gray = (unsigned char)(gray_val * 0.7f);

        if (targetStrength[i] > strongTargetThresh) {
            strongCount++;

            float strength_factor = min(1.0f, targetStrength[i] / strongTargetThresh);

            // 获取随机颜色倾向
            int x = i % width;
            int y = i / width;
            int seed = x * 65537 + y * 131071;
            float hue_tilt = ((seed >> 8) & 0xFF) / 255.0f;

            // 发光效果
            float brightness = 0.85f + strength_factor * 0.15f;
            float color_strength = 0.25f + strength_factor * 0.25f;

            float r_final, g_final, b_final;

            // 根据各向异性方向决定发光颜色（不同频率对应不同颜色）
            float hue = (originalHue[i] + hue_tilt * 0.3f);
            if (hue > 2.5f) hue -= 2.0f;

            if (hue < 0.8f) {
                // 红色/橙色发光（对应低频子孔径强）
                r_final = brightness;
                g_final = brightness * (0.5f + color_strength * 0.3f);
                b_final = brightness * (0.3f + color_strength * 0.2f);
            }
            else if (hue < 1.8f) {
                // 绿色/青色发光（对应中频子孔径强）
                r_final = brightness * (0.4f + color_strength * 0.3f);
                g_final = brightness;
                b_final = brightness * (0.5f + color_strength * 0.3f);
            }
            else {
                // 蓝色/紫色发光（对应高频子孔径强）
                r_final = brightness * (0.5f + color_strength * 0.3f);
                g_final = brightness * (0.4f + color_strength * 0.2f);
                b_final = brightness;
            }

            output->r[i] = (unsigned char)(r_final * 255);
            output->g[i] = (unsigned char)(g_final * 255);
            output->b[i] = (unsigned char)(b_final * 255);
        }
        else {
            // 非目标区域：纯灰度
            output->r[i] = dark_gray;
            output->g[i] = dark_gray;
            output->b[i] = dark_gray;
        }
    }

    cout << "   强目标=" << strongCount << " (" << 100.0 * strongCount / totalPixels << "%)" << endl;

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
            cout << "1. 测试区域模式 (2048x2048) - 快速处理" << endl;
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