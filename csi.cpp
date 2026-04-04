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

// 快速伪随机函数，用于制造单视散斑的颗粒感
inline float get_jitter(int x, int y) {
    float dot = x * 12.9898f + y * 78.233f;
    float sn = fmod(sin(dot) * 43758.5453f, 1.0f);
    return (sn < 0) ? sn + 1.0f : sn;
}

RGBImage* processCSI(Complex* input, int width, int height, const BandConfig& config) {
    cout << "\n执行：线性背景还原 + 高密度荧光绿爆破 (修正版)..." << endl;
    int totalPixels = width * height;
    RGBImage* output = new RGBImage(width, height);

    // 1. 频谱变换
    fftwf_complex* in = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
    fftwf_complex* spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
    for (int i = 0; i < totalPixels; i++) {
        in[i][0] = input[i].real(); in[i][1] = input[i].imag();
    }
    fftwf_plan p_for = fftwf_plan_dft_2d(height, width, in, spec, FFTW_FORWARD, FFTW_ESTIMATE);
    fftwf_execute(p_for);

    // 2. 物理多视提取 (9视物理底图)
    auto getPureMultiLook = [&](int channelIdx) -> vector<float> {
        const int K = 3;
        vector<float> channelMag(totalPixels, 0.0f);
        for (int k = 0; k < K; k++) {
            fftwf_complex* sub_spec = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
            memset(sub_spec, 0, sizeof(fftwf_complex) * totalPixels);
            int subIdx = channelIdx * K + k;
            float center = 0.2f + subIdx * (0.6f / 8.0f);
            for (int y = 0; y < height; y++) {
                float rel_y = (float)y / height;
                float weight = max(0.0f, (float)pow(cos(3.14159f * (rel_y - center) * 4.0f), 2));
                for (int x = 0; x < width; x++) {
                    int idx = y * width + x;
                    sub_spec[idx][0] = spec[idx][0] * weight;
                    sub_spec[idx][1] = spec[idx][1] * weight;
                }
            }
            fftwf_complex* out_c = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * totalPixels);
            fftwf_plan p_b = fftwf_plan_dft_2d(height, width, sub_spec, out_c, FFTW_BACKWARD, FFTW_ESTIMATE);
            fftwf_execute(p_b);
            for (int i = 0; i < totalPixels; i++)
                channelMag[i] += sqrt(out_c[i][0] * out_c[i][0] + out_c[i][1] * out_c[i][1]);
            fftwf_destroy_plan(p_b); fftwf_free(sub_spec); fftwf_free(out_c);
        }
        for (int i = 0; i < totalPixels; i++) channelMag[i] /= (float)K;
        return channelMag;
        };

    vector<float> mR = getPureMultiLook(0), mG = getPureMultiLook(1), mB = getPureMultiLook(2);

    // 3. 统计能量 (确定映射门限)
    double sum = 0, sq_sum = 0;
    for (int i = 0; i < totalPixels; i++) {
        float a = (mR[i] + mG[i] + mB[i]) / 3.0f;
        sum += a; sq_sum += (double)a * a;
    }
    float mu = (float)(sum / totalPixels), sig = (float)sqrt(sq_sum / totalPixels - (double)mu * mu);
    float hi = mu + 4.8f * sig;

    // 4. 定向增强逻辑
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y * width + x;
            float r = mR[i], g = mG[i], b = mB[i];
            float avg = (r + g + b) / 3.0f;

            // A. 背景线性保真：完全不抑制，保住地形所有灰度细节
            float L = (avg / (hi + 1e-6f)) * 255.0f;

            float fr, fg, fb;
            float threshold = 155.0f;

            if (L > threshold) {
                // B. 高光区单色绿化爆破
                float W = (L - threshold) / (255.0f - threshold);

                // --- 强化策略：收紧随机抖动范围 (0.88 - 1.12) 让色彩更密 ---
                float jitter = 0.88f + get_jitter(x, y) * 0.24f;

                // 核心：基于 L 的非线性绿色拉升
                // 使用 (2.5 + W * 4.5) 极大化增益，配合 jitter 制造颗粒感
                fg = L + (avg * config.colorEnhance * (2.5f + W * 4.5f) * jitter);

                // 二次荧光爆破曲线
                float glow = 1.0f + pow(W, 2.0f) * 1.8f;
                fg *= glow;

                // 彻底压低 R 和 B，防止变白，保持纯净绿色
                fr = L * (0.12f - W * 0.1f);
                fb = L * (0.12f - W * 0.1f);

                // 最终亮度加成
                float final_boost = 1.3f + W * 0.9f;
                fr *= final_boost; fg *= final_boost; fb *= final_boost;
            }
            else {
                // C. 背景区：维持原始物理色比，线性输出
                float scale = L / (avg + 1e-6f);
                fr = r * scale; fg = g * scale; fb = b * scale;
            }

            auto finalize = [&](float v) {
                return (unsigned char)max(0.0f, min(255.0f, pow(v / 255.0f, config.gamma) * 255.0f));
                };

            output->r[i] = finalize(fr);
            output->g[i] = finalize(fg);
            output->b[i] = finalize(fb);
        }
    }

    fftwf_destroy_plan(p_for); fftwf_free(in); fftwf_free(spec);
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

            cout << "\n√ 处理完成！结果已保存至程序文件夹内！"  << endl;

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