#include <matio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <fstream>
#include <iomanip>

namespace fs = std::filesystem;

struct Exceedance {
    std::string filename;
    int frame_index;
    double movement_score;
    double vthr;
    double difference;
};

std::vector<fs::path> find_mat_files(const fs::path &folder) {
    std::vector<fs::path> files;

    if (!fs::exists(folder)) {
        throw std::runtime_error("Folder does not exist: " + folder.string());
    }

    for (const auto &entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".mat") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

struct RGBImage {
    std::vector<std::vector<unsigned char>> red;
    std::vector<std::vector<unsigned char>> green;
    std::vector<std::vector<unsigned char>> blue;
};

RGBImage read_rgb_image(const fs::path &file) {
    mat_t *mat = Mat_Open(file.string().c_str(), MAT_ACC_RDONLY);

    if (!mat) {
        throw std::runtime_error("Cannot open " + file.string());
    }

    matvar_t *res = Mat_VarRead(mat, "Res");

    if (!res || res->class_type != MAT_C_STRUCT) {
        if (res) Mat_VarFree(res);
        Mat_Close(mat);
        throw std::runtime_error("'Res' structure not found");
    }

    matvar_t *s = Mat_VarGetStructFieldByName(res, "S", 0);

    if (!s || s->class_type != MAT_C_DOUBLE ||
        s->rank != 3 || s->dims[2] != 3 || !s->data) {
        Mat_VarFree(res);
        Mat_Close(mat);
        throw std::runtime_error("Res.S is not a 3-channel RGB image");
    }

    const size_t rows = s->dims[0];     
    const size_t columns = s->dims[1];   
    const auto *data = static_cast<const double *>(s->data);

    double maximum = 0.0;
    for (size_t i = 0; i < rows * columns * 3; ++i) {
        maximum = std::max(maximum, data[i]);
    }
    const double scale = maximum <= 1.0 ? 255.0 : 1.0;

    RGBImage image{
        std::vector<std::vector<unsigned char>>(
            rows, std::vector<unsigned char>(columns)),
        std::vector<std::vector<unsigned char>>(
            rows, std::vector<unsigned char>(columns)),
        std::vector<std::vector<unsigned char>>(
            rows, std::vector<unsigned char>(columns))
    };

    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            const size_t pixel = row + rows * column;

            image.red[row][column] =
                static_cast<unsigned char>(std::clamp(data[pixel] * scale, 0.0, 255.0));

            image.green[row][column] =
                static_cast<unsigned char>(std::clamp(data[pixel + rows * columns] * scale, 0.0, 255.0));

            image.blue[row][column] =
                static_cast<unsigned char>(std::clamp(data[pixel + 2 * rows * columns] * scale, 0.0, 255.0));
        }
    }

    Mat_VarFree(res);
    Mat_Close(mat);
    return image;
}

double extract_vthr(const fs::path &file) {
    mat_t *mat = Mat_Open(file.string().c_str(), MAT_ACC_RDONLY);
    if (!mat) return 0.0;
    
    matvar_t *ini = Mat_VarRead(mat, "Ini");
    if (!ini) {
        Mat_Close(mat);
        return 0.0;
    }
    
    matvar_t *vthr_field = Mat_VarGetStructFieldByName(ini, "VThr", 0);
    double vthr = 0.0;
    if (vthr_field && vthr_field->data) {
        vthr = *static_cast<double*>(vthr_field->data);
    }
    
    Mat_VarFree(ini);
    Mat_Close(mat);
    return vthr;
}

std::string extract_filename_from_ini(const fs::path &file) {
    mat_t *mat = Mat_Open(file.string().c_str(), MAT_ACC_RDONLY);
    if (!mat) return "";
    
    matvar_t *ini = Mat_VarRead(mat, "Ini");
    if (!ini) {
        Mat_Close(mat);
        return "";
    }
    
    matvar_t *ff_field = Mat_VarGetStructFieldByName(ini, "ff", 0);
    std::string filename = "";
    if (ff_field && ff_field->data) {
        char* str = (char*)ff_field->data;
        filename = std::string(str);
    }
    
    Mat_VarFree(ini);
    Mat_Close(mat);
    return filename;
}

double compute_movement(const cv::Mat& prev_frame, const cv::Mat& curr_frame) {
    cv::Mat prev_gray, curr_gray;
    cv::cvtColor(prev_frame, prev_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat diff;
    cv::absdiff(prev_gray, curr_gray, diff);
    
    cv::Scalar mean_diff = cv::mean(diff);
    
    return mean_diff[0] / 255.0;
}

cv::Mat rgbimage_to_cvmat(const RGBImage& image) {
    int height = image.red.size();
    int width = image.red[0].size();
    
    cv::Mat frame(height, width, CV_8UC3);
    
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            frame.at<cv::Vec3b>(row, col) = cv::Vec3b(
                image.blue[row][col],
                image.green[row][col],
                image.red[row][col]
            );
        }
    }
    
    return frame;
}

cv::Mat add_overlay(const cv::Mat& frame, double movement, double vthr, int frame_num, bool exceeded) {
    cv::Mat overlay = frame.clone();
    
    cv::rectangle(overlay, cv::Point(10, 10), cv::Point(500, 100), cv::Scalar(0, 0, 0), -1);
    cv::addWeighted(overlay, 0.6, frame, 0.4, 0, overlay);
    
    std::string text1 = "Frame: " + std::to_string(frame_num);
    cv::putText(overlay, text1, cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    std::string text2 = "Movement: " + std::to_string(movement).substr(0, 6);
    cv::putText(overlay, text2, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    std::string text3 = "VThr: " + std::to_string(vthr).substr(0, 6);
    cv::putText(overlay, text3, cv::Point(20, 85), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);
    
    cv::Scalar status_color = exceeded ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    std::string status = exceeded ? "EXCEEDED!" : "OK";
    cv::putText(overlay, status, cv::Point(400, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2);
    
    return overlay;
}


void inspect_ini(const std::string &file) {
    mat_t *mat = Mat_Open(file.c_str(), MAT_ACC_RDONLY);
    matvar_t *ini = Mat_VarRead(mat, "Ini");

    auto names = Mat_VarGetStructFieldnames(ini);
    const unsigned count = Mat_VarGetNumberOfFields(ini);

    for (unsigned i = 0; i < count; ++i) {
        matvar_t *field = Mat_VarGetStructFieldByIndex(ini, i, 0);

        std::cout << i << ": " << names[i];

        if (field && field->class_type == MAT_C_DOUBLE &&
            field->data && Mat_VarGetSize(field) == 1) {
            std::cout << " = "
                      << *static_cast<double *>(field->data);
        }

        std::cout << '\n';
    }

    Mat_VarFree(ini);
    Mat_Close(mat);
}

void print_exceedance_report(const std::vector<Exceedance>& exceedances, int total_frames) {
    std::cout << "\n" << std::string(70, '=') << std::endl;
    std::cout << "THRESHOLD EXCEEDANCE REPORT" << std::endl;
    std::cout << std::string(70, '=') << std::endl;
    
    std::cout << "Total frames processed: " << total_frames << std::endl;
    std::cout << "Total exceedances: " << exceedances.size() << std::endl;
    
    if (total_frames > 0) {
        std::cout << "Percentage: " << std::fixed << std::setprecision(2) 
                  << (100.0 * exceedances.size() / total_frames) << "%" << std::endl;
    }
    
    if (exceedances.empty()) {
        std::cout << "\nNo threshold exceedances found!" << std::endl;
        return;
    }
    
    auto max_it = std::max_element(exceedances.begin(), exceedances.end(),
        [](const Exceedance& a, const Exceedance& b) {
            return a.difference < b.difference;
        });
    
    std::cout << "\nMaximum exceedance:" << std::endl;
    std::cout << "  File: " << max_it->filename << std::endl;
    std::cout << "  Frame: " << max_it->frame_index << std::endl;
    std::cout << "  Movement: " << max_it->movement_score << std::endl;
    std::cout << "  VThr: " << max_it->vthr << std::endl;
    std::cout << "  Difference: " << max_it->difference << std::endl;
    
    std::cout << "\nFirst 20 exceedances:" << std::endl;
    std::cout << "Frame\tFile\t\t\tMovement\tVThr\t\tDiff" << std::endl;
    std::cout << "--------------------------------------------------------" << std::endl;
    
    int count = std::min(20, (int)exceedances.size());
    for (int i = 0; i < count; i++) {
        const auto& e = exceedances[i];
        std::cout << e.frame_index << "\t" 
                  << e.filename.substr(0, 20) << "\t"
                  << std::fixed << std::setprecision(4) << e.movement_score << "\t"
                  << e.vthr << "\t\t"
                  << e.difference << std::endl;
    }
    
    if (exceedances.size() > 20) {
        std::cout << "... and " << exceedances.size() - 20 << " more" << std::endl;
    }
    std::cout << std::string(70, '=') << std::endl;
}

void save_exceedances_to_csv(const std::vector<Exceedance>& exceedances, const std::string& filename = "exceedances.csv") {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        std::cerr << "Could not create " << filename << std::endl;
        return;
    }
    
    csv << "File,Frame,Movement_Score,VThr,Difference\n";
    for (const auto& e : exceedances) {
        csv << e.filename << ","
            << e.frame_index << ","
            << e.movement_score << ","
            << e.vthr << ","
            << e.difference << "\n";
    }
    csv.close();
    std::cout << "\nExceedances saved to " << filename << std::endl;
}


int main() {
    try {
        const std::string data_path = "/Users/simeonstamboliyski/Documents/GitHub/GATE-2026/"
                                     "detector-data-exploratory-analysis/detector_dataset";
        
        const auto files = find_mat_files(data_path);

        if (files.empty()) {
            std::cerr << "No .mat files found.\n";
            return 1;
        }

        std::cout << "Found " << files.size() << " .mat files\n";
        
        std::vector<Exceedance> exceedances;
        int total_frames = 0;
        
        cv::Mat prev_frame;
        bool first_frame = true;
        int global_frame_counter = 0;
        
        for (const auto &file : files) {
            std::cout << "\nProcessing: " << file.filename().string() << std::endl;
            
            double vthr = extract_vthr(file);
            std::string video_filename = extract_filename_from_ini(file);
            std::cout << "  VThr = " << vthr << std::endl;
            std::cout << "  Video: " << video_filename << std::endl;
            
            auto image = read_rgb_image(file);
            cv::Mat curr_frame = rgbimage_to_cvmat(image);
            
            if (!first_frame) {
                double movement_score = compute_movement(prev_frame, curr_frame);
                
                if (movement_score > vthr) {
                    Exceedance e;
                    e.filename = file.filename().string();
                    e.frame_index = global_frame_counter;
                    e.movement_score = movement_score;
                    e.vthr = vthr;
                    e.difference = movement_score - vthr;
                    exceedances.push_back(e);
                    
                    std::cout << "  >> EXCEEDANCE at frame " << global_frame_counter 
                             << ": movement=" << std::fixed << std::setprecision(4) 
                             << movement_score << " > VThr=" << vthr 
                             << " (diff=" << movement_score - vthr << ")" << std::endl;
                }
            }
            
            prev_frame = curr_frame.clone();
            first_frame = false;
            global_frame_counter++;
            total_frames++;
        }
        
        print_exceedance_report(exceedances, total_frames);
        save_exceedances_to_csv(exceedances);

        const auto first_image = read_rgb_image(files.front());
        const int height = static_cast<int>(first_image.red.size());
        const int width = static_cast<int>(first_image.red[0].size());

        const double fps = 1.0;
        cv::VideoWriter video(
            "detector_video_with_analysis.mp4",
            cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
            fps,
            cv::Size(width, height)
        );

        if (!video.isOpened()) {
            throw std::runtime_error("Could not create detector_video_with_analysis.mp4");
        }

        std::set<int> exceedance_frames;
        for (const auto& e : exceedances) {
            exceedance_frames.insert(e.frame_index);
        }

        first_frame = true;
        int frame_idx = 0;
        
        for (const auto &file : files) {
            const auto image = read_rgb_image(file);
            cv::Mat frame = rgbimage_to_cvmat(image);
            
            double vthr = extract_vthr(file);
            
            double movement = 0.0;
            bool exceeded = false;
            
            if (!first_frame) {
                movement = compute_movement(prev_frame, frame);
                exceeded = (movement > vthr);
            }
            
            cv::Mat frame_with_overlay = add_overlay(frame, movement, vthr, frame_idx, exceeded);
            
            video.write(frame_with_overlay);
            std::cout << "Added frame " << frame_idx << " from " << file.filename().string() 
                      << (exceeded ? " [EXCEEDED!]" : "") << std::endl;
            
            prev_frame = frame.clone();
            first_frame = false;
            frame_idx++;
        }

        video.release();
        std::cout << "\nVideo saved as: detector_video_with_analysis.mp4" << std::endl;

        inspect_ini(data_path + "/0132_20170414_2216-20170414_2230.mat");

    } catch (const std::exception &error) {
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }

    return 0;
}