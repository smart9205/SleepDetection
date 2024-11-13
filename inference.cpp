#define STB_IMAGE_IMPLEMENTATION
#include "./stb/stb_image.h"
#include "./stb/drawing.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "./stb/stb_image_write.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "./stb/stb_image_resize.h"
static const uint8_t color[3] = {0xff, 0, 0};

// #include "inference_nv12.h"
#include "venus.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include <memory.h>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <cstring>
#include "debug.h"
#define IS_ALIGN_64(x) (((size_t)x) & 0x3F)

#define FACE_MODEL_PATH "./model/face.bin"
#define LANDMARK_MODEL_PATH "./model/landmark.bin"

using namespace std;
using namespace magik::venus;



typedef struct
{
    unsigned char* image;  
    int w;
    int h;
}input_info_t;


struct PixelOffset {
    int top;
    int bottom;
    int left;
    int right;
};

void check_pixel_offset(PixelOffset &pixel_offset){
    // 5 5 -> 6 4
    // padding size not is Odd number
    if(pixel_offset.top % 2 == 1){
        pixel_offset.top += 1;
        pixel_offset.bottom -=1;
    }
    if(pixel_offset.left % 2 == 1){
        pixel_offset.left += 1;
        pixel_offset.right -=1;
    }
}

uint8_t* read_bin(const char* path)
{
    std::ifstream infile;
    infile.open(path, std::ios::binary | std::ios::in);
    infile.seekg(0, std::ios::end);
    int length = infile.tellg();
    infile.seekg(0, std::ios::beg);
    uint8_t* buffer_pointer = new uint8_t[length];
    infile.read((char*)buffer_pointer, length);
    infile.close();
    return buffer_pointer;
}

// float* read_bin(const char* path)
// {
//     std::ifstream infile;
//     infile.open(path, std::ios::binary | std::ios::in);
//     infile.seekg(0, std::ios::end);
//     int length = infile.tellg();
//     printf("length:%d\n",length);
//     infile.seekg(0, std::ios::beg);
//     float* buffer_pointer = new float[(length)];
//     infile.read((char*)buffer_pointer, length);
//     infile.close();
//     return buffer_pointer;
// }

std::vector<std::string> splitString(std::string srcStr, std::string delimStr,bool repeatedCharIgnored = false)
{
    std::vector<std::string> resultStringVector;
    std::replace_if(srcStr.begin(), srcStr.end(), [&](const char& c){if(delimStr.find(c)!=std::string::npos){return true;}else{return false;}}, delimStr.at(0));
    size_t pos=srcStr.find(delimStr.at(0));
    std::string addedString="";
    while (pos!=std::string::npos) {
        addedString=srcStr.substr(0,pos);
        if (!addedString.empty()||!repeatedCharIgnored) {
            resultStringVector.push_back(addedString);
        }
        srcStr.erase(srcStr.begin(), srcStr.begin()+pos+1);
        pos=srcStr.find(delimStr.at(0));
    }
    addedString=srcStr;
    if (!addedString.empty()||!repeatedCharIgnored) {
        resultStringVector.push_back(addedString);
    }
    return resultStringVector;
}

void write_input_bin(std::unique_ptr<const venus::Tensor>& tensor, std::string name = "mnn.bin") {
	std::ofstream outFile(name, std::ios::out | std::ios::binary);
	auto shape = tensor->shape();
    std::cout << "input shape: " << std::endl;
	int size = 1;
	for(auto s : shape) {
        std::cout << s << ",";
		size *= s;
	}
    std::cout << std::endl;
	const uint8_t *data = tensor->data<uint8_t>();
	outFile.write((char*)data, sizeof(uint8_t)*size);
    outFile.close();
}

void write_output_bin(std::unique_ptr<const venus::Tensor>& tensor, std::string name = "mnn.bin") {
	std::ofstream outFile(name, std::ios::out | std::ios::binary);
	auto shape = tensor->shape();
    std::cout << "output shape: " << std::endl;
	int size = 1;
	for(auto s : shape) {
        std::cout << s << ",";
		size *= s;
	}
    std::cout << std::endl;
	const float *data = tensor->data<float>();
	outFile.write((char*)data, sizeof(float)*size);
    outFile.close();

    // int len = std::min(100, size);
    // for(int i = 0; i < len; i++){
    //     printf("%f, ", data[i]);
    // }
    // printf("\n");
}

vector<vector<float>> min_boxes = {{10.0, 16.0, 24.0}, {32.0, 48.0}, {64.0, 96.0}, {128.0, 192.0, 256.0}};
vector<float> strides = {8.0, 16.0, 32.0, 64.0};

vector<vector<float>> generate_priors(const vector<vector<int>>& feature_map_list, const vector<vector<float>>& shrinkage_list, const vector<int>& image_size, const vector<vector<float>>& min_boxes) {
    vector<vector<float>> priors;
    for (size_t index = 0; index < feature_map_list[0].size(); ++index) {
        float scale_w = image_size[0] / shrinkage_list[0][index];
        float scale_h = image_size[1] / shrinkage_list[1][index];
        for (int j = 0; j < feature_map_list[1][index]; ++j) {
            for (int i = 0; i < feature_map_list[0][index]; ++i) {
                float x_center = (i + 0.5) / scale_w;
                float y_center = (j + 0.5) / scale_h;

                for (float min_box : min_boxes[index]) {
                    float w = min_box / image_size[0];
                    float h = min_box / image_size[1];
                    priors.push_back({x_center, y_center, w, h});
                }
            }
        }
    }
    // cout << "priors nums:" << priors.size() << endl;
    // Clipping the priors to be within [0.0, 1.0]
    for (auto& prior : priors) {  
        for (auto& val : prior) {  
            val = std::min(std::max(val, 0.0f), 1.0f);  
        }  
    }  

    return priors;
}

vector<vector<float>> define_img_size(const vector<int>& image_size) {
    vector<vector<int>> feature_map_w_h_list;
    vector<vector<float>> shrinkage_list;
    for (int size : image_size) {
        vector<int> feature_map;
        for (float stride : strides) {
            feature_map.push_back(static_cast<int>(ceil(size / stride)));
        }
        feature_map_w_h_list.push_back(feature_map);
    }

    for (size_t i = 0; i < image_size.size(); ++i) {
        shrinkage_list.push_back(strides);
    }
    return generate_priors(feature_map_w_h_list, shrinkage_list, image_size, min_boxes);
}

vector<vector<float>> convert_locations_to_boxes(const vector<vector<float>>& locations, const vector<vector<float>>& priors, float center_variance, float size_variance) {
    vector<vector<float>> boxes;
    for (size_t i = 0; i < locations.size(); ++i) {
        vector<float> box;
        for (size_t j = 0; j < locations[i].size() / 4; ++j) {
            float cx = locations[i][j * 4 + 0] * center_variance * priors[i][2] + priors[i][0];
            float cy = locations[i][j * 4 + 1] * center_variance * priors[i][3] + priors[i][1];
            float w = exp(locations[i][j * 4 + 2] * size_variance) * priors[i][2];
            float h = exp(locations[i][j * 4 + 3] * size_variance) * priors[i][3];
            box.push_back(cx);
            box.push_back(cy);
            box.push_back(w);
            box.push_back(h);
        }
        boxes.push_back(box);
    }
    return boxes;
}

vector<vector<float>> center_form_to_corner_form(const vector<vector<float>>& locations) {
    vector<vector<float>> boxes;
    for (const auto& loc : locations) {
        vector<float> box;
        for (size_t i = 0; i < loc.size() / 4; ++i) {
            float cx = loc[i * 4 + 0];
            float cy = loc[i * 4 + 1];
            float w = loc[i * 4 + 2];
            float h = loc[i * 4 + 3];
            float xmin = cx - w / 2;
            float ymin = cy - h / 2;
            float xmax = cx + w / 2;
            float ymax = cy + h / 2;
            box.push_back(xmin);
            box.push_back(ymin);
            box.push_back(xmax);
            box.push_back(ymax);
        }
        boxes.push_back(box);
    }
    return boxes;
}

float area_of(float left, float top, float right, float bottom) {
    float width = max(0.0f, right - left);
    float height = max(0.0f, bottom - top);
    return width * height;
}

float iou_of(const vector<float>& box0, const vector<float>& box1) {
    float overlap_left = max(box0[0], box1[0]);
    float overlap_top = max(box0[1], box1[1]);
    float overlap_right = min(box0[2], box1[2]);
    float overlap_bottom = min(box0[3], box1[3]);

    float overlap_area = area_of(overlap_left, overlap_top, overlap_right, overlap_bottom);
    float area0 = area_of(box0[0], box0[1], box0[2], box0[3]);
    float area1 = area_of(box1[0], box1[1], box1[2], box1[3]);
    float total_area = area0 + area1 - overlap_area;
    if (total_area <= 0.0f) return 0.0f;
    return overlap_area / total_area;
}

vector<vector<float>> hard_nms(const vector<vector<float>>& box_scores, float iou_threshold, int top_k = -1, int candidate_size = 200) {
    vector<int> idx(box_scores.size());
    iota(idx.begin(), idx.end(), 0);

    sort(idx.begin(), idx.end(), [&box_scores](int i1, int i2) {
        return box_scores[i1].back() < box_scores[i2].back();
    });

    if (candidate_size > 0 && candidate_size < (int)idx.size()) {
        idx.resize(candidate_size);
    }

    vector<vector<float>> picked;
    while (!idx.empty()) {
        int current = idx.back();
        const auto& current_box = box_scores[current];
        picked.push_back(current_box);
        if (top_k > 0 && (int)picked.size() >= top_k) break;
        idx.pop_back();

        for (auto it = idx.begin(); it != idx.end();) {
            float iou = iou_of(box_scores[*it], current_box);
            if (iou > iou_threshold) {
                it = idx.erase(it);
            } else {
                ++it;
            }
        }
    }
    return picked;
}

vector<vector<float>> predict(float width, float height, const vector<vector<float>>& scores, const vector<vector<float>>& boxes, float prob_threshold, float iou_threshold = 0.3, int top_k = -1) {
    vector<vector<float>> final_boxes;
    vector<vector<float>> box_scores; // Combine boxes and scores in the required format
    for (size_t i = 0; i < boxes.size(); ++i) {
        vector<float> box_score = boxes[i];
        box_score.push_back(scores[i][1]); // Assuming class score is at index 1
        if (scores[i][1] > prob_threshold) {
            box_scores.push_back(box_score);
        }
    }
    
    vector<vector<float>> picked = hard_nms(box_scores, iou_threshold, top_k);

    // Convert coordinates back to original scale and print
    for (const auto& box : picked) {
        cout << "Box: ";
        vector<float> face_box;
        for (size_t i = 0; i < 4; ++i) {
            float coord = i % 2 == 0 ? box[i] * width : box[i] * height;
            face_box.push_back((int)coord);
            cout << coord << " ";
        }
        final_boxes.push_back(face_box);
        cout << "Score: " << box.back() << endl;
    }
    return final_boxes;
}

void softmax(const float* input, float* output, int w, int h, int c) {
    const float* in_data = input;
    int first = h;
    int second = c;
    int third = w;

    int softmax_size = w * h;
    float* softmax_data = (float*)malloc(softmax_size * sizeof(float));
    float* max = (float*)malloc(softmax_size * sizeof(float));
    for (int f = 0; f < first; ++f) {
        for (int t = 0; t < third; ++t) {
            int m_under = f * third + t;
            max[m_under] = -FLT_MAX;
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                max[m_under] = in_data[i_under] > max[m_under] ? in_data[i_under] : max[m_under];
            }
            softmax_data[m_under] = 0;
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                float temp = in_data[i_under];
                softmax_data[m_under] += exp(temp - max[m_under]);
            }
            for (int s = 0; s < second; ++s) {
                int i_under = f * third * second + s * third + t;
                float input_num = in_data[i_under];
                float softmax_num = exp(input_num - max[m_under]) / softmax_data[m_under];
                output[i_under] = softmax_num;
            }
        }
    }
}


float euclidean_distance(const std::array<float, 2>& p1, const std::array<float, 2>& p2) {  
    return std::sqrt(std::pow(p1[0] - p2[0], 2) + std::pow(p1[1] - p2[1], 2));  
}  
  
float calculate_EAR(const std::vector<std::array<float, 2>>& eye) {  
    float A = euclidean_distance(eye[1], eye[5]);  
    float B = euclidean_distance(eye[2], eye[4]);  
    float C = euclidean_distance(eye[0], eye[3]);  
  
    float ear = (A + B) / (2.0 * C);  
    return ear;  
}  

// Custom clamp function  
template <typename T>  
T clamp(T value, T min, T max) {  
    return std::max(min, std::min(value, max));  
}  
  

int main(int argc, char **argv) {
    int ret = 0;
    /* set cpu affinity */
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(0, &mask);
    if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
        fprintf(stderr, "set cpu affinity failed, %s\n", strerror(errno));
        return -1;
    }

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        exit(0);
    }

    ret = venus::venus_init();
    if (0 != ret) {
        fprintf(stderr, "venus init failed.\n");
        return ret;
    }

    void *handle = NULL;

    int ori_img_h = -1;
    int ori_img_w = -1;
    float scale = 1.0;
    int face_in_w = 320, face_in_h = 240;
    int emo_in_w = 112, emo_in_h = 112;

    PixelOffset pixel_offset;
    std::unique_ptr<venus::Tensor> input;
    std::unique_ptr<venus::Tensor> landmark_input;

    std::unique_ptr<venus::BaseNet> face_net = venus::net_create(venus::TensorFormat::NHWC, venus::ShareMemoryMode::ALL_SEPARABLE_MEM);
    std::unique_ptr<venus::BaseNet> landmark_net = venus::net_create(venus::TensorFormat::NHWC, venus::ShareMemoryMode::ALL_SEPARABLE_MEM);
	
    ret = face_net->load_model(FACE_MODEL_PATH);
    ret = landmark_net->load_model(LANDMARK_MODEL_PATH);

    int comp;  
  
    // Load the image as RGB  
    unsigned char* imagedata = stbi_load(argv[1], &ori_img_w, &ori_img_h, &comp, 3);  
    if (imagedata == nullptr) {  
        std::cerr << "Error loading image\n";  
        return 1;  
    }  
    printf("Original image width, height: %d %d \n", ori_img_w, ori_img_h);  

    magik::venus::shape_t temp_inshape;
    temp_inshape.push_back(1);
    temp_inshape.push_back(ori_img_h);
    temp_inshape.push_back(ori_img_w);
    temp_inshape.push_back(4);
    venus::Tensor input_tensor(temp_inshape);
    uint8_t *temp_indata = input_tensor.mudata<uint8_t>();

    for (int i = 0; i < ori_img_h; i ++)
	{
		for (int j = 0; j < ori_img_w; j++)
		{
			temp_indata[i*ori_img_w*4 + j*4 + 0] = imagedata[i*ori_img_w*3 + j*3 + 2];
			temp_indata[i*ori_img_w*4 + j*4 + 1] = imagedata[i*ori_img_w*3 + j*3 + 1];
			temp_indata[i*ori_img_w*4 + j*4 + 2] = imagedata[i*ori_img_w*3 + j*3 + 0];
			temp_indata[i*ori_img_w*4 + j*4 + 3] = 0;
		}
	}
    
    input = face_net->get_input(0);
    magik::venus::shape_t input_shape = input->shape();
    printf("face model-->%d %d %d \n",input_shape[1], input_shape[2], input_shape[3]);
    input->reshape({1, face_in_h, face_in_w , 4});
  

    magik::venus::BsExtendParam param;
    param.pad_val = 0;
    param.pad_type = magik::venus::BsPadType::SYMMETRY;
    param.in_layout = magik::venus::ChannelLayout::RGBA;
    param.out_layout = magik::venus::ChannelLayout::RGBA;
    warp_resize(input_tensor, *input, &param);
   
    face_net->run();


    std::unique_ptr<const venus::Tensor> out_0 = face_net->get_output(0); 
    std::unique_ptr<const venus::Tensor> out_1 = face_net->get_output(1);

    const float* output_data_0 = out_0->data<float>();
    const float* output_data_1 = out_1->data<float>();

    auto shape_0 = out_0->shape(); // scores
    auto shape_1 = out_1->shape(); // boxes

    int scores_size = shape_0[0]*shape_0[1]*shape_0[2]; // 1,4420,2
    int boxes_size  = shape_1[0]*shape_1[1]*shape_1[2]; // 1,4420,4,

    float* output_data_0_softmax = (float*)malloc(scores_size * sizeof(float));
    softmax(output_data_0, output_data_0_softmax, shape_0[0], shape_0[1], shape_0[2]);

    vector<vector<float>> scores;
    vector<vector<float>> boxes;

    // Assuming shape_0[1] == shape_1[1]: give the number of detections
    for (int i = 0; i < shape_0[1]; ++i) {
        // Extract scores
        vector<float> score;
        for (int j = 0; j < shape_0[2]; ++j) {
            score.push_back(output_data_0_softmax[i * shape_0[2] + j]);
        }
        scores.push_back(score);
  
        // Extract boxes
        vector<float> box;
        // Assuming shape_0[2] == 4, for [x1, y1, x2, y2]
        for (int k = 0; k < shape_1[2]; ++k) {
            box.push_back(output_data_1[i * shape_1[2] + k]);
        }
        boxes.push_back(box);
    }

    vector<int> input_size = {320, 240};
    float center_variance = 0.1;
    float size_variance = 0.2;
    float prob_threshold = 0.7;
    float iou_threshold = 0.3;
    int top_k = -1;

    vector<vector<float>> priors = define_img_size(input_size);
    
    vector<vector<float>> converted_boxes = convert_locations_to_boxes(boxes, priors, center_variance, size_variance);

    vector<vector<float>> final_boxes = center_form_to_corner_form(converted_boxes);

    vector<vector<float>> final_face_boxes = predict(ori_img_w, ori_img_h, scores, final_boxes, prob_threshold, iou_threshold, top_k);

    if (final_face_boxes.size() > 0){
        cout << final_face_boxes.size() << " Face Detected!!! " << endl;
        landmark_input = landmark_net->get_input(0);
        magik::venus::shape_t landmark_input_shape = landmark_input->shape();
        printf("landmark detection model-->%d %d %d \n",landmark_input_shape[1], landmark_input_shape[2], landmark_input_shape[3]);
    }

    int n_face = 1;      
    for (const auto& face_box : final_face_boxes) {  
        int roi_x = static_cast<int>(face_box[0]);  
        int roi_y = static_cast<int>(face_box[1]);  
        int roi_w = static_cast<int>(face_box[2] - face_box[0]);  
        int roi_h = static_cast<int>(face_box[3] - face_box[1]);  
  
        roi_x = std::max(roi_x, 0);  
        roi_y = std::max(roi_y, 0);  
        roi_w = std::min(roi_w, ori_img_w - roi_x);  
        roi_h = std::min(roi_h, ori_img_h - roi_y);  
  
        if (roi_w % 2 == 1) roi_w += 1;  
        if (roi_h % 2 == 1) roi_h += 1;  
  
        if (roi_w <= 0 || roi_h <= 0) {  
            std::cerr << "Invalid ROI dimensions after adjustment\n";  
            continue;  
        }  
  
        std::cout << "face_" << n_face << ": " << roi_x << " " << roi_y << " " << roi_w << " " << roi_h << std::endl;  
  
        // Use the RGB data directly for the detected face region  
        std::vector<uint8_t> faceData(roi_w * roi_h * 3);
        for (int y = 0; y < roi_h; ++y) {
            for (int x = 0; x < roi_w; ++x) {
                int srcIndex = ((roi_y + y) * ori_img_w + (roi_x + x)) * 3;
                int dstIndex = (y * roi_w + x) * 3;
                faceData[dstIndex + 0] = imagedata[srcIndex + 0]; // R
                faceData[dstIndex + 1] = imagedata[srcIndex + 1]; // G
                faceData[dstIndex + 2] = imagedata[srcIndex + 2]; // B
            }
        }
  
        // Convert RGB to RGBA  
        std::vector<uint8_t> rgbaData(roi_w * roi_h * 4);  
        for (int i = 0; i < roi_h * roi_w; ++i) {  
            rgbaData[i * 4 + 0] = faceData[i * 3 + 0]; // R  
            rgbaData[i * 4 + 1] = faceData[i * 3 + 1]; // G  
            rgbaData[i * 4 + 2] = faceData[i * 3 + 2]; // B  
            rgbaData[i * 4 + 3] = 0;                   // A  
        }  
  
        magik::venus::shape_t temp_inshape = {1, roi_h, roi_w, 4};  
        magik::venus::Tensor input_tensor(temp_inshape);  
        uint8_t* temp_indata = input_tensor.mudata<uint8_t>();  
        std::copy(rgbaData.begin(), rgbaData.end(), temp_indata);  
  
        // Get model input tensor and reshape
        // auto input = landmark_net->get_input(0);
        std::unique_ptr<venus::Tensor> input = landmark_net->get_input(0);
  
        // Resize using warp_resize  
        magik::venus::BsExtendParam param;  
        param.pad_val = 0;  
        param.pad_type = magik::venus::BsPadType::SYMMETRY;
        param.in_layout = magik::venus::ChannelLayout::RGBA;
        param.out_layout = magik::venus::ChannelLayout::RGBA;
  
        warp_resize(input_tensor, *input, &param);

        // debug_1108
        // uint8_t *indata = input->mudata<uint8_t>();
        // auto shape = input->shape(); // N H W C
        // uint8_t* imagedata_bin = read_bin("./data/cropped_face_112_112_3.bin");

        // int size = 1;
        // printf("shape:");
        // for(auto s : shape) {
        //     printf("%d, ", s);
        //     size *= s;
        // }
        // printf("\n");

        // int in_h = shape[1];
        // int in_w = shape[2];
        // for (int i = 0; i < in_h; i ++) {
        //     for (int j = 0; j < in_w; j++) {
        //         indata[i*in_w*4 + j*4 + 0] = imagedata_bin[i*in_w*3 + j*3 + 0];
        //         indata[i*in_w*4 + j*4 + 1] = imagedata_bin[i*in_w*3 + j*3 + 1];
        //         indata[i*in_w*4 + j*4 + 2] = imagedata_bin[i*in_w*3 + j*3 + 2];
        //         indata[i*in_w*4 + j*4 + 3] = 0;
        //     }
        // }

        // Run the model
        landmark_net->run();

        // std::vector<std::string> output_names = landmark_net->get_output_names();
        // for (int i = 0; i < int(output_names.size()); i++) {
        //     std::cout << "output_" << i << ": " << output_names[i] << std::endl;
        //     std::unique_ptr<const venus::Tensor> output_tensor = landmark_net->get_output(i);
        //     std::string output_name = output_names[i] + ".bin";
        //     write_output_bin(output_tensor, output_name);
        // }
  
        std::unique_ptr<const venus::Tensor> output = landmark_net->get_output(0);  
        const float* landmark_data = output->data<float>();  
  
        std::vector<std::array<float, 2>> landmarks;  
        int num_landmarks = output->shape()[3] / 2;  
        std::cout << "num_landmarks: " << num_landmarks << std::endl;  
  
        for (int i = 0; i < num_landmarks; ++i) {  
            landmarks.push_back({landmark_data[i * 2], landmark_data[i * 2 + 1]});  
        }  
  
        std::cout << "Number of landmarks detected: " << landmarks.size() << std::endl;  
  
        if (landmarks.size() < 68) {  
            std::cerr << "Not enough landmarks detected\n";  
            continue;  
        }  
  
        std::vector<std::array<float, 2>> left_eye_landmarks(landmarks.begin() + 36, landmarks.begin() + 42);  
        std::vector<std::array<float, 2>> right_eye_landmarks(landmarks.begin() + 42, landmarks.begin() + 48);  
  
        float left_EAR = calculate_EAR(left_eye_landmarks);  
        float right_EAR = calculate_EAR(right_eye_landmarks);  
        float avg_EAR = (left_EAR + right_EAR) / 2.0;  
  
        std::cout << "Average EAR: " << avg_EAR << std::endl;  
  
        n_face++;  
    }  
  
    ret = venus::venus_deinit();  
    if (0 != ret) {  
        fprintf(stderr, "venus deinit failed.\n");  
        return ret;  
    }  
    return 0;  
}

