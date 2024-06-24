#include "layer.h"
#include "net.h"

#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>
#include<iostream>
#include<cmath>

#define SSD_NMS_THRESH  0.45f
#define SSD_CONF_THRESH 0.25f
#define SSD_TARGET_SIZE 300  

struct Object
{
	cv::Rect_<float> rect;
	int label;
	float prob;
};


static inline float intersection_area(const Object& a, const Object& b)
{
	cv::Rect_<float> inter = a.rect & b.rect;
	return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
	int i = left;
	int j = right;
	float p = faceobjects[(left + right) / 2].prob;

	while (i <= j)
	{
		while (faceobjects[i].prob > p)
			i++;

		while (faceobjects[j].prob < p)
			j--;

		if (i <= j)
		{
			// swap
			std::swap(faceobjects[i], faceobjects[j]);

			i++;
			j--;
		}
	}

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (left < j) qsort_descent_inplace(faceobjects, left, j);
		}
#pragma omp section
		{
			if (i < right) qsort_descent_inplace(faceobjects, i, right);
		}
	}
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
	if (faceobjects.empty())
		return;

	qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
	picked.clear();

	const int n = faceobjects.size();

	std::vector<float> areas(n);
	for (int i = 0; i < n; i++)
	{
		areas[i] = faceobjects[i].rect.area();
	}

	for (int i = 0; i < n; i++)
	{
		const Object& a = faceobjects[i];

		int keep = 1;
		for (int j = 0; j < (int)picked.size(); j++)
		{
			const Object& b = faceobjects[picked[j]];

			if (!agnostic && a.label != b.label)
				continue;

			// intersection over union
			float inter_area = intersection_area(a, b);
			float union_area = areas[i] + areas[picked[j]] - inter_area;
			// float IoU = inter_area / union_area
			if (inter_area / union_area > nms_threshold)
				keep = 0;
		}

		if (keep)
			picked.push_back(i);
	}
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in, const ncnn::Mat& feat_bbox, const ncnn::Mat& feat_cls, float prob_threshold, std::vector<Object>& objects)
{
	const int num_grid_y = feat_bbox.c;
	const int num_grid_x = feat_bbox.h;
	const int num_bbox_out = feat_bbox.w;
	const int num_cls_out = feat_cls.w;

	const int num_anchors = anchors.w / 2;
	const int cls_walk = num_cls_out / num_anchors;
	const int num_class = cls_walk - 1; // ignore one foreground cls
	const int bbox_walk = num_bbox_out / num_anchors;

	const float wh_ratio_clip = 16 / 1000;
	const float max_ratio = abs(log(wh_ratio_clip));


	for (int i = 0; i < num_grid_y; i++)
	{
		for (int j = 0; j < num_grid_x; j++)
		{
			const float* feat_b = feat_bbox.channel(i).row(j); //bbox
			const float* feat_c = feat_cls.channel(i).row(j); //cls
			for (int k = 0; k < num_anchors; k++)
			{	
				//std::cout << feat_c[k] << std::endl;
				const float anchor_w = anchors[k * 2];
				const float anchor_h = anchors[k * 2 + 1];
				const float* featptr_b = feat_b + k * bbox_walk; //bbox
				const float* featptr_c = feat_c + k * cls_walk; //cls

				//std::cout << box_confidence << std::endl;
				int label = -1;
				float label_score = -FLT_MAX;
				for (int c = 0; c < num_class; c++)
				{
					float score = featptr_c[c];
					//std::cout << score << std::endl;
					///*
					if (score > prob_threshold && score > label_score)
					{
						label = c;
						label_score = score;
						float dx = featptr_b[0] * 0.1f; //std = [0.1,0.1,0.2,0.2] mean = 0 ignore
						float dy = featptr_b[1] * 0.1f;
						float dw = featptr_b[2] * 0.2f;
						float dh = featptr_b[3] * 0.2f;

						dw = std::max(std::min(dw, max_ratio), -max_ratio);
						dh = std::max(std::min(dh, max_ratio), -max_ratio);

						float center_anchor_x = (j + 0.5f) * stride; //prior anchor center_x
						float center_anchor_y = (i + 0.5f) * stride; //prior anchor center_y

						float pb_cx = center_anchor_x + anchor_w * dx; //final predict center_x
						float pb_cy = center_anchor_y + anchor_h * dy; //final predict center_y
						float pb_w = exp(dw) * anchor_w;
						float pb_h = exp(dh) * anchor_h;

						//top-left, bottom-right
						float x0 = pb_cx - pb_w * 0.5f;
						float y0 = pb_cy - pb_h * 0.5f;
						float x1 = pb_cx + pb_w * 0.5f;
						float y1 = pb_cy + pb_h * 0.5f;

						x0 = std::max(std::min(x0, (float)SSD_TARGET_SIZE), (float)0);
						y0 = std::max(std::min(y0, (float)SSD_TARGET_SIZE), (float)0);
						x1 = std::max(std::min(x1, (float)SSD_TARGET_SIZE), (float)0);
						y1 = std::max(std::min(y1, (float)SSD_TARGET_SIZE), (float)0);

						Object obj;
						obj.rect.x = x0;
						obj.rect.y = y0;
						obj.rect.width = x1 - x0;
						obj.rect.height = y1 - y0;
						obj.label = label;
						obj.prob = label_score;

						objects.push_back(obj);
					}
					//*/
				}
			}
		}
	}
}

static int detect_ssd(const cv::Mat& bgr, std::vector<Object>& objects)
{
	ncnn::Net ssd;

	//ssd.opt.use_vulkan_compute = true;
	//ssd.opt.use_bf16_storage = true;

	if (ssd.load_param("C:\\Users\\93414\\Desktop\\Project\\SSD\\SSD\\ssd_300_opt.param"))
		exit(-1);
	if (ssd.load_model("C:\\Users\\93414\\Desktop\\Project\\SSD\\SSD\\ssd_300_opt.bin"))
		exit(-1);

	const int target_size = SSD_TARGET_SIZE; //常类型，即不可改变
	const float prob_threshold = SSD_CONF_THRESH;
	const float nms_threshold = SSD_NMS_THRESH;

	int img_w = bgr.cols;
	int img_h = bgr.rows;

	float scale_x = (float)img_w / target_size;
	float scale_y = (float)img_h / target_size;

	ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, target_size, target_size);

	const float mean_vals[3] = { 123.675f, 116.28f, 103.53f };
	const float norm_vals[3] = { 1.f, 1.f, 1.f };
	in.substract_mean_normalize(mean_vals, norm_vals);

	ncnn::Extractor ex = ssd.create_extractor();

	ex.input("input.1", in);

	std::vector<Object> proposals;

	//anchor setting from mmdet/configs/ssd/ssd_300
	///*
	// stride 8
	{
		ncnn::Mat bbox_out, cls_out;
		ex.extract("171", cls_out);
		ex.extract("172", bbox_out);
		ncnn::Mat anchors(8);
		anchors[0] = 21.f;
		anchors[1] = 21.f;
		anchors[2] = 30.7408f;
		anchors[3] = 30.7408f;
		anchors[4] = 29.6984f;
		anchors[5] = 14.8492f;
		anchors[6] = 14.8492f;
		anchors[7] = 29.6984f;

		std::vector<Object> objects8;
		generate_proposals(anchors, 8, in, bbox_out, cls_out, prob_threshold, objects8);

		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
	}
	
	// stride 16
	{
		ncnn::Mat bbox_out, cls_out;
		ex.extract("193", cls_out);
		ex.extract("194", bbox_out);
		ncnn::Mat anchors(12);
		anchors[0] = 45.f;
		anchors[1] = 45.f;
		anchors[2] = 66.7458f;
		anchors[3] = 66.7458f;
		anchors[4] = 63.6396f;
		anchors[5] = 31.8198f;
		anchors[6] = 31.8198f;
		anchors[7] = 63.6396f;
		anchors[8] = 77.9422f;
		anchors[9] = 25.980800000000002f;
		anchors[10] = 25.980800000000002f;
		anchors[11] = 77.9422f;

		std::vector<Object> objects16;
		generate_proposals(anchors, 16, in, bbox_out, cls_out, prob_threshold, objects16);

		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
	}
	
	// stride 32
	{
		ncnn::Mat out;
		ncnn::Mat bbox_out, cls_out;
		ex.extract("215", cls_out);
		ex.extract("216", bbox_out);
		ncnn::Mat anchors(12);
		anchors[0] = 99.f;
		anchors[1] = 99.f;
		anchors[2] = 123.07320000000001f;
		anchors[3] = 123.07320000000001f;
		anchors[4] = 140.0072f;
		anchors[5] = 70.0036f;
		anchors[6] = 70.0036f;
		anchors[7] = 140.0072f;
		anchors[8] = 171.473f;
		anchors[9] = 57.1576f;
		anchors[10] = 57.1576f;
		anchors[11] = 171.473f;

		std::vector<Object> objects32;
		generate_proposals(anchors, 32, in, bbox_out, cls_out, prob_threshold, objects32);

		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
	}
	// stride 64
	{
		ncnn::Mat out;
		ncnn::Mat bbox_out, cls_out;
		ex.extract("237", cls_out);
		ex.extract("238", bbox_out);
		ncnn::Mat anchors(12);
		anchors[0] = 153.f;
		anchors[1] = 153.f;
		anchors[2] = 177.9634f;
		anchors[3] = 177.9634f;
		anchors[4] = 216.3746f;
		anchors[5] = 108.1874f;
		anchors[6] = 108.1874f;
		anchors[7] = 216.3746f;
		anchors[8] = 265.0038f;
		anchors[9] = 88.3346f;
		anchors[10] = 88.3346f;
		anchors[11] = 265.0038f;

		std::vector<Object> objects64;
		generate_proposals(anchors, 64, in, bbox_out, cls_out, prob_threshold, objects64);

		proposals.insert(proposals.end(), objects64.begin(), objects64.end());
	}
	// stride 100
	{
		ncnn::Mat out;
		ncnn::Mat bbox_out, cls_out;
		ex.extract("259", cls_out);
		ex.extract("260", bbox_out);
		ncnn::Mat anchors(8);
		anchors[0] = 207.f;
		anchors[1] = 207.f;
		anchors[2] = 232.437f;
		anchors[3] = 232.437f;
		anchors[4] = 292.7422f;
		anchors[5] = 146.371f;
		anchors[6] = 146.371f;
		anchors[7] = 292.7422f;

		std::vector<Object> objects100;
		generate_proposals(anchors, 100, in, bbox_out, cls_out, prob_threshold, objects100);

		proposals.insert(proposals.end(), objects100.begin(), objects100.end());
	}
	//*/
	
	// stride 300
	{
		ncnn::Mat bbox_out, cls_out;
		ex.extract("281", cls_out);
		ex.extract("282", bbox_out);
		ncnn::Mat anchors(8);
		anchors[0] = 261.f;
		anchors[1] = 261.f;
		anchors[2] = 286.73159999999996f;
		anchors[3] = 286.73159999999996f;
		anchors[4] = 369.10979999999995f;
		anchors[5] = 184.5548f;
		anchors[6] = 184.5548f;
		anchors[7] = 369.10979999999995f;

		std::vector<Object> objects300;
		generate_proposals(anchors, 300, in, bbox_out, cls_out, prob_threshold, objects300);

		proposals.insert(proposals.end(), objects300.begin(), objects300.end());
	}
	
	
	// sort all proposals by score from highest to lowest
	qsort_descent_inplace(proposals);

	// apply nms with nms_threshold
	std::vector<int> picked;
	nms_sorted_bboxes(proposals, picked, nms_threshold);

	int count = picked.size();

	objects.resize(count);
	for (int i = 0; i < count; i++)
	{
		objects[i] = proposals[picked[i]];

		// remap on original image
		float x0 = (objects[i].rect.x) * scale_x;
		float y0 = (objects[i].rect.y) * scale_y;
		float x1 = (objects[i].rect.x + objects[i].rect.width) * scale_x;
		float y1 = (objects[i].rect.y + objects[i].rect.height) * scale_y;

		// clip
		x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
		y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
		x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
		y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

		objects[i].rect.x = x0;
		objects[i].rect.y = y0;
		objects[i].rect.width = x1 - x0;
		objects[i].rect.height = y1 - y0;
	}

	return 0;
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
	static const char* class_names[] = {
		"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
		"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
		"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
		"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
		"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
		"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
		"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
		"microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
		"hair drier", "toothbrush"
	};

	cv::Mat image = bgr.clone();

	for (size_t i = 0; i < objects.size(); i++)
	{
		const Object& obj = objects[i];

		fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
			obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

		cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

		char text[256];
		sprintf_s(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

		int baseLine = 0;
		cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

		int x = obj.rect.x;
		int y = obj.rect.y - label_size.height - baseLine;
		if (y < 0)
			y = 0;
		if (x + label_size.width > image.cols)
			x = image.cols - label_size.width;

		cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
			cv::Scalar(255, 255, 255), -1);

		cv::putText(image, text, cv::Point(x, y + label_size.height),
			cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
	}

	cv::imshow("image", image);
	cv::waitKey(0);
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
		return -1;
	}

	const char* imagepath = argv[1];
	cv::Mat m = cv::imread(imagepath, 1);
	if (m.empty())
	{
		fprintf(stderr, "cv::imread %s failed\n", imagepath);
		return -1;
	}

	std::vector<Object> objects;

	double start = GetTickCount();
	detect_ssd(m, objects);
	double end = GetTickCount();

	fprintf(stderr, "cost time:  %.5f ms \n", (end - start));

	draw_objects(m, objects);

	return 0;
}