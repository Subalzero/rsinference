// Live webcam face detection with the version-RFB-640 UltraFace ONNX model,
// running through inferrs::TensorRTRunner and OpenCV.
//
// Usage:
//   tensorrtfacedetecttest [model.onnx] [camera_index] [score_threshold]
//
// Defaults:
//   model           <repo>/models/version-RFB-640.onnx (set by CMake)
//   camera_index    0
//   score_threshold 0.7
//
// The first run parses the ONNX model and builds a TensorRT engine next to it
// (`version-RFB-640.engine`); later runs deserialize the cached engine and
// start immediately. Press ESC or q to quit.
//
// At runtime the TensorRT/CUDA and OpenCV DLLs must be on PATH, e.g. from a
// Developer PowerShell:
//   $env:PATH = "C:\TensorRT-11.0.0.114\bin;<build>\vcpkg_installed\x64-windows\bin;$env:PATH"

#include "tensorrtrunner.h"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifndef INFERTENSORRT_DEFAULT_MODEL_PATH
#define INFERTENSORRT_DEFAULT_MODEL_PATH "models/version-RFB-640.onnx"
#endif

namespace
{
	struct Detection
	{
		cv::Rect2f box;
		float score;
	};

	template <typename T>
	std::string shape_to_string(const std::vector<T>& shape)
	{
		std::ostringstream stream;
		stream << '[';
		for (size_t i = 0; i < shape.size(); ++i)
		{
			if (i != 0)
			{
				stream << ", ";
			}
			stream << shape[i];
		}
		stream << ']';
		return stream.str();
	}

	//! Resize to the model input, convert BGR -> RGB, normalize to
	//! (x - 127) / 128 and lay the image out as planar float32 CHW.
	std::vector<unsigned char> preprocess(const cv::Mat& bgr, int width, int height)
	{
		cv::Mat rgb;
		cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
		cv::resize(rgb, rgb, cv::Size(width, height));

		cv::Mat normalized;
		rgb.convertTo(normalized, CV_32F, 1.0 / 128.0, -127.0 / 128.0);

		const size_t plane_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * sizeof(float);
		std::vector<unsigned char> bytes(3 * plane_bytes);

		std::vector<cv::Mat> channels;
		cv::split(normalized, channels);
		for (int channel = 0; channel < 3; ++channel)
		{
			std::memcpy(bytes.data() + channel * plane_bytes, channels[channel].data, plane_bytes);
		}

		return bytes;
	}

	float intersection_over_union(const cv::Rect2f& a, const cv::Rect2f& b)
	{
		const float x1 = std::max(a.x, b.x);
		const float y1 = std::max(a.y, b.y);
		const float x2 = std::min(a.x + a.width, b.x + b.width);
		const float y2 = std::min(a.y + a.height, b.y + b.height);

		const float overlap_width = x2 - x1;
		const float overlap_height = y2 - y1;
		if (overlap_width <= 0.0F || overlap_height <= 0.0F)
		{
			return 0.0F;
		}

		const float intersection = overlap_width * overlap_height;
		const float area_a = a.width * a.height;
		const float area_b = b.width * b.height;
		return intersection / (area_a + area_b - intersection);
	}

	std::vector<Detection> apply_nms(std::vector<Detection> detections, float iou_threshold)
	{
		std::sort(detections.begin(), detections.end(),
			[](const Detection& a, const Detection& b) { return a.score > b.score; });

		std::vector<Detection> kept;
		std::vector<unsigned char> suppressed(detections.size(), 0);
		for (size_t i = 0; i < detections.size(); ++i)
		{
			if (suppressed[i])
			{
				continue;
			}

			kept.push_back(detections[i]);
			for (size_t j = i + 1; j < detections.size(); ++j)
			{
				if (!suppressed[j] && intersection_over_union(detections[i].box, detections[j].box) > iou_threshold)
				{
					suppressed[j] = 1;
				}
			}
		}
		return kept;
	}

	//! The model already applies softmax to `scores` and decodes `boxes`, so
	//! only thresholding, scaling to the frame and NMS are left to do here.
	std::vector<Detection> postprocess(const std::vector<inferrs::Tensor>& outputs,
		const cv::Size& frame_size, float score_threshold, float iou_threshold)
	{
		const inferrs::Tensor* scores = nullptr;
		const inferrs::Tensor* boxes = nullptr;
		for (const auto& tensor : outputs)
		{
			if (tensor.name == "scores")
			{
				scores = &tensor;
			}
			else if (tensor.name == "boxes")
			{
				boxes = &tensor;
			}
		}

		// Fall back to the output order used by the ONNX graph.
		if ((scores == nullptr || boxes == nullptr) && outputs.size() >= 2)
		{
			scores = &outputs[0];
			boxes = &outputs[1];
		}
		if (scores == nullptr || boxes == nullptr)
		{
			return {};
		}

		const size_t score_stride = scores->shape.empty() ? 2 : static_cast<size_t>(scores->shape.back());
		const size_t num_anchors = scores->shape.size() >= 3 ? static_cast<size_t>(scores->shape[1])
			: scores->data.size() / (score_stride * sizeof(float));
		const size_t box_stride = boxes->shape.empty() ? 4 : static_cast<size_t>(boxes->shape.back());

		if (score_stride < 2 || box_stride < 4
			|| scores->data.size() < num_anchors * score_stride * sizeof(float)
			|| boxes->data.size() < num_anchors * box_stride * sizeof(float))
		{
			std::cerr << "Unexpected output tensor layout (scores shape " << shape_to_string(scores->shape)
				<< ", boxes shape " << shape_to_string(boxes->shape) << ")\n";
			return {};
		}

		const auto* score_data = reinterpret_cast<const float*>(scores->data.data());
		const auto* box_data = reinterpret_cast<const float*>(boxes->data.data());

		const float max_x = static_cast<float>(frame_size.width - 1);
		const float max_y = static_cast<float>(frame_size.height - 1);

		std::vector<Detection> detections;
		for (size_t i = 0; i < num_anchors; ++i)
		{
			const float score = score_data[i * score_stride + 1]; // class 0 = background
			if (score < score_threshold)
			{
				continue;
			}

			const float* box = box_data + i * box_stride;
			const float x1 = std::clamp(box[0] * static_cast<float>(frame_size.width), 0.0F, max_x);
			const float y1 = std::clamp(box[1] * static_cast<float>(frame_size.height), 0.0F, max_y);
			const float x2 = std::clamp(box[2] * static_cast<float>(frame_size.width), 0.0F, max_x);
			const float y2 = std::clamp(box[3] * static_cast<float>(frame_size.height), 0.0F, max_y);
			if (x2 <= x1 || y2 <= y1)
			{
				continue;
			}

			detections.push_back(Detection{ cv::Rect2f(x1, y1, x2 - x1, y2 - y1), score });
		}

		return apply_nms(std::move(detections), iou_threshold);
	}

	void draw_detections(cv::Mat& frame, const std::vector<Detection>& detections, double fps)
	{
		for (const auto& detection : detections)
		{
			const cv::Scalar color(0, 255, 0);
			cv::rectangle(frame, static_cast<cv::Rect>(detection.box), color, 2);

			const std::string label = cv::format("%.2f", detection.score);
			const cv::Point origin(static_cast<int>(detection.box.x),
				std::max(12, static_cast<int>(detection.box.y) - 6));
			cv::putText(frame, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
		}

		const std::string status = cv::format("FPS %.1f | faces %zu", fps, detections.size());
		cv::putText(frame, status, cv::Point(10, 28), cv::FONT_HERSHEY_SIMPLEX, 0.7,
			cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		std::filesystem::path model_path = INFERTENSORRT_DEFAULT_MODEL_PATH;
		int camera_index = 0;
		float score_threshold = 0.7F;

		if (argc > 1)
		{
			model_path = argv[1];
		}
		if (argc > 2)
		{
			camera_index = std::stoi(argv[2]);
		}
		if (argc > 3)
		{
			score_threshold = std::stof(argv[3]);
		}
		if (argc > 4)
		{
			std::cerr << "usage: " << argv[0] << " [model.onnx] [camera_index] [score_threshold]\n";
			return 2;
		}

		model_path = std::filesystem::absolute(model_path);
		if (!std::filesystem::exists(model_path))
		{
			std::cerr << "Model not found: " << model_path << "\n";
			return 1;
		}

		cv::VideoCapture capture(camera_index);
		if (!capture.isOpened())
		{
			std::cerr << "Could not open camera " << camera_index
				<< ". Try another index or check camera permissions.\n";
			return 1;
		}

		capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
		capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

		inferrs::RunnerParams params;
		// locateFile() requires a non-empty dataDirs list; an empty entry
		// means "use onnxFileName as given".
		params.onnxFileName = model_path.string();
		params.dataDirs = { "" };
		params.nContexts = 1;

		std::cout << "Building/loading TensorRT engine from " << model_path << " ..." << std::endl;
		inferrs::TensorRTRunner runner(params);
		if (!runner.build())
		{
			std::cerr << "Failed to build the TensorRT engine.\n";
			return 1;
		}

		const std::vector<int> input_shape = runner.get_input_shape(0);
		if (input_shape.size() != 4 || input_shape[1] != 3)
		{
			std::cerr << "Unexpected model input shape: " << shape_to_string(input_shape) << "\n";
			return 1;
		}

		const int input_height = input_shape[2];
		const int input_width = input_shape[3];

		std::cout << "Model ready\n"
			<< "  input : " << runner.get_input_tensor_name(0) << ' ' << shape_to_string(input_shape) << '\n';
		for (size_t i = 0; i < runner.get_number_of_outputs(); ++i)
		{
			std::cout << "  output: " << runner.get_output_tensor_name(i) << ' '
				<< shape_to_string(runner.get_output_shape(i)) << '\n';
		}

		std::cout << "Press ESC or q to quit.\n";
		cv::Mat frame;
		double fps = 0.0;
		const std::string window = "TensorRT UltraFace (version-RFB-640)";

		while (capture.read(frame) && !frame.empty())
		{
			// Mirror the preview so it behaves like a selfie camera.
			cv::flip(frame, frame, 1);

			const int64_t start = cv::getTickCount();

			inferrs::Tensor input;
			input.name = runner.get_input_tensor_name(0);
			input.shape = { 1, 3, input_height, input_width };
			input.dtype = inferrs::DataType::FP32;
			input.data = preprocess(frame, input_width, input_height);

			std::vector<inferrs::Tensor> inputs;
			inputs.push_back(std::move(input));

			runner.process(inputs);

			std::vector<inferrs::Tensor> outputs;
			runner.get(outputs);

			const std::vector<Detection> detections = postprocess(outputs, frame.size(), score_threshold, 0.5F);

			const double frame_ms = (cv::getTickCount() - start) * 1000.0 / cv::getTickFrequency();
			const double instant_fps = frame_ms > 0.0 ? 1000.0 / frame_ms : 0.0;
			fps = fps > 0.0 ? 0.9 * fps + 0.1 * instant_fps : instant_fps;

			draw_detections(frame, detections, fps);

			cv::imshow(window, frame);

			const int key = cv::waitKey(1) & 0xFF;
			if (key == 27 || key == 'q' || key == 'Q')
			{
				break;
			}
			if (cv::getWindowProperty(window, cv::WND_PROP_VISIBLE) < 1.0)
			{
				break;
			}
		}

		capture.release();
		cv::destroyAllWindows();
		return 0;
	}
	catch (const std::exception& error)
	{
		std::cerr << "Error: " << error.what() << '\n';
		return 1;
	}
}
