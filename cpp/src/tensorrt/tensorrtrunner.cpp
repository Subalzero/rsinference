#include "tensorrtrunner.h"

#include <cassert>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace rsinfer
{
	namespace
	{
		//! Maps a TensorRT element type onto the public rsinfer::DataType enum.
		rsinfer::DataType to_inferrs_data_type(nvinfer1::DataType data_type) noexcept
		{
			switch (data_type)
			{
			case nvinfer1::DataType::kFLOAT: return rsinfer::DataType::FP32;
			case nvinfer1::DataType::kHALF:  return rsinfer::DataType::FP16;
			case nvinfer1::DataType::kINT8:  return rsinfer::DataType::INT8;
			case nvinfer1::DataType::kBF16:  return rsinfer::DataType::BF16;
			case nvinfer1::DataType::kINT4:  return rsinfer::DataType::INT4;
			case nvinfer1::DataType::kFP4:   return rsinfer::DataType::FP4;
			default:                         return rsinfer::DataType::UNKNOWN;
			}
		}
	} // namespace

	TensorRTRunner::TensorRTRunner()
	{

	}

	TensorRTRunner::~TensorRTRunner()
	{

	}

	TensorRTRunner::TensorRTRunner(const RunnerParams& params) :
		_params(params)
	{

	}

	TensorRTRunner::TensorRTRunner(TensorRTRunner&& temp) noexcept
	{
		_params = std::move(temp._params);
		_engine = std::move(temp._engine);
		_runtime = std::move(temp._runtime);

		temp._engine.reset();
		temp._runtime.reset();
	}

	TensorRTRunner& TensorRTRunner::operator=(TensorRTRunner&& temp) noexcept
	{
		_params = std::move(temp._params);
		_engine = std::move(temp._engine);
		_runtime = std::move(temp._runtime);

		temp._engine.reset();
		temp._runtime.reset();

		return *this;
	}

	bool TensorRTRunner::build(const std::function<bool(IOptimizationProfile*)>& optimization)
	{
		auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
		if (!builder)
		{
			return false;
		}

		std::vector<char> plan_data;
		bool load_engine_success = false;
		if (load_engine(plan_data))
		{
			_runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
			if (!_runtime)
			{
				load_engine_success = false;
			}

			_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
				_runtime->deserializeCudaEngine(plan_data.data(), plan_data.size()));
			if (!_engine)
			{
				load_engine_success = false;
			}

			if (_runtime && _engine)
			{
				sample::gLogInfo << "Loaded model from .engine file." << std::endl;
				load_engine_success = true;
			}
		}

		if (!load_engine_success)
		{
			auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
			if (!network)
			{
				return false;
			}

			auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
			if (!config)
			{
				return false;
			}

			auto parser
				= std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
			if (!parser)
			{
				return false;
			}

			if (optimization)
			{
				nvinfer1::IOptimizationProfile* optimization_profile = builder->createOptimizationProfile();
				bool ret = optimization(optimization_profile);
				if (!ret)
				{
					return false;
				}
				config->addOptimizationProfile(optimization_profile);
			}

			auto timingCache = std::unique_ptr<nvinfer1::ITimingCache>();

			auto constructed = construct_network(builder, network, config, parser, timingCache);
			if (!constructed)
			{
				return false;
			}

			// CUDA stream used for profiling by the builder.
			auto profileStream = samplesCommon::makeCudaStream();
			if (!profileStream)
			{
				return false;
			}
			config->setProfileStream(*profileStream);

			std::unique_ptr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
			if (!plan)
			{
				return false;
			}

			if (save_engine_to_plan_file(plan.get()))
			{
				sample::gLogInfo << "Saved engine as plan file" << std::endl;
			}

			if (timingCache != nullptr && !_params.timingCacheFile.empty())
			{
				samplesCommon::updateTimingCacheFile(
					sample::gLogger.getTRTLogger(), _params.timingCacheFile, timingCache.get(), *builder);
			}

			_runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
			if (!_runtime)
			{
				return false;
			}

			_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
				_runtime->deserializeCudaEngine(plan->data(), plan->size()));
			if (!_engine)
			{
				return false;
			}
		}

		uint32_t nireq = _params.nContexts;
		for (uint32_t i = 0; i < nireq; ++i)
		{
			_execution_contexts.emplace_back(_engine->createExecutionContext());
			_idle_contexts.push(_execution_contexts[i].get());
		}

		return true;
	}

	void TensorRTRunner::process(const std::vector<Tensor>& input_tensors)
	{
		// Create RAII buffer manager object
		std::unique_lock<std::mutex> engine_lock(_engine_mut);
		samplesCommon::BufferManager buffers(_engine);
		engine_lock.unlock();

		std::unique_lock<std::mutex> idle_context_lock(_idle_contexts_mut);
		_idle_contexts_cond.wait(idle_context_lock, [&]() { return !_idle_contexts.empty(); });
		nvinfer1::IExecutionContext* context = _idle_contexts.front();
		_idle_contexts.pop();
		idle_context_lock.unlock();

		for (int32_t i = 0, e = _engine->getNbIOTensors(); i < e; i++)
		{
			auto const name = _engine->getIOTensorName(i);
			context->setTensorAddress(name, buffers.getDeviceBuffer(name));
		}

		process_input(buffers, context, input_tensors);

		// Memcpy from host input buffers to device input buffers
		buffers.copyInputToDevice();

		std::unique_lock<std::mutex> busy_contexts_lock(_busy_contexts_mut);
		_busy_contexts.push(context);
		_busy_contexts_cond.notify_one();
		busy_contexts_lock.unlock();

		context->executeV2(buffers.getDeviceBindings().data());

		// Memcpy from device output buffers to host output buffers
		buffers.copyOutputToHost();

		verify_output(buffers, context);

		busy_contexts_lock.lock();
		_busy_contexts_cond.wait(busy_contexts_lock, [&]() { return !_busy_contexts.empty(); });
		_busy_contexts.pop();
		busy_contexts_lock.unlock();

		idle_context_lock.lock();
		_idle_contexts.push(context);
		_idle_contexts_cond.notify_one();
		idle_context_lock.unlock();
	}

	void TensorRTRunner::process_async(const std::vector<Tensor>& input_tensors)
	{
		std::thread(&TensorRTRunner::process, this, input_tensors).detach();
	}

	void TensorRTRunner::get(std::vector<Tensor>& result)
	{
		std::unique_lock<std::mutex> inference_results_lock(_inference_results_mut);
		_inference_results_cond.wait(inference_results_lock, [&]() { return !_inference_results.empty(); });
		result = std::move(_inference_results.front());
		_inference_results.pop();
	}

	std::vector<int> TensorRTRunner::get_input_shape(uint32_t index) const
	{
		std::vector<int> vect{};

		nvinfer1::Dims dims = _engine->getTensorShape(get_input_tensor_name(index).c_str());
		for (int i = 0; i < dims.nbDims; ++i)
		{
			vect.push_back(static_cast<int>(dims.d[i]));
		}
		return vect;
	}

	std::vector<int> TensorRTRunner::get_output_shape(uint32_t index) const
	{
		std::vector<int> vect{};

		nvinfer1::Dims dims = _engine->getTensorShape(get_output_tensor_name(index).c_str());
		for (int i = 0; i < dims.nbDims; ++i)
		{
			vect.push_back(static_cast<int>(dims.d[i]));
		}
		return vect;
	}

	nvinfer1::DataType TensorRTRunner::get_tensor_data_type(const std::string& name) const
	{
		nvinfer1::DataType data_type = _engine->getTensorDataType(name.c_str());
		return data_type;
	}

	constexpr size_t TensorRTRunner::get_tensor_data_size(const nvinfer1::DataType& data_type) const
	{
		switch (data_type)
		{
		case nvinfer1::DataType::kFLOAT:      return 4;
		case nvinfer1::DataType::kHALF:       return 2;
		case nvinfer1::DataType::kINT8:       return 1;
		case nvinfer1::DataType::kINT32:      return 4;
		case nvinfer1::DataType::kBOOL:       return 1;
		case nvinfer1::DataType::kUINT8:      return 1;
		case nvinfer1::DataType::kFP8:        return 1;
		case nvinfer1::DataType::kE8M0:       return 1;
		case nvinfer1::DataType::kBF16:       return 2;
		case nvinfer1::DataType::kINT64:      return 8;
		// 4-bit types are packed: two elements per byte.
		case nvinfer1::DataType::kINT4:       return 2;
		case nvinfer1::DataType::kFP4:        return 2;
		default:                              return 0;
		}
	}

	std::string TensorRTRunner::get_input_tensor_name(uint32_t index) const
	{
		std::vector<std::string> input_names;

		int numIOTensors = _engine->getNbIOTensors();

		for (int i = 0; i < numIOTensors; ++i) {
			const char* tensorName = _engine->getIOTensorName(i);

			if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kINPUT) {
				input_names.push_back(tensorName);
			}
		}

		return input_names[index];
	}

	std::string TensorRTRunner::get_output_tensor_name(uint32_t index) const
	{
		std::vector<std::string> output_names;

		int numIOTensors = _engine->getNbIOTensors();

		for (int i = 0; i < numIOTensors; ++i) {
			const char* tensorName = _engine->getIOTensorName(i);

			if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT) {
				output_names.push_back(tensorName);
			}
		}

		return output_names[index];
	}

	size_t TensorRTRunner::get_number_of_inputs() const
	{
		// Step 1: Get the number of I/O tensors
		int numIOTensors = _engine->getNbIOTensors();
		int numInputLayers = 0;

		// Step 2: Loop through all tensors and count the inputs
		for (int i = 0; i < numIOTensors; ++i) {
			// Get the tensor name at index i
			const char* tensorName = _engine->getIOTensorName(i);

			// Step 3: Check if this tensor is an input
			if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kINPUT) {
				numInputLayers++;
			}
		}

		return numInputLayers;
	}

	size_t TensorRTRunner::get_number_of_outputs() const
	{
		// Step 1: Get the number of I/O tensors
		int numIOTensors = _engine->getNbIOTensors();
		int numOutputLayers = 0;

		// Step 2: Loop through all tensors and count the outputs
		for (int i = 0; i < numIOTensors; ++i) {
			// Get the tensor name at index i
			const char* tensorName = _engine->getIOTensorName(i);

			// Step 3: Check if this tensor is an output
			if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT) {
				numOutputLayers++;
			}
		}

		return numOutputLayers;
	}

	bool TensorRTRunner::construct_network(std::unique_ptr<nvinfer1::IBuilder>& builder, std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config, std::unique_ptr<nvonnxparser::IParser>& parser, std::unique_ptr<nvinfer1::ITimingCache>& timingCache)
	{
		// The network type is configured by the parser; kept for signature compatibility.
		(void)network;

		auto parsed = parser->parseFromFile(samplesCommon::locateFile(_params.onnxFileName, _params.dataDirs).c_str(),
			static_cast<int>(sample::gLogger.getReportableSeverity()));
		if (!parsed)
		{
			return false;
		}

		// TensorRT 11 removed the weak-typing builder flags (kFP16/kBF16/kINT8):
		// strongly typed networks are always enabled, so the precision of every
		// tensor is determined by the ONNX graph. RunnerParams::fp16/bf16/int8
		// are therefore not applied here.

		if (_params.timingCacheFile.size())
		{
			timingCache = samplesCommon::buildTimingCacheFromFile(
				sample::gLogger.getTRTLogger(), *config, _params.timingCacheFile);
		}

		samplesCommon::enableDLA(builder.get(), config.get(), _params.dlaCore);

		return true;
	}

	bool TensorRTRunner::process_input(const samplesCommon::BufferManager& buffers, nvinfer1::IExecutionContext* context, const std::vector<Tensor>& tensors)
	{
		auto n_inputs = get_number_of_inputs();
		std::size_t count = tensors.size();
		assert(count == n_inputs);
		for (size_t i = 0; i < count; ++i)
		{
			auto& tensor = tensors[i];
			std::string tensor_name = tensor.name;

			std::vector<int> input_shape;
			input_shape.reserve(tensor.shape.size());
			for (auto dim : tensor.shape)
			{
				input_shape.push_back(static_cast<int>(dim));
			}
			context->setInputShape(tensor_name.c_str(), vector_to_dims(input_shape));

			const nvinfer1::DataType engine_data_type = get_tensor_data_type(tensor_name);
			assert(tensor.dtype == to_inferrs_data_type(engine_data_type));

			size_t input_size = 1;
			std::vector<int> input_dims = dims_to_vector(context->getTensorShape(tensor_name.c_str()));
			for (size_t d = 0; d < input_dims.size(); ++d)
			{
				assert(input_dims[d] != -1);
				input_size *= input_dims[d];
			}

			auto hostDataBuffer = buffers.getHostBuffer(get_input_tensor_name(static_cast<uint32_t>(i)));

			memcpy(hostDataBuffer, tensor.data.data(),
				samplesCommon::getNbBytes(engine_data_type, static_cast<int64_t>(input_size)));
		}

		return true;
	}

	bool TensorRTRunner::verify_output(const samplesCommon::BufferManager& buffers, const nvinfer1::IExecutionContext* context)
	{
		auto n_outputs = get_number_of_outputs();
		std::vector<Tensor> output_vect;
		for (uint32_t i = 0; i < n_outputs; ++i)
		{
			std::string tensor_name = get_output_tensor_name(i);
			size_t output_size = 1;
			std::vector<int> output_dims = dims_to_vector(context->getTensorShape(tensor_name.c_str()));
			for (size_t d = 0; d < output_dims.size(); ++d)
			{
				assert(output_dims[d] != -1);
				output_size *= output_dims[d];
			}

			unsigned char* output = static_cast<unsigned char*>(buffers.getHostBuffer(get_output_tensor_name(i)));

			Tensor out;
			const nvinfer1::DataType data_type = get_tensor_data_type(tensor_name);
			const size_t output_bytes = samplesCommon::getNbBytes(data_type, static_cast<int64_t>(output_size));
			out.name = tensor_name;
			out.shape.assign(output_dims.begin(), output_dims.end());
			out.data = std::vector<unsigned char>(output, output + output_bytes);
			out.dtype = to_inferrs_data_type(data_type);

			output_vect.push_back(std::move(out));
		}
		std::lock_guard<std::mutex> inference_results_lock(_inference_results_mut);
		_inference_results.push(std::move(output_vect));
		_inference_results_cond.notify_one();
		return true;
	}

	bool TensorRTRunner::load_engine(std::vector<char>& data) {
		std::istringstream stream(_params.onnxFileName);
		std::string file_name;
		std::getline(stream, file_name, '.');
		std::string engineFilePath = file_name + ".engine";
		std::ifstream file(engineFilePath, std::ios::binary);
		if (!file) {
			//throw std::runtime_error("Could not open engine file: " + engineFilePath);
			sample::gLogWarning << "Could not open engine file: " + engineFilePath << std::endl;
			sample::gLogInfo << "Will attempt to create engine file from .onnx model." << std::endl;
			return false;
		}

		file.seekg(0, std::ifstream::end);
		size_t fileSize = static_cast<size_t>(file.tellg());
		file.seekg(0, std::ifstream::beg);

		std::vector<char> engineData(fileSize);
		file.read(engineData.data(), static_cast<std::streamsize>(fileSize));
		file.close();

		data = engineData;
		return true;
	}

	bool TensorRTRunner::save_engine_to_plan_file(nvinfer1::IHostMemory* plan)
	{
		std::istringstream stream(_params.onnxFileName);
		std::string file_name;
		std::getline(stream, file_name, '.');
		std::string planFilePath = file_name + ".engine";
		std::ofstream planFile(planFilePath, std::ios::binary);
		planFile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
		planFile.close();
		return true;
	}

	nvinfer1::Dims TensorRTRunner::vector_to_dims(const std::vector<int>& vec)
	{
		assert(vec.size() <= 8);
		nvinfer1::Dims dims;
		dims.nbDims = static_cast<int32_t>(vec.size());
		for (int32_t i = 0; i < dims.nbDims; ++i)
		{
			dims.d[i] = vec[i];
		}

		return dims;
	}

	std::vector<int> TensorRTRunner::dims_to_vector(const nvinfer1::Dims& dims)
	{
		std::vector<int> vec;
		vec.reserve(8);
		for (int32_t i = 0; i < dims.nbDims; ++i)
		{
			vec.push_back(static_cast<int>(dims.d[i]));
		}
		return vec;
	}
} // namespace inferrs
