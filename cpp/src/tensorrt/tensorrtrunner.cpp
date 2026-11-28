#include <stdexcept>

#include "tensorrtrunner.h"

#include <cassert>
#include <cstdio>
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
			case nvinfer1::DataType::kINT32: return rsinfer::DataType::INT32;
			case nvinfer1::DataType::kBOOL:  return rsinfer::DataType::BOOL;
			case nvinfer1::DataType::kUINT8: return rsinfer::DataType::UINT8;
			case nvinfer1::DataType::kFP8:   return rsinfer::DataType::FP8;
			case nvinfer1::DataType::kE8M0:  return rsinfer::DataType::E8M0;
			case nvinfer1::DataType::kINT64: return rsinfer::DataType::INT64;
			case nvinfer1::DataType::kBF16:  return rsinfer::DataType::BF16;
			case nvinfer1::DataType::kINT4:  return rsinfer::DataType::INT4;
			case nvinfer1::DataType::kFP4:   return rsinfer::DataType::FP4;
			default:                         return rsinfer::DataType::UNKNOWN;
			}
		}

		//! Stable 64-bit FNV-1a hash of the optimization profile, so each set of
		//! min/opt/max ranges gets its own cache file. std::hash is not used
		//! because its value is not guaranteed to be stable between builds.
		uint64_t profile_hash(const std::map<std::string, RunnerParams::ProfileDims>& profile)
		{
			uint64_t h = 14695981039346656037ull;
			auto mix = [&h](const void* bytes, size_t n)
			{
				const auto* p = static_cast<const unsigned char*>(bytes);
				for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
			};
			for (const auto& [name, dims] : profile)
			{
				mix(name.data(), name.size() + 1); // include the NUL as a separator
				for (const nvinfer1::Dims* d : { &dims.min, &dims.opt, &dims.max })
				{
					const int64_t rank = d->nbDims;
					mix(&rank, sizeof(rank));
					for (int32_t i = 0; i < d->nbDims; ++i)
					{
						const int64_t v = d->d[i];
						mix(&v, sizeof(v));
					}
				}
			}
			return h;
		}

		void check_cuda(cudaError_t status, const char* what)
		{
			if (status != cudaSuccess)
				throw std::runtime_error(std::string(what) + " failed: " + cudaGetErrorString(status));
		}

		int64_t volume(const nvinfer1::Dims& dims)
		{
			int64_t v = 1;
			for (int32_t i = 0; i < dims.nbDims; ++i)
				v *= dims.d[i];
			return v;
		}

		//! "<model>.engine" for static builds, "<model>.dynamic-<profile hash>.engine"
		//! for dynamic ones, so changing the profile never reuses an engine built
		//! for other shape ranges. Splits at the last '.', not the first, so
		//! "./models/foo.onnx" works.
		std::string engine_path_for(const RunnerParams& params)
		{
			const std::string& onnx_file = params.onnxFileName;
			const auto slash = onnx_file.find_last_of("/\\");
			const auto dot = onnx_file.rfind('.');
			const std::string base = (dot == std::string::npos || (slash != std::string::npos && dot < slash))
				? onnx_file
				: onnx_file.substr(0, dot);
			if (!params.dynamic)
				return base + ".engine";
			char hash[17];
			std::snprintf(hash, sizeof(hash), "%016llx", static_cast<unsigned long long>(profile_hash(params.profile)));
			return base + ".dynamic-" + hash + ".engine";
		}

		// The engine cache has no metadata sidecar, so it cannot prove that a
		// serialized plan was built with the current builder options. Only reuse
		// it when all build settings are at their defaults.
		bool has_builder_overrides(const RunnerParams& params)
		{
			return params.dlaCore != -1 || params.int8 || params.fp16 || params.bf16 || !params.builderFlags.empty()
				|| params.builderOptimizationLevel || params.avgTimingIterations || params.maxAuxStreams
				|| params.maxThreads || params.tacticSources || params.profilingVerbosity
				|| params.engineCapability || params.hardwareCompatibilityLevel || params.defaultDeviceType
				|| params.runtimePlatform || params.tilingOptimizationLevel || params.tilingL2Limit
				|| !params.previewFeatures.empty() || !params.pluginsToSerialize.empty()
				|| !params.layerDeviceTypes.empty() || !params.allowGpuFallback
				|| !params.memoryPoolLimits.empty() || !params.timingCacheFile.empty();
		}
	} // namespace

	TensorRTRunner::TensorRTRunner()
	{

	}

	TensorRTRunner::~TensorRTRunner()
	{
		// Stream callbacks of process_async requests still use `this`.
		{
			std::unique_lock<std::mutex> lock(_inflight_mut);
			_inflight_cond.wait(lock, [&]() { return _inflight == 0; });
		}
		release_slots();
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
		if (_params.maxThreads && !builder->setMaxThreads(*_params.maxThreads))
			return false;

		std::vector<char> plan_data;
		bool load_engine_success = false;
		if (!has_builder_overrides(_params) && load_engine(plan_data))
		{
			_runtime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
			if (_runtime)
			{
				_engine = std::shared_ptr<nvinfer1::ICudaEngine>(
					_runtime->deserializeCudaEngine(plan_data.data(), plan_data.size()));
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

			for (const auto flag : _params.builderFlags)
				config->setFlag(flag);
			if (_params.builderOptimizationLevel)
				config->setBuilderOptimizationLevel(*_params.builderOptimizationLevel);
			if (_params.avgTimingIterations)
				config->setAvgTimingIterations(*_params.avgTimingIterations);
			if (_params.maxAuxStreams && !config->setMaxAuxStreams(*_params.maxAuxStreams))
				return false;
			if (_params.tacticSources && !config->setTacticSources(*_params.tacticSources))
				return false;
			if (_params.profilingVerbosity)
				config->setProfilingVerbosity(*_params.profilingVerbosity);
			if (_params.engineCapability)
				config->setEngineCapability(*_params.engineCapability);
			if (_params.hardwareCompatibilityLevel)
				config->setHardwareCompatibilityLevel(*_params.hardwareCompatibilityLevel);
			if (_params.defaultDeviceType)
				config->setDefaultDeviceType(*_params.defaultDeviceType);
			if (_params.runtimePlatform)
				config->setRuntimePlatform(*_params.runtimePlatform);
			if (_params.tilingOptimizationLevel && !config->setTilingOptimizationLevel(*_params.tilingOptimizationLevel))
				return false;
			if (_params.tilingL2Limit && !config->setL2LimitForTiling(*_params.tilingL2Limit))
				return false;
			for (const auto& [feature, enabled] : _params.previewFeatures)
				config->setPreviewFeature(feature, enabled);
			if (!_params.pluginsToSerialize.empty())
			{
				std::vector<const char*> plugin_paths;
				plugin_paths.reserve(_params.pluginsToSerialize.size());
				for (const auto& path : _params.pluginsToSerialize)
					plugin_paths.push_back(path.c_str());
				config->setPluginsToSerialize(plugin_paths.data(), static_cast<int32_t>(plugin_paths.size()));
			}
			for (const auto& [pool, size] : _params.memoryPoolLimits)
				config->setMemoryPoolLimit(pool, size);

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

		if (_params.nContexts == 0)
		{
			return false;
		}
		if ((!_params.inputTensorNames.empty() && _params.inputTensorNames.size() != get_number_of_inputs()) ||
			(!_params.outputTensorNames.empty() && _params.outputTensorNames.size() != get_number_of_outputs()))
		{
			return false;
		}
		for (const auto& name : _params.inputTensorNames)
		{
			if (_engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kINPUT)
				return false;
		}
		for (const auto& name : _params.outputTensorNames)
		{
			if (_engine->getTensorIOMode(name.c_str()) != nvinfer1::TensorIOMode::kOUTPUT)
				return false;
		}

		return init_slots();
	}

	bool TensorRTRunner::init_slots()
	{
		release_slots();
		_io.clear();
		_input_order.clear();
		_output_order.clear();

		for (int32_t i = 0, n = _engine->getNbIOTensors(); i < n; ++i)
		{
			const char* name = _engine->getIOTensorName(i);
			_io.push_back({ name, _engine->getTensorIOMode(name) == TensorIOMode::kINPUT, _engine->getTensorDataType(name) });
		}
		auto index_of = [this](const std::string& name) -> std::optional<size_t>
		{
			for (size_t i = 0; i < _io.size(); ++i)
				if (_io[i].name == name) return i;
			return std::nullopt;
		};
		for (uint32_t i = 0, n = static_cast<uint32_t>(get_number_of_inputs()); i < n; ++i)
		{
			auto idx = index_of(get_input_tensor_name(i));
			if (!idx) return false;
			_input_order.push_back(*idx);
		}
		for (uint32_t i = 0, n = static_cast<uint32_t>(get_number_of_outputs()); i < n; ++i)
		{
			auto idx = index_of(get_output_tensor_name(i));
			if (!idx) return false;
			_output_order.push_back(*idx);
		}

		try
		{
			for (uint32_t i = 0; i < _params.nContexts; ++i)
			{
				auto slot = std::make_unique<Slot>();
				slot->owner = this;
				slot->context.reset(_engine->createExecutionContext());
				if (!slot->context)
				{
					release_slots();
					return false;
				}
				check_cuda(cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
				slot->buffers.resize(_io.size());
				// Pre-allocate buffers for tensors whose shape is static, so the first
				// request does not pay for allocation. Dynamic ones grow on demand.
				for (size_t t = 0; t < _io.size(); ++t)
				{
					const nvinfer1::Dims dims = _engine->getTensorShape(_io[t].name.c_str());
					bool is_static = true;
					for (int32_t d = 0; d < dims.nbDims; ++d)
						is_static = is_static && dims.d[d] >= 0;
					if (is_static)
						ensure_capacity(slot->buffers[t], samplesCommon::getNbBytes(_io[t].type, volume(dims)));
				}
				_idle_slots.push(slot.get());
				_slots.push_back(std::move(slot));
			}
		}
		catch (const std::exception& e)
		{
			sample::gLogError << "Failed to create execution slots: " << e.what() << std::endl;
			release_slots();
			return false;
		}
		return true;
	}

	void TensorRTRunner::release_slots() noexcept
	{
		for (auto& slot : _slots)
		{
			if (slot->stream)
				cudaStreamSynchronize(slot->stream);
			destroy_graph(*slot);
			for (auto& buffer : slot->buffers)
			{
				if (buffer.device) cudaFree(buffer.device);
				if (buffer.host) cudaFreeHost(buffer.host);
				buffer = {};
			}
			slot->context.reset();
			if (slot->stream)
				cudaStreamDestroy(slot->stream);
		}
		_slots.clear();
		_idle_slots = {};
	}

	void TensorRTRunner::ensure_capacity(IOBuffer& buffer, size_t bytes)
	{
		// TensorRT wants a valid address even for an empty tensor.
		const size_t needed = bytes == 0 ? 1 : bytes;
		if (needed <= buffer.capacity)
			return;
		// The slot is owned exclusively and its stream is idle, so the old
		// buffers are no longer in use.
		if (buffer.device) cudaFree(buffer.device);
		if (buffer.host) cudaFreeHost(buffer.host);
		buffer.device = nullptr;
		buffer.host = nullptr;
		buffer.capacity = 0;
		check_cuda(cudaMalloc(&buffer.device, needed), "cudaMalloc");
		check_cuda(cudaHostAlloc(&buffer.host, needed, cudaHostAllocDefault), "cudaHostAlloc");
		buffer.capacity = needed;
	}

	TensorRTRunner::Slot* TensorRTRunner::acquire_slot()
	{
		std::unique_lock<std::mutex> lock(_idle_slots_mut);
		if (_slots.empty())
			throw std::runtime_error("TensorRT runner has no execution contexts");
		_idle_slots_cond.wait(lock, [&]() { return !_idle_slots.empty(); });
		Slot* slot = _idle_slots.front();
		_idle_slots.pop();
		return slot;
	}

	void TensorRTRunner::return_slot(Slot* slot) noexcept
	{
		{
			std::lock_guard<std::mutex> lock(_idle_slots_mut);
			_idle_slots.push(slot);
		}
		_idle_slots_cond.notify_one();
	}

	TensorRTRunner::Slot* TensorRTRunner::enqueue(std::span<const TensorView> input_tensors)
	{
		if (input_tensors.size() != _input_order.size())
		{
			throw std::runtime_error("Expected " + std::to_string(_input_order.size()) + " input tensors, got "
				+ std::to_string(input_tensors.size()));
		}

		Slot* slot = acquire_slot();
		// On any failure, wait out whatever was already queued on the stream
		// before the slot (and its buffers) can be reused.
		struct SlotGuard
		{
			TensorRTRunner* self;
			Slot* slot;
			~SlotGuard()
			{
				if (!slot) return;
				cudaStreamSynchronize(slot->stream);
				self->return_slot(slot);
			}
		} guard{ this, slot };

		nvinfer1::IExecutionContext* context = slot->context.get();

		// Match each view to its engine input by name, then resolve its shape.
		std::vector<size_t> view_io(input_tensors.size());
		for (size_t v = 0; v < input_tensors.size(); ++v)
		{
			const TensorView& view = input_tensors[v];
			size_t idx = _io.size();
			for (size_t i : _input_order)
			{
				if (_io[i].name == view.name)
				{
					idx = i;
					break;
				}
			}
			if (idx == _io.size())
				throw std::runtime_error("Unknown input tensor: " + std::string(view.name));
			for (size_t u = 0; u < v; ++u)
			{
				if (view_io[u] == idx)
					throw std::runtime_error("Input tensor given twice: " + std::string(view.name));
			}
			view_io[v] = idx;

			if (view.dtype != to_inferrs_data_type(_io[idx].type))
				throw std::runtime_error("Input tensor '" + _io[idx].name + "' has a data type that does not match the engine");
			if (view.shape.size() > static_cast<size_t>(nvinfer1::Dims::MAX_DIMS))
				throw std::runtime_error("Input tensor '" + _io[idx].name + "' has too many dimensions");
			nvinfer1::Dims dims{};
			dims.nbDims = static_cast<int32_t>(view.shape.size());
			for (int32_t d = 0; d < dims.nbDims; ++d)
				dims.d[d] = view.shape[d];
			if (!context->setInputShape(_io[idx].name.c_str(), dims))
				throw std::runtime_error("setInputShape failed for tensor: " + _io[idx].name);
		}
		if (!context->allInputDimensionsSpecified())
			throw std::runtime_error("Not all input dimensions were specified");

		// Size every buffer from the resolved shapes and bind it.
		bool buffers_moved = false;
		for (size_t i = 0; i < _io.size(); ++i)
		{
			IOBuffer& buffer = slot->buffers[i];
			buffer.shape = context->getTensorShape(_io[i].name.c_str());
			for (int32_t d = 0; d < buffer.shape.nbDims; ++d)
			{
				if (buffer.shape.d[d] < 0)
					throw std::runtime_error("Unresolved dynamic dimension in tensor: " + _io[i].name);
			}
			buffer.bytes = samplesCommon::getNbBytes(_io[i].type, volume(buffer.shape));
			const void* previous = buffer.device;
			ensure_capacity(buffer, buffer.bytes);
			buffers_moved = buffers_moved || buffer.device != previous;
			if (!context->setTensorAddress(_io[i].name.c_str(), buffer.device))
				throw std::runtime_error("setTensorAddress failed for tensor: " + _io[i].name);
		}

		// Copy in, step 1: caller memory -> pinned. The device copies are queued
		// by record_work (directly or inside the slot's CUDA graph).
		for (size_t v = 0; v < input_tensors.size(); ++v)
		{
			IOBuffer& buffer = slot->buffers[view_io[v]];
			const auto& data = input_tensors[v].data;
			if (data.size() != buffer.bytes)
			{
				throw std::runtime_error("Input tensor '" + _io[view_io[v]].name + "' has " + std::to_string(data.size())
					+ " bytes, expected " + std::to_string(buffer.bytes));
			}
			if (!data.empty())
				std::memcpy(buffer.host, data.data(), data.size());
		}

		// The graph is only valid for the shapes it was recorded with.
		std::vector<int64_t> key;
		for (size_t i : _input_order)
		{
			const nvinfer1::Dims& shape = slot->buffers[i].shape;
			key.push_back(shape.nbDims);
			key.insert(key.end(), shape.d, shape.d + shape.nbDims);
		}
		launch(*slot, std::move(key), buffers_moved);

		guard.slot = nullptr; // queued successfully; the caller owns the slot now
		return slot;
	}

	void TensorRTRunner::record_work(Slot& slot)
	{
		for (size_t i : _input_order)
		{
			IOBuffer& buffer = slot.buffers[i];
			check_cuda(cudaMemcpyAsync(buffer.device, buffer.host, buffer.bytes, cudaMemcpyHostToDevice, slot.stream),
				"cudaMemcpyAsync (host to device)");
		}

		if (!slot.context->enqueueV3(slot.stream))
			throw std::runtime_error("enqueueV3 failed");

		for (size_t i : _output_order)
		{
			IOBuffer& buffer = slot.buffers[i];
			check_cuda(cudaMemcpyAsync(buffer.host, buffer.device, buffer.bytes, cudaMemcpyDeviceToHost, slot.stream),
				"cudaMemcpyAsync (device to host)");
		}
	}

	void TensorRTRunner::launch(Slot& slot, std::vector<int64_t>&& key, bool buffers_moved)
	{
		const bool same_shapes = key == slot.graph_key;
		if (!same_shapes || buffers_moved)
		{
			// New shapes or new buffers: the recorded graph no longer applies.
			destroy_graph(slot);
			slot.ran_direct = false;
			slot.graph_key = std::move(key);
		}

		if (slot.graph)
		{
			check_cuda(cudaGraphLaunch(slot.graph, slot.stream), "cudaGraphLaunch");
			return;
		}

		// TensorRT may do deferred setup in the first enqueueV3 after a shape
		// change, so a shape is captured only once it has run directly.
		if (_params.cudaGraph && !slot.graph_unsupported && slot.ran_direct)
		{
			if (capture_graph(slot))
			{
				check_cuda(cudaGraphLaunch(slot.graph, slot.stream), "cudaGraphLaunch");
				return;
			}
			slot.graph_unsupported = true;
			sample::gLogWarning << "CUDA graph capture is not supported for this engine; using enqueueV3 directly" << std::endl;
		}

		record_work(slot);
		slot.ran_direct = true;
	}

	bool TensorRTRunner::capture_graph(Slot& slot)
	{
		// Thread-local mode: other threads may allocate (other slots growing
		// their buffers) while this stream is being captured.
		if (cudaStreamBeginCapture(slot.stream, cudaStreamCaptureModeThreadLocal) != cudaSuccess)
		{
			cudaGetLastError();
			return false;
		}

		bool recorded = true;
		try
		{
			record_work(slot);
		}
		catch (const std::exception& e)
		{
			sample::gLogWarning << "CUDA graph capture failed: " << e.what() << std::endl;
			recorded = false;
		}

		// Always end the capture, or the stream stays unusable.
		cudaGraph_t graph = nullptr;
		const cudaError_t end_status = cudaStreamEndCapture(slot.stream, &graph);
		cudaGraphExec_t exec = nullptr;
		const bool ok = recorded && end_status == cudaSuccess && graph
			&& cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
		if (graph)
			cudaGraphDestroy(graph);
		if (!ok)
		{
			if (exec)
				cudaGraphExecDestroy(exec);
			cudaGetLastError(); // capture errors are not sticky; clear them
			return false;
		}
		slot.graph = exec;
		return true;
	}

	void TensorRTRunner::destroy_graph(Slot& slot) noexcept
	{
		if (slot.graph)
		{
			cudaGraphExecDestroy(slot.graph);
			slot.graph = nullptr;
		}
	}

	std::vector<Tensor> TensorRTRunner::collect_outputs(const Slot& slot) const
	{
		std::vector<Tensor> outputs;
		outputs.reserve(_output_order.size());
		for (size_t i : _output_order)
		{
			const IOBuffer& buffer = slot.buffers[i];
			const auto* bytes = static_cast<const unsigned char*>(buffer.host);
			Tensor out;
			out.name = _io[i].name;
			out.shape.assign(buffer.shape.d, buffer.shape.d + buffer.shape.nbDims);
			out.data.assign(bytes, bytes + buffer.bytes);
			out.dtype = to_inferrs_data_type(_io[i].type);
			outputs.push_back(std::move(out));
		}
		return outputs;
	}

	void TensorRTRunner::push_completion(Completion completion)
	{
		{
			std::lock_guard<std::mutex> lock(_completions_mut);
			_completions.emplace(std::move(completion));
		}
		_completions_cond.notify_one();
	}

	std::vector<TensorView> TensorRTRunner::views_of(const std::vector<Tensor>& tensors)
	{
		std::vector<TensorView> views;
		views.reserve(tensors.size());
		for (const auto& t : tensors)
			views.push_back({ t.name, t.shape, t.dtype, t.data });
		return views;
	}

	void TensorRTRunner::process(const std::vector<Tensor>& input_tensors)
	{
		const auto views = views_of(input_tensors);
		process(std::span<const TensorView>(views));
	}

	void TensorRTRunner::process(std::span<const TensorView> input_tensors)
	{
		{
			std::lock_guard<std::mutex> lock(_completions_mut);
			++_pending_requests;
		}

		try
		{
			Slot* slot = enqueue(input_tensors);
			const cudaError_t status = cudaStreamSynchronize(slot->stream);
			if (status != cudaSuccess)
			{
				return_slot(slot);
				check_cuda(status, "TensorRT inference");
			}
			Completion outputs;
			try
			{
				outputs = collect_outputs(*slot);
			}
			catch (...)
			{
				return_slot(slot);
				throw;
			}
			return_slot(slot);
			push_completion(std::move(outputs));
		}
		catch (...)
		{
			finish_request();
			throw;
		}

		finish_request();
	}

	void TensorRTRunner::process_async(const std::vector<Tensor>& input_tensors)
	{
		const auto views = views_of(input_tensors);
		process_async(std::span<const TensorView>(views));
	}

	void TensorRTRunner::process_async(std::span<const TensorView> input_tensors)
	{
		{
			std::lock_guard<std::mutex> lock(_completions_mut);
			++_pending_requests;
		}

		Slot* slot = nullptr;
		try
		{
			slot = enqueue(input_tensors);
		}
		catch (...)
		{
			finish_request();
			throw;
		}

		{
			std::lock_guard<std::mutex> lock(_inflight_mut);
			++_inflight;
		}
		// The callback runs on a CUDA driver thread once copy-out has finished.
		// It must not call CUDA, so it only copies the pinned outputs and
		// hands the slot back.
		const cudaError_t status = cudaStreamAddCallback(slot->stream, &TensorRTRunner::on_stream_complete, slot, 0);
		if (status != cudaSuccess)
		{
			// Could not register the callback: finish the request here instead.
			on_stream_complete(slot->stream, cudaStreamSynchronize(slot->stream), slot);
		}
	}

	void CUDART_CB TensorRTRunner::on_stream_complete(cudaStream_t, cudaError_t status, void* user_data)
	{
		Slot* slot = static_cast<Slot*>(user_data);
		TensorRTRunner* self = slot->owner;

		Completion completion;
		if (status != cudaSuccess)
		{
			completion = std::string("Asynchronous inference failed: ") + cudaGetErrorString(status);
		}
		else
		{
			try
			{
				completion = self->collect_outputs(*slot);
			}
			catch (const std::exception& e)
			{
				completion = std::string("Asynchronous inference failed: ") + e.what();
			}
		}

		// The stream has drained up to this callback, so the slot is free.
		self->return_slot(slot);
		try
		{
			self->push_completion(std::move(completion));
		}
		catch (...)
		{
			sample::gLogError << "process_async: could not queue the result" << std::endl;
		}
		self->finish_request();

		std::lock_guard<std::mutex> lock(self->_inflight_mut);
		--self->_inflight;
		self->_inflight_cond.notify_all();
	}

	void TensorRTRunner::get(std::vector<Tensor>& result)
	{
		std::unique_lock<std::mutex> completions_lock(_completions_mut);
		_completions_cond.wait(completions_lock, [&]() {
			return !_completions.empty() || _pending_requests == 0;
		});
		if (_completions.empty())
			throw std::runtime_error("No inference request is pending");

		Completion completion = std::move(_completions.front());
		_completions.pop();
		completions_lock.unlock();

		if (auto* error = std::get_if<std::string>(&completion))
			throw std::runtime_error(*error);
		result = std::move(std::get<std::vector<Tensor>>(completion));
	}

	void TensorRTRunner::finish_request()
	{
		{
			std::lock_guard<std::mutex> lock(_completions_mut);
			--_pending_requests;
		}
		_completions_cond.notify_all();
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

	size_t TensorRTRunner::get_tensor_data_size(const nvinfer1::DataType& data_type) const
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
		// 4-bit types are packed two elements per byte; report the one-byte
		// storage unit, matching the public API contract.
		case nvinfer1::DataType::kINT4:       return 1;
		case nvinfer1::DataType::kFP4:        return 1;
		default:                              return 0;
		}
	}

	std::string TensorRTRunner::get_input_tensor_name(uint32_t index) const
	{
		if (!_params.inputTensorNames.empty())
		{
			if (index >= _params.inputTensorNames.size())
				throw std::out_of_range("Input tensor index is outside input_tensor_names");
			return _params.inputTensorNames[index];
		}

		std::vector<std::string> input_names;

		int numIOTensors = _engine->getNbIOTensors();

		for (int i = 0; i < numIOTensors; ++i) {
			const char* tensorName = _engine->getIOTensorName(i);

			if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kINPUT) {
				input_names.push_back(tensorName);
			}
		}

		if (index >= input_names.size())
			throw std::out_of_range("Input tensor index is outside the engine's input tensors");
		return input_names[index];
	}

	std::string TensorRTRunner::get_output_tensor_name(uint32_t index) const
	{
		if (!_params.outputTensorNames.empty())
		{
			if (index >= _params.outputTensorNames.size())
				throw std::out_of_range("Output tensor index is outside output_tensor_names");
			return _params.outputTensorNames[index];
		}

		std::vector<std::string> output_names;

		int numIOTensors = _engine->getNbIOTensors();

		for (int i = 0; i < numIOTensors; ++i) {
			const char* tensorName = _engine->getIOTensorName(i);

			if (_engine->getTensorIOMode(tensorName) == TensorIOMode::kOUTPUT) {
				output_names.push_back(tensorName);
			}
		}

		if (index >= output_names.size())
			throw std::out_of_range("Output tensor index is outside the engine's output tensors");
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
		for (const auto& [layer_name, device_type] : _params.layerDeviceTypes)
		{
			bool found = false;
			for (int32_t i = 0; i < network->getNbLayers(); ++i)
			{
				auto* layer = network->getLayer(i);
				if (layer_name == layer->getName())
				{
					config->setDeviceType(layer, device_type);
					found = true;
					break;
				}
			}
			if (!found)
				throw std::runtime_error("Unknown TensorRT layer in layer_device_types: " + layer_name);
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

			samplesCommon::enableDLA(builder.get(), config.get(), _params.dlaCore, _params.allowGpuFallback);

		return true;
	}

	bool TensorRTRunner::load_engine(std::vector<char>& data) {
		std::string engineFilePath = _params.engineFile.empty()
			? engine_path_for(_params)
			: _params.engineFile;
		std::ifstream file(engineFilePath, std::ios::binary);
		if (!file) {
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
		std::string planFilePath = _params.engineFile.empty()
			? engine_path_for(_params)
			: _params.engineFile;
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

	std::expected<std::vector<Tensor>, std::string> TensorRTRunner::get_input_tensors_info() const
	{
		try
		{
			// get_input_tensor_name() reads input_tensor_names when it is set,
			// so iterate that list rather than the engine's input count.
			const size_t n_inputs = _params.inputTensorNames.empty()
				? get_number_of_inputs() : _params.inputTensorNames.size();
			std::vector<Tensor> tensors;
			tensors.reserve(n_inputs);

			for (uint32_t i = 0; i < n_inputs; ++i)
			{
				Tensor t;
				t.name = get_input_tensor_name(i);
				auto dims = get_input_shape(i);
				t.shape.assign(dims.begin(), dims.end());
				t.dtype = to_inferrs_data_type(get_tensor_data_type(t.name));
				tensors.push_back(std::move(t));
			}

			return tensors;
		}
		catch(const std::exception& e)
		{
			return std::unexpected(e.what());
		}
	}

	std::expected<std::vector<Tensor>, std::string> TensorRTRunner::get_output_tensors_info() const
	{
		try
		{
			// get_output_tensor_name() reads output_tensor_names when it is set,
			// so iterate that list rather than the engine's output count.
			const size_t n_outputs = _params.outputTensorNames.empty()
				? get_number_of_outputs() : _params.outputTensorNames.size();
			std::vector<Tensor> tensors;
			tensors.reserve(n_outputs);

			for (uint32_t i = 0; i < n_outputs; ++i)
			{
				Tensor t;
				t.name = get_output_tensor_name(i);
				auto dims = get_output_shape(i);
				t.shape.assign(dims.begin(), dims.end());
				t.dtype = to_inferrs_data_type(get_tensor_data_type(t.name));
				tensors.push_back(std::move(t));
			}

			return tensors;
		}
		catch(const std::exception& e)
		{
			return std::unexpected(e.what());
		}
	}

	RunnerParams TensorRTRunner::get_config() const
	{
		return _params;
	}
	
} // namespace inferrs
