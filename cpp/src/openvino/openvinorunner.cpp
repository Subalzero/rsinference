#include "openvinorunner.h"

#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rsinfer
{
    namespace
    {
        ov::element::Type to_ov_element_type(DataType data_type, ov::element::Type model_type)
        {
            switch (data_type)
            {
            case DataType::FP16: return ov::element::f16;
            case DataType::FP32: return ov::element::f32;
            case DataType::INT8:
                if (model_type == ov::element::i8 || model_type == ov::element::u8) return model_type;
                return ov::element::i8;
            case DataType::BF16: return ov::element::bf16;
            case DataType::INT4: return ov::element::i4;
            case DataType::FP4: return ov::element::f4e2m1;
            default: throw std::invalid_argument("Unsupported OpenVINO tensor data type");
            }
        }

        DataType from_ov_element_type(ov::element::Type type)
        {
            if (type == ov::element::f16) return DataType::FP16;
            if (type == ov::element::f32) return DataType::FP32;
            if (type == ov::element::i8 || type == ov::element::u8) return DataType::INT8;
            if (type == ov::element::bf16) return DataType::BF16;
            if (type == ov::element::i4) return DataType::INT4;
            if (type == ov::element::f4e2m1) return DataType::FP4;
            return DataType::UNKNOWN;
        }

        size_t tensor_byte_count(DataType data_type, size_t element_count)
        {
            switch (data_type)
            {
            case DataType::FP32: return element_count * sizeof(float);
            case DataType::FP16:
            case DataType::BF16: return element_count * sizeof(uint16_t);
            case DataType::INT8: return element_count;
            case DataType::INT4:
            case DataType::FP4: return (element_count + 1) / 2;
            default: throw std::invalid_argument("Unsupported OpenVINO tensor data type");
            }
        }

        ov::Shape to_ov_shape(const std::vector<int64_t>& dims)
        {
            ov::Shape shape;
            shape.reserve(dims.size());
            for (int64_t dim : dims)
            {
                if (dim < 0 || static_cast<uint64_t>(dim) > std::numeric_limits<size_t>::max())
                    throw std::invalid_argument("Tensor shape contains an invalid dimension");
                shape.push_back(static_cast<size_t>(dim));
            }
            return shape;
        }

        std::vector<int> to_public_shape(const ov::PartialShape& shape)
        {
            std::vector<int> result;
            result.reserve(shape.size());
            for (const auto& dim : shape)
            {
                if (!dim.is_static())
                    result.push_back(-1);
                else if (dim.get_length() > std::numeric_limits<int>::max())
                    throw std::overflow_error("OpenVINO tensor dimension exceeds int range");
                else
                    result.push_back(static_cast<int>(dim.get_length()));
            }
            return result;
        }

        OpenVINORunnerParams make_params(const std::string& model_xml, const std::string& device)
        {
            OpenVINORunnerParams params;
            params.modelXmlPath = model_xml;
            params.device = device;
            return params;
        }
    }

    ov::Core& OpenVINORunner::core()
    {
        // Leaked on purpose: see the declaration in openvinorunner.h.
        static ov::Core* instance = new ov::Core();
        return *instance;
    }

    OpenVINORunner::OpenVINORunner() = default;

    OpenVINORunner::~OpenVINORunner()
    {
        std::unique_lock<std::mutex> lock(_completions_mut);
        _completions_cond.wait(lock, [this]() { return _active_calls == 0; });
    }

    OpenVINORunner::OpenVINORunner(const OpenVINORunnerParams& params)
        : _params(params)
    {
        if (_params.modelXmlPath.empty())
            throw std::invalid_argument("OpenVINO modelXmlPath must not be empty");
        if (_params.device.empty())
            throw std::invalid_argument("OpenVINO device must not be empty");

        std::shared_ptr<ov::Model> model = _params.modelWeightsPath.empty()
            ? core().read_model(_params.modelXmlPath)
            : core().read_model(_params.modelXmlPath, _params.modelWeightsPath);
        ov::AnyMap compile_properties = _params.compileProperties;
        const auto add_property = [&compile_properties](const auto& property, const auto& value)
        {
            if (value.has_value())
                compile_properties[property.name()] = *value;
        };

        add_property(ov::hint::performance_mode, _params.performanceMode);
        add_property(ov::hint::inference_precision, _params.inferencePrecision);
        add_property(ov::hint::execution_mode, _params.executionMode);
        add_property(ov::hint::model_priority, _params.modelPriority);
        add_property(ov::hint::num_requests, _params.performanceHintNumRequests);
        add_property(ov::num_streams, _params.numStreams);
        add_property(ov::inference_num_threads, _params.inferenceNumThreads);
        add_property(ov::compilation_num_threads, _params.compilationNumThreads);
        add_property(ov::hint::scheduling_core_type, _params.schedulingCoreType);
        add_property(ov::hint::enable_cpu_pinning, _params.enableCpuPinning);
        add_property(ov::hint::enable_hyper_threading, _params.enableHyperThreading);
        add_property(ov::enable_profiling, _params.enableProfiling);
        add_property(ov::cache_dir, _params.cacheDir);

        _compiled_model = core().compile_model(model, _params.device, compile_properties);

        // The public Tensor adapter has a deliberately limited dtype enum.
        // Fail at model load instead of returning UNKNOWN or failing only when
        // an inference request tries to copy a tensor.
        for (const auto& port : _compiled_model.inputs())
        {
            if (from_ov_element_type(port.get_element_type()) == DataType::UNKNOWN)
                throw std::invalid_argument("Unsupported OpenVINO input tensor element type: " +
                    port.get_element_type().get_type_name());
        }
        for (const auto& port : _compiled_model.outputs())
        {
            if (from_ov_element_type(port.get_element_type()) == DataType::UNKNOWN)
                throw std::invalid_argument("Unsupported OpenVINO output tensor element type: " +
                    port.get_element_type().get_type_name());
        }

        const uint32_t request_count = _params.numInferRequests == 0
            ? static_cast<uint32_t>(_compiled_model.get_property(ov::optimal_number_of_infer_requests))
            : _params.numInferRequests;
        for (uint32_t i = 0; i < request_count; ++i)
        {
            _infer_requests.push_back(_compiled_model.create_infer_request());
        }
        for (auto& request : _infer_requests)
        {
            _idle_requests.push(&request);
        }
    }

    OpenVINORunner::OpenVINORunner(const std::string& xml_path, const std::string& default_device)
        : OpenVINORunner(make_params(xml_path, default_device))
    {
    }

    OpenVINORunner::OpenVINORunner(OpenVINORunner&& temp) noexcept
    {
        std::unique_lock<std::mutex> lock(temp._completions_mut);
        temp._completions_cond.wait(lock, [&temp]() { return temp._active_calls == 0; });
        _params = std::move(temp._params);
        _compiled_model = std::move(temp._compiled_model);
        _infer_requests = std::move(temp._infer_requests);
        _idle_requests = std::move(temp._idle_requests);
        _completions = std::move(temp._completions);
        _pending_requests = std::exchange(temp._pending_requests, 0);
    }

    OpenVINORunner& OpenVINORunner::operator=(OpenVINORunner&& temp) noexcept
    {
        if (this == &temp)
            return *this;

        {
            std::unique_lock<std::mutex> lock(_completions_mut);
            _completions_cond.wait(lock, [this]() { return _active_calls == 0; });
        }
        {
            std::unique_lock<std::mutex> lock(temp._completions_mut);
            temp._completions_cond.wait(lock, [&temp]() { return temp._active_calls == 0; });
        }

        _params = std::move(temp._params);
        _compiled_model = std::move(temp._compiled_model);
        _infer_requests = std::move(temp._infer_requests);
        _idle_requests = std::move(temp._idle_requests);
        _completions = std::move(temp._completions);
        _pending_requests = std::exchange(temp._pending_requests, 0);
        return *this;
    }

    std::vector<int> OpenVINORunner::get_input_shape(uint32_t index) const
    {
        if (index >= _compiled_model.inputs().size())
            throw std::out_of_range("OpenVINO input tensor index is out of range");
        return to_public_shape(_compiled_model.input(index).get_partial_shape());
    }

    std::vector<int> OpenVINORunner::get_output_shape(uint32_t index) const
    {
        if (index >= _compiled_model.outputs().size())
            throw std::out_of_range("OpenVINO output tensor index is out of range");
        return to_public_shape(_compiled_model.output(index).get_partial_shape());
    }

    std::string OpenVINORunner::get_input_tensor_name(uint32_t index) const
    {
        if (index >= _compiled_model.inputs().size())
            throw std::out_of_range("OpenVINO input tensor index is out of range");
        return _compiled_model.input(index).get_any_name();
    }

    std::string OpenVINORunner::get_output_tensor_name(uint32_t index) const
    {
        if (index >= _compiled_model.outputs().size())
            throw std::out_of_range("OpenVINO output tensor index is out of range");
        return _compiled_model.output(index).get_any_name();
    }

    size_t OpenVINORunner::get_number_of_inputs() const { return _compiled_model.inputs().size(); }
    size_t OpenVINORunner::get_number_of_outputs() const { return _compiled_model.outputs().size(); }

    ov::element::Type OpenVINORunner::get_tensor_element_type(const std::string& tensor_name) const
    {
        for (const auto& input : _compiled_model.inputs())
        {
            const auto names = input.get_names();
            if (names.count(tensor_name)) return input.get_element_type();
        }
        for (const auto& output : _compiled_model.outputs())
        {
            const auto names = output.get_names();
            if (names.count(tensor_name)) return output.get_element_type();
        }
        throw std::invalid_argument("Unknown OpenVINO tensor name: " + tensor_name);
    }

    rsinfer::DataType OpenVINORunner::get_tensor_data_type(const std::string& tensor_name) const
    {
        return from_ov_element_type(get_tensor_element_type(tensor_name));
    }

    size_t OpenVINORunner::get_tensor_data_size(const rsinfer::DataType& data_type) const
    {
        switch (data_type)
        {
        case DataType::FP32: return 4;
        case DataType::FP16:
        case DataType::BF16: return 2;
        case DataType::INT8: return 1;
        case DataType::INT4:
        case DataType::FP4: return 2;
        default: throw std::invalid_argument("Unsupported OpenVINO tensor data type");
        }
    }

    void OpenVINORunner::begin_request()
    {
        std::lock_guard<std::mutex> lock(_completions_mut);
        if (_params.modelXmlPath.empty())
            throw std::runtime_error("OpenVINO runner is not initialized");
        ++_pending_requests;
        ++_active_calls;
    }

    void OpenVINORunner::finish_request() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(_completions_mut);
            --_pending_requests;
            --_active_calls;
        }
        _completions_cond.notify_all();
    }

    std::vector<rsinfer::Tensor> OpenVINORunner::process_impl(const std::vector<rsinfer::Tensor>& input_tensors)
    {
        if (input_tensors.size() != _compiled_model.inputs().size())
            throw std::invalid_argument("Input tensor count does not match the OpenVINO model");

        ov::InferRequest* request = nullptr;
        {
            std::unique_lock<std::mutex> lock(_idle_requests_mut);
            _idle_requests_cond.wait(lock, [this]() { return !_idle_requests.empty(); });
            request = _idle_requests.front();
            _idle_requests.pop();
        }

        struct RequestReturn
        {
            OpenVINORunner* runner;
            ov::InferRequest* request;
            ~RequestReturn()
            {
                std::lock_guard<std::mutex> lock(runner->_idle_requests_mut);
                runner->_idle_requests.push(request);
                runner->_idle_requests_cond.notify_one();
            }
        } return_request{ this, request };

        std::vector<bool> assigned(_compiled_model.inputs().size(), false);
        for (size_t input_pos = 0; input_pos < input_tensors.size(); ++input_pos)
        {
            const auto& input = input_tensors[input_pos];
            size_t port_index = input_pos;
            if (!input.name.empty())
            {
                port_index = _compiled_model.inputs().size();
                for (size_t i = 0; i < _compiled_model.inputs().size(); ++i)
                {
                    if (_compiled_model.input(i).get_names().count(input.name))
                    {
                        port_index = i;
                        break;
                    }
                }
            }
            if (port_index >= _compiled_model.inputs().size() || assigned[port_index])
                throw std::invalid_argument("Input tensor name is unknown or duplicated: " + input.name);

            const auto& port = _compiled_model.input(port_index);
            const ov::Shape shape = to_ov_shape(input.shape);
            if (!port.get_partial_shape().compatible(ov::PartialShape(shape)))
                throw std::invalid_argument("Input tensor shape does not match model input: " + port.get_any_name());

            const ov::element::Type input_type = to_ov_element_type(input.dtype, port.get_element_type());
            if (input_type != port.get_element_type())
                throw std::invalid_argument("Input tensor data type does not match model input: " + port.get_any_name());
            const size_t expected_bytes = tensor_byte_count(input.dtype, ov::shape_size(shape));
            if (input.data.size() != expected_bytes)
                throw std::invalid_argument("Input tensor byte size does not match its shape: " + port.get_any_name());

            ov::Tensor input_tensor(input_type, shape, const_cast<unsigned char*>(input.data.data()));
            request->set_tensor(port, input_tensor);
            assigned[port_index] = true;
        }
        request->infer();

        std::vector<rsinfer::Tensor> outputs;
        outputs.reserve(_compiled_model.outputs().size());
        for (size_t i = 0; i < _compiled_model.outputs().size(); ++i)
        {
            const auto& port = _compiled_model.output(i);
            ov::Tensor output_tensor = request->get_tensor(port);
            rsinfer::Tensor output;
            output.name = port.get_any_name();
            output.dtype = from_ov_element_type(output_tensor.get_element_type());
            const ov::Shape output_shape = output_tensor.get_shape();
            output.shape.assign(output_shape.begin(), output_shape.end());
            output.data.resize(output_tensor.get_byte_size());
            if (!output.data.empty())
                std::memcpy(output.data.data(), output_tensor.data(), output.data.size());
            outputs.push_back(std::move(output));
        }
        return outputs;
    }

    void OpenVINORunner::process(const std::vector<rsinfer::Tensor>& input_tensors)
    {
        begin_request();
        try
        {
            auto result = process_impl(input_tensors);
            {
                std::lock_guard<std::mutex> lock(_completions_mut);
                _completions.emplace(std::move(result));
            }
            _completions_cond.notify_one();
        }
        catch (...)
        {
            finish_request();
            throw;
        }
        finish_request();
    }

    void OpenVINORunner::process_async(const std::vector<rsinfer::Tensor>& input_tensors)
    {
        begin_request();
        try
        {
            std::thread([this](std::vector<rsinfer::Tensor> inputs)
            {
                try
                {
                    auto result = process_impl(inputs);
                    {
                        std::lock_guard<std::mutex> lock(_completions_mut);
                        _completions.emplace(std::move(result));
                    }
                }
                catch (const std::exception& e)
                {
                    std::lock_guard<std::mutex> lock(_completions_mut);
                    _completions.emplace(std::string(e.what()));
                }
                catch (...)
                {
                    std::lock_guard<std::mutex> lock(_completions_mut);
                    _completions.emplace(std::string("unknown asynchronous inference failure"));
                }
                finish_request();
            }, input_tensors).detach();
        }
        catch (...)
        {
            finish_request();
            throw;
        }
    }

    void OpenVINORunner::get(std::vector<rsinfer::Tensor>& output_tensors)
    {
        std::unique_lock<std::mutex> lock(_completions_mut);
        _completions_cond.wait(lock, [this]() {
            return !_completions.empty() || _pending_requests == 0;
        });
        if (_completions.empty())
            throw std::runtime_error("No inference request is pending");

        Completion completion = std::move(_completions.front());
        _completions.pop();
        lock.unlock();
        if (const auto* error = std::get_if<std::string>(&completion))
            throw std::runtime_error(*error);
        output_tensors = std::move(std::get<std::vector<rsinfer::Tensor>>(completion));
    }

    std::vector<std::string> OpenVINORunner::get_available_devices()
    {
        return core().get_available_devices();
    }

    std::string OpenVINORunner::get_device_name(const std::string& device)
    {
        return core().get_property(device, ov::device::full_name);
    }
}
