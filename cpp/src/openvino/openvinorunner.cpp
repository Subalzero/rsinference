#include "openvinorunner.h"
#include "common/datatypeconversion.h"

#include <cstring>
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>

namespace rsinfer
{
    namespace
    {
        ov::element::Type to_ov_element_type(DataType data_type)
        {
            switch (data_type)
            {
            case DataType::FP16: return ov::element::f16;
            case DataType::FP32: return ov::element::f32;
            case DataType::INT64: return ov::element::i64;
            case DataType::INT32: return ov::element::i32;
            case DataType::INT8: return ov::element::i8;
            case DataType::UINT8: return ov::element::u8;
            case DataType::BOOL: return ov::element::boolean;
            case DataType::FP8: return ov::element::f8e4m3;
            case DataType::FP8E5M2: return ov::element::f8e5m2;
            case DataType::E8M0: return ov::element::f8e8m0;
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
            if (type == ov::element::i32) return DataType::INT32;
            if (type == ov::element::i64) return DataType::INT64;
            if (type == ov::element::i8) return DataType::INT8;
            if (type == ov::element::u8) return DataType::UINT8;
            if (type == ov::element::boolean) return DataType::BOOL;
            if (type == ov::element::f8e4m3) return DataType::FP8;
            if (type == ov::element::f8e5m2) return DataType::FP8E5M2;
            if (type == ov::element::f8e8m0) return DataType::E8M0;
            if (type == ov::element::bf16) return DataType::BF16;
            if (type == ov::element::i4) return DataType::INT4;
            if (type == ov::element::f4e2m1) return DataType::FP4;
            return DataType::UNKNOWN;
        }

        size_t tensor_byte_count(DataType data_type, size_t element_count)
        {
            if (data_type_element_size(data_type) == 0)
                throw std::invalid_argument("Unsupported OpenVINO tensor data type");
            return packed_byte_count(data_type, element_count);
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

        std::vector<int64_t> to_info_shape(const ov::PartialShape& shape)
        {
            if (shape.rank().is_dynamic())
                return {};
            std::vector<int64_t> result;
            result.reserve(shape.size());
            for (const auto& dim : shape)
                result.push_back(dim.is_static() ? dim.get_length() : -1);
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
        {
            std::unique_lock<std::mutex> lock(_completions_mut);
            _completions_cond.wait(lock, [this]() { return _active_calls == 0; });
        }
        // A callback may still be returning after it signalled completion;
        // wait() only returns once the request has fully finished.
        for (auto& slot : _slots)
        {
            if (!slot->started_async)
                continue;
            try
            {
                slot->request.wait();
            }
            catch (...)
            {
                // Its error was already reported through get().
            }
        }
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
            auto slot = std::make_unique<Slot>();
            slot->owner = this;
            slot->request = _compiled_model.create_infer_request();
            slot->owned_inputs.resize(_compiled_model.inputs().size());
            Slot* raw = slot.get();
            slot->request.set_callback([raw](std::exception_ptr error) { raw->owner->on_async_complete(*raw, error); });
            _idle_slots.push(raw);
            _slots.push_back(std::move(slot));
        }
    }

    OpenVINORunner::OpenVINORunner(const std::string& xml_path, const std::string& default_device)
        : OpenVINORunner(make_params(xml_path, default_device))
    {
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
        const size_t size = data_type_element_size(data_type);
        if (size == 0) throw std::invalid_argument("Unsupported OpenVINO tensor data type");
        return size;
    }

    std::expected<std::vector<rsinfer::Tensor>, std::string> OpenVINORunner::get_input_tensors_info() const
    {
        try
        {
            std::vector<rsinfer::Tensor> inputs;
            inputs.reserve(_compiled_model.inputs().size());
            for (const auto& port : _compiled_model.inputs())
            {
                rsinfer::Tensor input;
                input.name = port.get_any_name();
                input.dtype = from_ov_element_type(port.get_element_type());
                input.shape = to_info_shape(port.get_partial_shape());
                inputs.push_back(std::move(input));
            }
            return inputs;
        }
        catch (const std::exception& e)
        {
            return std::unexpected(e.what());
        }
    }

    std::expected<std::vector<rsinfer::Tensor>, std::string> OpenVINORunner::get_output_tensors_info() const
    {
        try
        {
            std::vector<rsinfer::Tensor> outputs;
            outputs.reserve(_compiled_model.outputs().size());
            for (const auto& port : _compiled_model.outputs())
            {
                rsinfer::Tensor output;
                output.name = port.get_any_name();
                output.dtype = from_ov_element_type(port.get_element_type());
                output.shape = to_info_shape(port.get_partial_shape());
                outputs.push_back(std::move(output));
            }
            return outputs;
        }
        catch (const std::exception& e)
        {
            return std::unexpected(e.what());
        }
    }

    OpenVINORunnerParams OpenVINORunner::get_config() const
    {
        return _params;
    }

    uint64_t OpenVINORunner::begin_request()
    {
        std::lock_guard<std::mutex> lock(_completions_mut);
        if (_params.modelXmlPath.empty())
            throw std::runtime_error("OpenVINO runner is not initialized");
        ++_active_calls;
        // The ticket fixes this request's place in get()'s order.
        return _completions.reserve();
    }

    void OpenVINORunner::finish_request() noexcept
    {
        {
            std::lock_guard<std::mutex> lock(_completions_mut);
            --_active_calls;
        }
        _completions_cond.notify_all();
    }

    void OpenVINORunner::complete_request(uint64_t ticket, Completion completion) noexcept
    {
        {
            std::lock_guard<std::mutex> lock(_completions_mut);
            try
            {
                _completions.complete(ticket, std::move(completion));
            }
            catch (...)
            {
                _completions.cancel(ticket);
            }
        }
        _completions_cond.notify_all();
    }

    void OpenVINORunner::cancel_request(uint64_t ticket) noexcept
    {
        {
            std::lock_guard<std::mutex> lock(_completions_mut);
            _completions.cancel(ticket);
        }
        _completions_cond.notify_all();
    }

    OpenVINORunner::Slot* OpenVINORunner::acquire_slot()
    {
        Slot* slot = nullptr;
        {
            std::unique_lock<std::mutex> lock(_idle_slots_mut);
            if (_slots.empty())
                throw std::runtime_error("OpenVINO runner has no infer requests");
            _idle_slots_cond.wait(lock, [this]() { return !_idle_slots.empty(); });
            slot = _idle_slots.front();
            _idle_slots.pop();
        }
        // An async request hands its slot back from its callback, which may not
        // have returned yet; until it has, the request reports itself busy.
        if (slot->started_async)
        {
            try
            {
                slot->request.wait();
            }
            catch (...)
            {
                // Its error was already reported through get().
            }
            slot->started_async = false;
        }
        return slot;
    }

    void OpenVINORunner::return_slot(Slot* slot) noexcept
    {
        {
            std::lock_guard<std::mutex> lock(_idle_slots_mut);
            _idle_slots.push(slot);
        }
        _idle_slots_cond.notify_one();
    }

    void OpenVINORunner::bind_inputs(Slot& slot, std::span<const OpenVINOInputView> input_tensors, bool borrow)
    {
        if (input_tensors.size() != _compiled_model.inputs().size())
            throw std::invalid_argument("Input tensor count does not match the OpenVINO model");

        std::vector<bool> assigned(_compiled_model.inputs().size(), false);
        for (size_t input_pos = 0; input_pos < input_tensors.size(); ++input_pos)
        {
            const auto& input = input_tensors[input_pos];
            size_t port_index = input_pos;
            if (!input.name.empty())
            {
                port_index = _compiled_model.inputs().size();
                const std::string name(input.name);
                for (size_t i = 0; i < _compiled_model.inputs().size(); ++i)
                {
                    if (_compiled_model.input(i).get_names().count(name))
                    {
                        port_index = i;
                        break;
                    }
                }
            }
            if (port_index >= _compiled_model.inputs().size() || assigned[port_index])
                throw std::invalid_argument("Input tensor name is unknown or duplicated: " + std::string(input.name));

            const auto& port = _compiled_model.input(port_index);
            if (!port.get_partial_shape().compatible(ov::PartialShape(input.shape)))
                throw std::invalid_argument("Input tensor shape does not match model input: " + port.get_any_name());

            const ov::element::Type input_type = to_ov_element_type(input.dtype);
            if (input_type != port.get_element_type())
                throw std::invalid_argument("Input tensor data type does not match model input: " + port.get_any_name());
            const size_t expected_bytes = tensor_byte_count(input.dtype, ov::shape_size(input.shape));
            if (input.data.size() != expected_bytes)
                throw std::invalid_argument("Input tensor byte size does not match its shape: " + port.get_any_name());

            if (borrow)
            {
                // The request only reads it, and only until infer() returns.
                ov::Tensor input_tensor(input_type, input.shape, const_cast<unsigned char*>(input.data.data()));
                slot.request.set_tensor(port, input_tensor);
            }
            else
            {
                ov::Tensor& owned = slot.owned_inputs[port_index];
                if (!owned || owned.get_element_type() != input_type || owned.get_shape() != input.shape)
                    owned = ov::Tensor(input_type, input.shape);
                if (!input.data.empty())
                    std::memcpy(owned.data(), input.data.data(), input.data.size());
                slot.request.set_tensor(port, owned);
            }
            assigned[port_index] = true;
        }
    }

    std::vector<rsinfer::Tensor> OpenVINORunner::collect_outputs(Slot& slot) const
    {
        std::vector<rsinfer::Tensor> outputs;
        outputs.reserve(_compiled_model.outputs().size());
        for (size_t i = 0; i < _compiled_model.outputs().size(); ++i)
        {
            const auto& port = _compiled_model.output(i);
            ov::Tensor output_tensor = slot.request.get_tensor(port);
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

    std::vector<OpenVINOInputView> OpenVINORunner::views_of(const std::vector<rsinfer::Tensor>& tensors)
    {
        std::vector<OpenVINOInputView> views;
        views.reserve(tensors.size());
        for (const auto& t : tensors)
            views.push_back({ t.name, to_ov_shape(t.shape), t.dtype, t.data });
        return views;
    }

    void OpenVINORunner::process(const std::vector<rsinfer::Tensor>& input_tensors)
    {
        const auto views = views_of(input_tensors);
        process(std::span<const OpenVINOInputView>(views));
    }

    void OpenVINORunner::process(std::span<const OpenVINOInputView> input_tensors)
    {
        const uint64_t ticket = begin_request();
        Slot* slot = nullptr;
        try
        {
            slot = acquire_slot();
            bind_inputs(*slot, input_tensors, /*borrow=*/true);
            slot->request.infer();
            // Queue the result before freeing the request, as the async path does.
            complete_request(ticket, collect_outputs(*slot));
            return_slot(slot);
        }
        catch (...)
        {
            if (slot)
                return_slot(slot);
            // The error goes to this caller; get() skips the ticket.
            cancel_request(ticket);
            finish_request();
            throw;
        }
        finish_request();
    }

    void OpenVINORunner::process_async(const std::vector<rsinfer::Tensor>& input_tensors)
    {
        const auto views = views_of(input_tensors);
        process_async(std::span<const OpenVINOInputView>(views));
    }

    void OpenVINORunner::process_async(std::span<const OpenVINOInputView> input_tensors)
    {
        const uint64_t ticket = begin_request();
        Slot* slot = nullptr;
        try
        {
            slot = acquire_slot();
            bind_inputs(*slot, input_tensors, /*borrow=*/false);
            slot->ticket = ticket;
            slot->async_pending = true;
            slot->started_async = true;
            slot->request.start_async();
        }
        catch (...)
        {
            if (slot)
            {
                slot->async_pending = false;
                slot->started_async = false;
                return_slot(slot);
            }
            cancel_request(ticket);
            finish_request();
            throw;
        }
    }

    void OpenVINORunner::on_async_complete(Slot& slot, std::exception_ptr error) noexcept
    {
        // Runs on an OpenVINO thread when the request finishes.
        if (!slot.async_pending)
            return;
        slot.async_pending = false;

        Completion completion;
        try
        {
            if (error)
                std::rethrow_exception(error);
            completion = collect_outputs(slot);
        }
        catch (const std::exception& e)
        {
            completion = std::string("OpenVINO asynchronous inference failed: ") + e.what();
        }
        catch (...)
        {
            completion = std::string("OpenVINO asynchronous inference failed");
        }

        // Queue the result before freeing the request (the ticket must be read
        // before another request can take the slot).
        complete_request(slot.ticket, std::move(completion));
        return_slot(&slot);
        finish_request();
    }

    void OpenVINORunner::get(std::vector<rsinfer::Tensor>& output_tensors)
    {
        std::unique_lock<std::mutex> lock(_completions_mut);
        _completions_cond.wait(lock, [this]() { return _completions.ready(); });
        std::optional<Completion> completion = _completions.take();
        lock.unlock();
        if (!completion)
            throw std::runtime_error("No inference request is pending");
        if (const auto* error = std::get_if<std::string>(&*completion))
            throw std::runtime_error(*error);
        output_tensors = std::move(std::get<std::vector<rsinfer::Tensor>>(*completion));
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
