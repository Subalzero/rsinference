#include "rsinference.h"
#include "openvinorunner.h"
#include "common/datatypeconversion.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    using json = nlohmann::json;

    enum ErrorCode
    {
        ERROR_CODE_SUCCESS = 0,
        ERROR_CODE_INVALID_ARGUMENTS = 1,
        ERROR_CODE_CREATION_FAILED = 2,
        ERROR_CODE_INFERENCE_FAILED = 3
    };

    void set_error(RSInferError* error, int code, const std::string& message)
    {
        if (!error) return;
        error->code = code;
        error->message = static_cast<char*>(std::malloc(message.size() + 1));
        if (error->message)
            std::memcpy(error->message, message.c_str(), message.size() + 1);
    }

    rsinfer::OpenVINORunner* get_runner(RSInferenceRunner* runner)
    {
        if (!runner || !runner->runner || runner->backend != RSInferenceBackend_OpenVINO)
            throw std::invalid_argument("Runner is null or is not an OpenVINO runner");
        return static_cast<rsinfer::OpenVINORunner*>(runner->runner);
    }


    char* duplicate_string(const std::string& value)
    {
        char* result = static_cast<char*>(std::malloc(value.size() + 1));
        if (result) std::memcpy(result, value.c_str(), value.size() + 1);
        return result;
    }

    std::string parse_config(const char* config, rsinfer::OpenVINORunnerParams& params)
    {
        if (!config || !*config) return {};
        const json j = json::parse(config, nullptr, false);
        if (j.is_discarded() || !j.is_object())
            return "json_config_str is not a valid JSON object";

        auto string_value = [&j](const char* key, std::string& target)
        {
            if (!j.contains(key)) return true;
            if (!j[key].is_string()) return false;
            target = j[key].get<std::string>();
            return true;
        };
        if (!string_value("device", params.device)) return "device must be a string";
        if (!string_value("model_weights_path", params.modelWeightsPath)) return "model_weights_path must be a string";
        if (j.contains("cache_dir"))
        {
            if (!j["cache_dir"].is_string()) return "cache_dir must be a string";
            params.cacheDir = j["cache_dir"].get<std::string>();
        }

        if (j.contains("num_infer_requests"))
        {
            const auto& value = j["num_infer_requests"];
            if ((!value.is_number_unsigned() && !value.is_number_integer()) ||
                (value.is_number_integer() && value.get<int64_t>() < 0) ||
                (value.is_number_unsigned() && value.get<uint64_t>() > std::numeric_limits<uint32_t>::max()) ||
                (value.is_number_integer() && value.get<int64_t>() > std::numeric_limits<uint32_t>::max()))
                return "num_infer_requests must be an unsigned 32-bit integer";
            params.numInferRequests = value.get<uint32_t>();
        }

        if (j.contains("performance_mode"))
        {
            if (!j["performance_mode"].is_string()) return "performance_mode must be a string";
            const auto value = j["performance_mode"].get<std::string>();
            if (value == "LATENCY") params.performanceMode = ov::hint::PerformanceMode::LATENCY;
            else if (value == "THROUGHPUT") params.performanceMode = ov::hint::PerformanceMode::THROUGHPUT;
            else if (value == "CUMULATIVE_THROUGHPUT") params.performanceMode = ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
            else return "performance_mode must be LATENCY, THROUGHPUT, or CUMULATIVE_THROUGHPUT";
        }

        if (j.contains("inference_precision"))
        {
            if (!j["inference_precision"].is_string()) return "inference_precision must be a string";
            const auto value = j["inference_precision"].get<std::string>();
            if (value == "f32") params.inferencePrecision = ov::element::f32;
            else if (value == "f16") params.inferencePrecision = ov::element::f16;
            else if (value == "bf16") params.inferencePrecision = ov::element::bf16;
            else return "inference_precision must be f32, f16, or bf16";
        }

        if (j.contains("execution_mode"))
        {
            if (!j["execution_mode"].is_string()) return "execution_mode must be a string";
            const auto value = j["execution_mode"].get<std::string>();
            if (value == "PERFORMANCE") params.executionMode = ov::hint::ExecutionMode::PERFORMANCE;
            else if (value == "ACCURACY") params.executionMode = ov::hint::ExecutionMode::ACCURACY;
            else return "execution_mode must be PERFORMANCE or ACCURACY";
        }

        if (j.contains("model_priority"))
        {
            if (!j["model_priority"].is_string()) return "model_priority must be a string";
            const auto value = j["model_priority"].get<std::string>();
            if (value == "LOW") params.modelPriority = ov::hint::Priority::LOW;
            else if (value == "MEDIUM") params.modelPriority = ov::hint::Priority::MEDIUM;
            else if (value == "HIGH") params.modelPriority = ov::hint::Priority::HIGH;
            else return "model_priority must be LOW, MEDIUM, or HIGH";
        }

        if (j.contains("performance_hint_num_requests"))
        {
            if (!j["performance_hint_num_requests"].is_number_unsigned() ||
                j["performance_hint_num_requests"].get<uint64_t>() > std::numeric_limits<uint32_t>::max())
                return "performance_hint_num_requests must be an unsigned 32-bit integer";
            params.performanceHintNumRequests = j["performance_hint_num_requests"].get<uint32_t>();
        }

        if (j.contains("num_streams"))
        {
            const auto& value = j["num_streams"];
            if (value.is_number_integer())
            {
                const int64_t count = value.get<int64_t>();
                if (count < -2 || count > std::numeric_limits<int32_t>::max())
                    return "num_streams must be AUTO, NUMA, or an integer >= 0";
                params.numStreams = ov::streams::Num{ static_cast<int32_t>(count) };
            }
            else if (value.is_string() && value.get<std::string>() == "AUTO") params.numStreams = ov::streams::AUTO;
            else if (value.is_string() && value.get<std::string>() == "NUMA") params.numStreams = ov::streams::NUMA;
            else return "num_streams must be AUTO, NUMA, or an integer >= 0";
        }

        auto integer_property = [&j](const char* key, std::optional<int32_t>& target)
        {
            if (!j.contains(key)) return true;
            if (!j[key].is_number_integer()) return false;
            const int64_t value = j[key].get<int64_t>();
            if (value < 1 || value > std::numeric_limits<int32_t>::max()) return false;
            target = static_cast<int32_t>(value);
            return true;
        };
        if (!integer_property("inference_num_threads", params.inferenceNumThreads))
            return "inference_num_threads must be a positive 32-bit integer";
        if (!integer_property("compilation_num_threads", params.compilationNumThreads))
            return "compilation_num_threads must be a positive 32-bit integer";

        if (j.contains("scheduling_core_type"))
        {
            if (!j["scheduling_core_type"].is_string()) return "scheduling_core_type must be a string";
            const auto value = j["scheduling_core_type"].get<std::string>();
            if (value == "ANY_CORE") params.schedulingCoreType = ov::hint::SchedulingCoreType::ANY_CORE;
            else if (value == "PCORE_ONLY") params.schedulingCoreType = ov::hint::SchedulingCoreType::PCORE_ONLY;
            else if (value == "ECORE_ONLY") params.schedulingCoreType = ov::hint::SchedulingCoreType::ECORE_ONLY;
            else return "scheduling_core_type must be ANY_CORE, PCORE_ONLY, or ECORE_ONLY";
        }

        for (const auto& [key, target] : std::vector<std::pair<const char*, std::optional<bool>*>>{
            {"enable_cpu_pinning", &params.enableCpuPinning},
            {"enable_hyper_threading", &params.enableHyperThreading},
            {"enable_profiling", &params.enableProfiling}})
        {
            if (j.contains(key))
            {
                if (!j[key].is_boolean()) return std::string(key) + " must be a boolean";
                *target = j[key].get<bool>();
            }
        }

        if (j.contains("compile_properties"))
        {
            if (!j["compile_properties"].is_object()) return "compile_properties must be a JSON object";
            for (auto it = j["compile_properties"].begin(); it != j["compile_properties"].end(); ++it)
            {
                const auto& value = it.value();
                if (value.is_boolean()) params.compileProperties[it.key()] = value.get<bool>();
                else if (value.is_number_unsigned()) params.compileProperties[it.key()] = value.get<uint64_t>();
                else if (value.is_number_integer()) params.compileProperties[it.key()] = value.get<int64_t>();
                else if (value.is_number_float()) params.compileProperties[it.key()] = value.get<double>();
                else if (value.is_string()) params.compileProperties[it.key()] = value.get<std::string>();
                else return "compile_properties values must be strings, numbers, or booleans";
            }
        }

        return {};
    }

    //! Inverse of parse_config: emits the same keys so the result can be fed
    //! back to rs_inference_runner_create. Unset optional settings are omitted.
    json config_to_json(const rsinfer::OpenVINORunnerParams& params)
    {
        json j = json::object();
        j["device"] = params.device;
        if (!params.modelWeightsPath.empty()) j["model_weights_path"] = params.modelWeightsPath;
        if (params.cacheDir) j["cache_dir"] = *params.cacheDir;
        j["num_infer_requests"] = params.numInferRequests;

        if (params.performanceMode)
        {
            switch (*params.performanceMode)
            {
            case ov::hint::PerformanceMode::LATENCY: j["performance_mode"] = "LATENCY"; break;
            case ov::hint::PerformanceMode::THROUGHPUT: j["performance_mode"] = "THROUGHPUT"; break;
            case ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT: j["performance_mode"] = "CUMULATIVE_THROUGHPUT"; break;
            default: break;
            }
        }
        if (params.inferencePrecision) j["inference_precision"] = params.inferencePrecision->get_type_name();
        if (params.executionMode)
            j["execution_mode"] = *params.executionMode == ov::hint::ExecutionMode::ACCURACY ? "ACCURACY" : "PERFORMANCE";
        if (params.modelPriority)
        {
            switch (*params.modelPriority)
            {
            case ov::hint::Priority::LOW: j["model_priority"] = "LOW"; break;
            case ov::hint::Priority::MEDIUM: j["model_priority"] = "MEDIUM"; break;
            case ov::hint::Priority::HIGH: j["model_priority"] = "HIGH"; break;
            default: break;
            }
        }
        if (params.performanceHintNumRequests) j["performance_hint_num_requests"] = *params.performanceHintNumRequests;

        if (params.numStreams)
        {
            const int32_t n = params.numStreams->num;
            if (n == ov::streams::AUTO.num) j["num_streams"] = "AUTO";
            else if (n == ov::streams::NUMA.num) j["num_streams"] = "NUMA";
            else j["num_streams"] = n;
        }
        if (params.inferenceNumThreads) j["inference_num_threads"] = *params.inferenceNumThreads;
        if (params.compilationNumThreads) j["compilation_num_threads"] = *params.compilationNumThreads;
        if (params.schedulingCoreType)
        {
            switch (*params.schedulingCoreType)
            {
            case ov::hint::SchedulingCoreType::ANY_CORE: j["scheduling_core_type"] = "ANY_CORE"; break;
            case ov::hint::SchedulingCoreType::PCORE_ONLY: j["scheduling_core_type"] = "PCORE_ONLY"; break;
            case ov::hint::SchedulingCoreType::ECORE_ONLY: j["scheduling_core_type"] = "ECORE_ONLY"; break;
            default: break;
            }
        }
        if (params.enableCpuPinning) j["enable_cpu_pinning"] = *params.enableCpuPinning;
        if (params.enableHyperThreading) j["enable_hyper_threading"] = *params.enableHyperThreading;
        if (params.enableProfiling) j["enable_profiling"] = *params.enableProfiling;

        if (!params.compileProperties.empty())
        {
            json props = json::object();
            for (const auto& [key, value] : params.compileProperties)
            {
                if (value.is<bool>()) props[key] = value.as<bool>();
                else if (value.is<uint64_t>()) props[key] = value.as<uint64_t>();
                else if (value.is<int64_t>()) props[key] = value.as<int64_t>();
                else if (value.is<double>()) props[key] = value.as<double>();
                else props[key] = value.as<std::string>(); // strings, and anything printable
            }
            j["compile_properties"] = std::move(props);
        }
        return j;
    }

    rsinfer::DataType to_internal_type(RSInferDataType type)
    {
        const rsinfer::DataType internal = rsinfer::from_public_data_type(type);
        if (internal == rsinfer::DataType::UNKNOWN)
            throw std::invalid_argument("OpenVINO Tensor adapter does not support this input data type");
        return internal;
    }

    //! Views of the caller's input tensors: no data is copied here. The runner
    //! reads them in place (process) or copies them once into its own buffers
    //! (process_async) before the ABI call returns.
    std::vector<rsinfer::OpenVINOInputView> make_views(const RSInferTensorArray* input_tensors)
    {
        if (!input_tensors || (input_tensors->length && !input_tensors->tensors))
            throw std::invalid_argument("Input tensor array is null or malformed");
        std::vector<rsinfer::OpenVINOInputView> inputs;
        inputs.reserve(input_tensors->length);
        for (size_t i = 0; i < input_tensors->length; ++i)
        {
            const RSInferTensor* source = input_tensors->tensors[i];
            if (!source || !source->shape || !source->data ||
                (source->name_length && !source->name) ||
                (source->shape->length && !source->shape->data) ||
                (source->data->length && !source->data->data))
                throw std::invalid_argument("Input tensor is missing name, data, or shape");

            rsinfer::OpenVINOInputView view;
            if (source->name_length)
                view.name = std::string_view(source->name, source->name_length);
            view.dtype = to_internal_type(source->dtype);
            view.shape.reserve(source->shape->length);
            for (size_t d = 0; d < source->shape->length; ++d)
            {
                const long long dim = source->shape->data[d];
                if (dim < 0 || static_cast<unsigned long long>(dim) > std::numeric_limits<size_t>::max())
                    throw std::invalid_argument("Tensor shape contains an invalid dimension");
                view.shape.push_back(static_cast<size_t>(dim));
            }
            view.data = std::span<const unsigned char>(source->data->data, source->data->length);
            inputs.push_back(std::move(view));
        }
        return inputs;
    }

    RSInferTensorArray* make_output(const std::vector<rsinfer::Tensor>& outputs, RSInferError* error)
    {
        RSInferTensorArray* array = rs_infer_tensorarray_create(outputs.size());
        if (!array)
        {
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate output tensor array");
            return nullptr;
        }
        for (size_t i = 0; i < outputs.size(); ++i)
        {
            const auto& output = outputs[i];
            RSInferUCharArray* data = rs_infer_uchararray_create(output.data.size());
            if (!data)
            {
                set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate output tensor data");
                rs_infer_tensorarray_destroy(array);
                return nullptr;
            }
            if (!output.data.empty())
                std::memcpy(data->data, output.data.data(), output.data.size());
            RSInferLLArray* shape = rs_infer_llarray_create(output.shape.size());
            if (!shape)
            {
                set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate output tensor shape");
                rs_infer_uchararray_destroy(data);
                rs_infer_tensorarray_destroy(array);
                return nullptr;
            }
            for (size_t dim = 0; dim < output.shape.size(); ++dim)
                shape->data[dim] = static_cast<long long>(output.shape[dim]);
            array->tensors[i] = rs_infer_tensor_create(output.name.c_str(), output.name.size(),
                rsinfer::to_public_data_type(output.dtype), data, shape);
            if (!array->tensors[i])
            {
                set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate output tensor");
                rs_infer_uchararray_destroy(data);
                rs_infer_llarray_destroy(shape);
                rs_infer_tensorarray_destroy(array);
                return nullptr;
            }
        }
        return array;
    }
}

RSInferenceRunner* rs_inference_runner_create(const char* model_path, const char* json_config_str, RSInferError* error)
{
    if (!model_path || !*model_path)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, "Model path cannot be null or empty");
        return nullptr;
    }

    try
    {
        rsinfer::OpenVINORunnerParams params;
        params.modelXmlPath = model_path;
        const std::string config_error = parse_config(json_config_str, params);
        if (!config_error.empty())
        {
            set_error(error, ERROR_CODE_INVALID_ARGUMENTS, config_error);
            return nullptr;
        }

        std::unique_ptr<rsinfer::OpenVINORunner> backend(new rsinfer::OpenVINORunner(params));
        std::unique_ptr<RSInferenceRunner> runner(static_cast<RSInferenceRunner*>(std::malloc(sizeof(RSInferenceRunner))));
        if (!runner)
        {
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate RSInferenceRunner");
            return nullptr;
        }
        runner->runner = backend.release();
        runner->backend = RSInferenceBackend_OpenVINO;
        return runner.release();
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_CREATION_FAILED, std::string("Failed to create OpenVINO runner: ") + e.what());
        return nullptr;
    }
}

void rs_inference_runner_destroy(RSInferenceRunner* runner)
{
    if (!runner) return;
    if (runner->backend == RSInferenceBackend_OpenVINO)
        delete static_cast<rsinfer::OpenVINORunner*>(runner->runner);
    std::free(runner);
}

int rs_inference_runner_infer(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error)
{
    try
    {
        auto* backend = get_runner(runner);
        const auto inputs = make_views(input_tensors);
        backend->process(std::span<const rsinfer::OpenVINOInputView>(inputs));
        return ERROR_CODE_SUCCESS;
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return ERROR_CODE_INVALID_ARGUMENTS;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("OpenVINO inference failed: ") + e.what());
        return ERROR_CODE_INFERENCE_FAILED;
    }
}

int rs_inference_runner_infer_async(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error)
{
    try
    {
        auto* backend = get_runner(runner);
        const auto inputs = make_views(input_tensors);
        backend->process_async(std::span<const rsinfer::OpenVINOInputView>(inputs));
        return ERROR_CODE_SUCCESS;
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return ERROR_CODE_INVALID_ARGUMENTS;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("OpenVINO asynchronous inference failed: ") + e.what());
        return ERROR_CODE_INFERENCE_FAILED;
    }
}

RSInferTensorArray* rs_inference_runner_get_results(RSInferenceRunner* runner, RSInferError* error)
{
    try
    {
        auto* backend = get_runner(runner);
        std::vector<rsinfer::Tensor> outputs;
        backend->get(outputs);
        return make_output(outputs, error);
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO results: ") + e.what());
        return nullptr;
    }
}

RSInferTensorArray* rs_inference_runner_get_input_tensors_info(RSInferenceRunner* runner, RSInferError* error)
{
    try
    {
        auto* backend = get_runner(runner);
        const auto infos = backend->get_input_tensors_info();
        if (!infos.has_value())
        {
            set_error(error, ERROR_CODE_INFERENCE_FAILED, "Failed to get OpenVINO input tensors info: " + infos.error());
            return nullptr;
        }
        // Info tensors carry name, dtype and shape only; their data arrays are empty.
        // Dynamic dimensions are reported as -1.
        return make_output(infos.value(), error);
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return nullptr;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO input tensors info: ") + e.what());
        return nullptr;
    }
}

RSInferTensorArray* rs_inference_runner_get_output_tensors_info(RSInferenceRunner* runner, RSInferError* error)
{
    try
    {
        auto* backend = get_runner(runner);
        const auto infos = backend->get_output_tensors_info();
        if (!infos.has_value())
        {
            set_error(error, ERROR_CODE_INFERENCE_FAILED, "Failed to get OpenVINO output tensors info: " + infos.error());
            return nullptr;
        }
        // Info tensors carry name, dtype and shape only; their data arrays are empty.
        // Dynamic dimensions are reported as -1.
        return make_output(infos.value(), error);
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return nullptr;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO output tensors info: ") + e.what());
        return nullptr;
    }
}

char* rs_inference_runner_get_json_config(RSInferenceRunner* runner, RSInferError* error)
{
    try
    {
        auto* backend = get_runner(runner);
        char* result = duplicate_string(config_to_json(backend->get_config()).dump());
        if (!result)
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate OpenVINO JSON config string");
        return result; // freed by the caller with rs_infer_string_free
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return nullptr;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO JSON config: ") + e.what());
        return nullptr;
    }
}
