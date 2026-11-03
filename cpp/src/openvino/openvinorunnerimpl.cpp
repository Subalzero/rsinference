#include "rsinference.h"
#include "openvinorunner.h"

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

    size_t data_type_size(RSInferDataType type)
    {
        switch (type)
        {
        case RSInferDataType_Float32:
        case RSInferDataType_Int32: return 4;
        case RSInferDataType_Int64: return 8;
        case RSInferDataType_UInt8:
        case RSInferDataType_Bool: return 1;
        case RSInferDataType_Half:
        case RSInferDataType_BF16: return 2;
        // The API reports packed 4-bit storage as two elements per byte,
        // consistent with the TensorRT adapter.
        case RSInferDataType_Int4:
        case RSInferDataType_FP4: return 2;
        case RSInferDataType_FP8:
        case RSInferDataType_E8M0: return 1;
        default: return 0;
        }
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

    rsinfer::DataType to_internal_type(RSInferDataType type)
    {
        switch (type)
        {
        case RSInferDataType_Float32: return rsinfer::DataType::FP32;
        case RSInferDataType_Half: return rsinfer::DataType::FP16;
        case RSInferDataType_UInt8: return rsinfer::DataType::INT8;
        case RSInferDataType_BF16: return rsinfer::DataType::BF16;
        case RSInferDataType_Int4: return rsinfer::DataType::INT4;
        case RSInferDataType_FP4: return rsinfer::DataType::FP4;
        default: throw std::invalid_argument("OpenVINO Tensor adapter does not support this input data type");
        }
    }

    RSInferDataType to_public_type(rsinfer::DataType type)
    {
        switch (type)
        {
        case rsinfer::DataType::FP32: return RSInferDataType_Float32;
        case rsinfer::DataType::FP16: return RSInferDataType_Half;
        case rsinfer::DataType::INT8: return RSInferDataType_UInt8;
        case rsinfer::DataType::BF16: return RSInferDataType_BF16;
        case rsinfer::DataType::INT4: return RSInferDataType_Int4;
        case rsinfer::DataType::FP4: return RSInferDataType_FP4;
        default: return RSInferDataType_Unknown;
        }
    }

    std::vector<rsinfer::Tensor> make_inputs(const RSInferTensorArray* input_tensors)
    {
        if (!input_tensors || (input_tensors->length && !input_tensors->tensors))
            throw std::invalid_argument("Input tensor array is null or malformed");
        std::vector<rsinfer::Tensor> inputs;
        inputs.reserve(input_tensors->length);
        for (size_t i = 0; i < input_tensors->length; ++i)
        {
            const RSInferTensor* source = input_tensors->tensors[i];
            if (!source || !source->shape || !source->data ||
                (source->name_length && !source->name) ||
                (source->shape->length && !source->shape->data) ||
                (source->data->length && !source->data->data))
                throw std::invalid_argument("Input tensor is missing name, data, or shape");

            rsinfer::Tensor tensor;
            if (source->name_length)
                tensor.name.assign(source->name, source->name_length);
            tensor.dtype = to_internal_type(source->dtype);
            if (source->shape->length)
                tensor.shape.assign(source->shape->data, source->shape->data + source->shape->length);
            if (source->data->length)
                tensor.data.assign(source->data->data, source->data->data + source->data->length);
            inputs.push_back(std::move(tensor));
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
                to_public_type(output.dtype), data, shape);
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
        backend->process(make_inputs(input_tensors));
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
        backend->process_async(make_inputs(input_tensors));
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

RSInferIntArray* rs_inference_runner_get_input_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    try
    {
        const auto shape = get_runner(runner)->get_input_shape(index);
        RSInferIntArray* result = rs_infer_intarray_create(shape.size());
        if (!result)
        {
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate OpenVINO input shape");
            return nullptr;
        }
        for (size_t i = 0; i < shape.size(); ++i)
        {
            result->data[i] = shape[i];
        }
        return result;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO input shape: ") + e.what());
        return nullptr;
    }
}

RSInferIntArray* rs_inference_runner_get_output_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    try
    {
        const auto shape = get_runner(runner)->get_output_shape(index);
        RSInferIntArray* result = rs_infer_intarray_create(shape.size());
        if (!result)
        {
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate OpenVINO output shape");
            return nullptr;
        }
        for (size_t i = 0; i < shape.size(); ++i)
        {
            result->data[i] = shape[i];
        }
        return result;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO output shape: ") + e.what());
        return nullptr;
    }
}

RSInferDataType rs_inference_runner_get_tensor_data_type(RSInferenceRunner* runner, const char* tensor_name, RSInferError* error)
{
    if (!tensor_name)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, "Tensor name cannot be null");
        return RSInferDataType_Unknown;
    }
    try
    {
        const auto type = get_runner(runner)->get_tensor_element_type(tensor_name);
        if (type == ov::element::f32) return RSInferDataType_Float32;
        if (type == ov::element::f16) return RSInferDataType_Half;
        if (type == ov::element::bf16) return RSInferDataType_BF16;
        if (type == ov::element::i8) return RSInferDataType_UInt8;
        if (type == ov::element::i32) return RSInferDataType_Int32;
        if (type == ov::element::i64) return RSInferDataType_Int64;
        if (type == ov::element::u8) return RSInferDataType_UInt8;
        if (type == ov::element::boolean) return RSInferDataType_Bool;
        if (type == ov::element::i4) return RSInferDataType_Int4;
        if (type == ov::element::f4e2m1) return RSInferDataType_FP4;
        if (type == ov::element::f8e4m3 || type == ov::element::f8e5m2) return RSInferDataType_FP8;
        if (type == ov::element::f8e8m0) return RSInferDataType_E8M0;
        return RSInferDataType_Unknown;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO tensor data type: ") + e.what());
        return RSInferDataType_Unknown;
    }
}

size_t rs_inference_runner_get_tensor_data_size(RSInferenceRunner* runner, RSInferDataType data_type, RSInferError* error)
{
    try
    {
        get_runner(runner);
        const size_t size = data_type_size(data_type);
        if (!size) set_error(error, ERROR_CODE_INVALID_ARGUMENTS, "Unsupported tensor data type");
        return size;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return 0;
    }
}

const char* rs_inference_runner_get_input_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    try
    {
        return duplicate_string(get_runner(runner)->get_input_tensor_name(index));
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO input tensor name: ") + e.what());
        return nullptr;
    }
}

const char* rs_inference_runner_get_output_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    try
    {
        return duplicate_string(get_runner(runner)->get_output_tensor_name(index));
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get OpenVINO output tensor name: ") + e.what());
        return nullptr;
    }
}
