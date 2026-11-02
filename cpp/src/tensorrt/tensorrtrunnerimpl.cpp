#include "rsinference.h"
#include "tensorrtrunner.h"

#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

enum ErrorCodes
{
    ERROR_CODE_SUCCESS = 0,
    ERROR_CODE_INVALID_ARGUMENTS = 1,
    ERROR_CODE_CREATION_FAILED = 2,
    ERROR_CODE_INFERENCE_FAILED = 3,
    ERROR_CODE_UNKNOWN_BACKEND = 4
};

using json = nlohmann::json;

// The public C enum and the C++ rsinfer::DataType enum use different value
// orders, so they must be translated rather than cast.
static rsinfer::DataType to_rsinfer_data_type(RSInferDataType dtype)
{
    switch (dtype)
    {
    case RSInferDataType_Float32: return rsinfer::DataType::FP32;
    case RSInferDataType_Half:    return rsinfer::DataType::FP16;
    case RSInferDataType_BF16:    return rsinfer::DataType::BF16;
    case RSInferDataType_Int4:    return rsinfer::DataType::INT4;
    case RSInferDataType_FP4:     return rsinfer::DataType::FP4;
    // The public enum has no signed 8-bit type; UInt8 is the only 8-bit slot.
    case RSInferDataType_UInt8:   return rsinfer::DataType::INT8;
    default:                      return rsinfer::DataType::UNKNOWN;
    }
}

static RSInferDataType to_public_data_type(rsinfer::DataType dtype)
{
    switch (dtype)
    {
    case rsinfer::DataType::FP32: return RSInferDataType_Float32;
    case rsinfer::DataType::FP16: return RSInferDataType_Half;
    case rsinfer::DataType::BF16: return RSInferDataType_BF16;
    case rsinfer::DataType::INT8: return RSInferDataType_UInt8;
    case rsinfer::DataType::INT4: return RSInferDataType_Int4;
    case rsinfer::DataType::FP4:  return RSInferDataType_FP4;
    default:                      return RSInferDataType_Unknown;
    }
}

static RSInferDataType tensorrt_data_type_to_public(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return RSInferDataType_Float32;
    case nvinfer1::DataType::kHALF:  return RSInferDataType_Half;
    case nvinfer1::DataType::kINT32: return RSInferDataType_Int32;
    case nvinfer1::DataType::kBOOL:  return RSInferDataType_Bool;
    case nvinfer1::DataType::kUINT8: return RSInferDataType_UInt8;
    case nvinfer1::DataType::kFP8:   return RSInferDataType_FP8;
    case nvinfer1::DataType::kE8M0:  return RSInferDataType_E8M0;
    case nvinfer1::DataType::kBF16:  return RSInferDataType_BF16;
    case nvinfer1::DataType::kINT64: return RSInferDataType_Int64;
    case nvinfer1::DataType::kINT4:  return RSInferDataType_Int4;
    case nvinfer1::DataType::kFP4:   return RSInferDataType_FP4;
    // No signed 8-bit slot in the public enum; UInt8 is size-compatible.
    case nvinfer1::DataType::kINT8:  return RSInferDataType_UInt8;
    default:                         return RSInferDataType_Unknown;
    }
}

static bool public_data_type_to_tensorrt(RSInferDataType dtype, nvinfer1::DataType& result)
{
    switch (dtype)
    {
    case RSInferDataType_Float32: result = nvinfer1::DataType::kFLOAT; return true;
    case RSInferDataType_Int32:   result = nvinfer1::DataType::kINT32; return true;
    case RSInferDataType_Int64:   result = nvinfer1::DataType::kINT64; return true;
    case RSInferDataType_UInt8:   result = nvinfer1::DataType::kUINT8; return true;
    case RSInferDataType_Bool:    result = nvinfer1::DataType::kBOOL; return true;
    case RSInferDataType_Half:    result = nvinfer1::DataType::kHALF; return true;
    case RSInferDataType_FP8:     result = nvinfer1::DataType::kFP8; return true;
    case RSInferDataType_E8M0:    result = nvinfer1::DataType::kE8M0; return true;
    case RSInferDataType_BF16:    result = nvinfer1::DataType::kBF16; return true;
    case RSInferDataType_Int4:    result = nvinfer1::DataType::kINT4; return true;
    case RSInferDataType_FP4:     result = nvinfer1::DataType::kFP4; return true;
    default:                      return false;
    }
}

struct ProfileEntry
{
    std::string name;
    nvinfer1::Dims mn, opt, mx;
};

static bool to_dims(const json& j, nvinfer1::Dims& d)
{
    if (!j.is_array() || j.empty() || j.size() > nvinfer1::Dims::MAX_DIMS) return false;
    d.nbDims = static_cast<int32_t>(j.size());
    for (int32_t i = 0; i < d.nbDims; ++i)
    {
        if (!j[i].is_number_integer()) return false;
        d.d[i] = j[i].get<int64_t>();
    }
    return true;
}

// Returns an empty string on success, otherwise a message describing the problem.
static std::string load_config(const std::string& json_config, rsinfer::RunnerParams& params, std::vector<ProfileEntry>& entries)
{
    json j = json::parse(json_config, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object())
        return "json_config_str is not a valid JSON object";

    if (j.contains("n_contexts"))
    {
        if (!j["n_contexts"].is_number_integer() || j["n_contexts"].get<int64_t>() < 1)
            return "n_contexts must be an integer >= 1";
        params.nContexts = j["n_contexts"].get<uint32_t>();
    }

    if (j.contains("timing_cache"))
    {
        if (!j["timing_cache"].is_string())
            return "timing_cache must be a string";
        params.timingCacheFile = j["timing_cache"].get<std::string>();
    }

    if (j.contains("dla_core"))
    {
        if (!j["dla_core"].is_number_integer())
            return "dla_core must be an integer";
        params.dlaCore = j["dla_core"].get<int32_t>();
    }

    if (j.contains("data_dirs"))
    {
        if (!j["data_dirs"].is_array())
            return "data_dirs must be an array of strings";
        for (const auto& d : j["data_dirs"])
        {
            if (!d.is_string())
                return "data_dirs must be an array of strings";
            params.dataDirs.push_back(d.get<std::string>());
        }
    }

    if (j.contains("profile"))
    {
        if (!j["profile"].is_object())
            return "profile must be an object";
        for (auto it = j["profile"].begin(); it != j["profile"].end(); ++it)
        {
            ProfileEntry e;
            e.name = it.key();
            const json& v = it.value();
            if (!v.is_object() || !v.contains("min") || !v.contains("opt") || !v.contains("max") ||
                !to_dims(v["min"], e.mn) || !to_dims(v["opt"], e.opt) || !to_dims(v["max"], e.mx))
                return "Bad profile entry for tensor: " + e.name;
            entries.push_back(std::move(e));
        }
    }

    params.dynamic = !entries.empty();
    return {};
}

RSInferenceRunner* rs_inference_runner_create(const char* model_path, const char* json_config_str, RSInferError* error)
{
    if (!model_path)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Model path cannot be null");
        }
        return nullptr;
    }

    // Declared outside the try so the catch block can free them.
    RSInferenceRunner* runner = nullptr;
    rsinfer::TensorRTRunner* tensorrtrunner = nullptr;

    try
    {
        rsinfer::RunnerParams params;
        params.onnxFileName = model_path;
        // `locateFile()` requires at least one directory entry. An empty entry
        // means "use onnxFileName exactly as given", matching the C++ sample.
        params.dataDirs = { "" };

        std::vector<ProfileEntry> entries;
        if (json_config_str && *json_config_str)
        {
            std::string msg = load_config(json_config_str, params, entries);
            if (!msg.empty())
            {
                if (error)
                {
                    error->code = ERROR_CODE_INVALID_ARGUMENTS;
                    error->message = strdup(msg.c_str());
                }
                return nullptr;
            }
        }

        runner = (RSInferenceRunner*)malloc(sizeof(RSInferenceRunner));
        if (!runner)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferenceRunner");
            }
            return nullptr;
        }

        tensorrtrunner = new rsinfer::TensorRTRunner(params);
        runner->runner = static_cast<void*>(tensorrtrunner);
        runner->backend = RSInferenceBackend_TensorRT;

        std::function<bool(nvinfer1::IOptimizationProfile*)> profile_cb;
        if (!entries.empty())
        {
            profile_cb = [&entries](nvinfer1::IOptimizationProfile* p)
            {
                for (const auto& e : entries)
                {
                    if (!p->setDimensions(e.name.c_str(), nvinfer1::OptProfileSelector::kMIN, e.mn) ||
                        !p->setDimensions(e.name.c_str(), nvinfer1::OptProfileSelector::kOPT, e.opt) ||
                        !p->setDimensions(e.name.c_str(), nvinfer1::OptProfileSelector::kMAX, e.mx))
                        return false;
                }
                return true;
            };
        }

        if (!tensorrtrunner->build(profile_cb))
        {
            delete tensorrtrunner;
            free(runner);
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to build the TensorRT engine");
            }
            return nullptr;
        }

        return runner;
    }
    catch (const std::exception& e)
    {
        delete tensorrtrunner; // deleting nullptr is a no-op
        free(runner);
        if (error)
        {
            error->code = ERROR_CODE_CREATION_FAILED;
            error->message = strdup(("Failed to create RSInferenceRunner: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

void rs_inference_runner_destroy(RSInferenceRunner* runner)
{
    if (runner)
    {
        if (runner->runner)
        {
            auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
            delete tensorrtrunner;
        }
        free(runner);
    }
}

int rs_inference_runner_infer(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error)
{
    if (!runner || !input_tensors)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner or input tensors cannot be null");
        }
        return ERROR_CODE_INVALID_ARGUMENTS;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<rsinfer::Tensor> inputs;

        for (size_t i = 0; i < input_tensors->length; ++i)
        {
            RSInferTensor* tensor = input_tensors->tensors[i];
            // The Rust side is allowed to hand us a tensor whose payload is
            // missing; report it instead of dereferencing null below.
            if (!tensor || !tensor->shape || !tensor->data ||
                (!tensor->shape->data && tensor->shape->length > 0) ||
                (!tensor->data->data && tensor->data->length > 0))
            {
                if (error)
                {
                    error->code = ERROR_CODE_INVALID_ARGUMENTS;
                    error->message = strdup("Input tensor is missing data or shape");
                }
                return ERROR_CODE_INVALID_ARGUMENTS;
            }
            rsinfer::Tensor t;
            t.name = tensor->name;
            t.shape.assign(tensor->shape->data, tensor->shape->data + tensor->shape->length);
            t.data.assign(tensor->data->data, tensor->data->data + tensor->data->length);
            t.dtype = to_rsinfer_data_type(tensor->dtype);
            inputs.push_back(std::move(t));
        }

        tensorrtrunner->process(inputs);

        return ERROR_CODE_SUCCESS;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Inference failed: " + std::string(e.what())).c_str());
        }
        return ERROR_CODE_INFERENCE_FAILED;
    }
}

int rs_inference_runner_infer_async(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error)
{
    if (!runner || !input_tensors)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner or input tensors cannot be null");
        }
        return ERROR_CODE_INVALID_ARGUMENTS;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<rsinfer::Tensor> inputs;

        for (size_t i = 0; i < input_tensors->length; ++i)
        {
            RSInferTensor* tensor = input_tensors->tensors[i];
            // The Rust side is allowed to hand us a tensor whose payload is
            // missing; report it instead of dereferencing null below.
            if (!tensor || !tensor->shape || !tensor->data ||
                (!tensor->shape->data && tensor->shape->length > 0) ||
                (!tensor->data->data && tensor->data->length > 0))
            {
                if (error)
                {
                    error->code = ERROR_CODE_INVALID_ARGUMENTS;
                    error->message = strdup("Input tensor is missing data or shape");
                }
                return ERROR_CODE_INVALID_ARGUMENTS;
            }
            rsinfer::Tensor t;
            t.name = tensor->name;
            t.shape.assign(tensor->shape->data, tensor->shape->data + tensor->shape->length);
            t.data.assign(tensor->data->data, tensor->data->data + tensor->data->length);
            t.dtype = to_rsinfer_data_type(tensor->dtype);
            inputs.push_back(std::move(t));
        }

        tensorrtrunner->process_async(inputs);

        return ERROR_CODE_SUCCESS;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Asynchronous inference failed: " + std::string(e.what())).c_str());
        }
        return ERROR_CODE_INFERENCE_FAILED;
    }
}

RSInferTensorArray* rs_inference_runner_get_results(RSInferenceRunner* runner, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<rsinfer::Tensor> outputs;
        tensorrtrunner->get(outputs);

        RSInferTensorArray* output_array = rs_infer_tensorarray_create(outputs.size());
        if (!output_array)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferTensorArray");
            }
            return nullptr;
        }

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            const auto& tensor = outputs[i];
            RSInferUCharArray* data_array = rs_infer_uchararray_create(tensor.data.size());
            if (!data_array)
            {
                if (error)
                {
                    error->code = ERROR_CODE_CREATION_FAILED;
                    error->message = strdup("Failed to allocate memory for RSInferUCharArray");
                }
                rs_infer_tensorarray_destroy(output_array);
                return nullptr;
            }
            memcpy(data_array->data, tensor.data.data(), tensor.data.size());
            RSInferLLArray* shape_array = rs_infer_llarray_create(tensor.shape.size());
            if (!shape_array)
            {
                if (error)
                {
                    error->code = ERROR_CODE_CREATION_FAILED;
                    error->message = strdup("Failed to allocate memory for RSInferLLArray");
                }
                rs_infer_uchararray_destroy(data_array);
                rs_infer_tensorarray_destroy(output_array);
                return nullptr;
            }
            for (size_t d = 0; d < tensor.shape.size(); ++d)
                shape_array->data[d] = static_cast<long long>(tensor.shape[d]);
            output_array->tensors[i] = rs_infer_tensor_create(tensor.name.c_str(), tensor.name.size(), to_public_data_type(tensor.dtype), data_array, shape_array);
            if (!output_array->tensors[i])
            {
                // `rs_infer_tensor_create` did not take ownership on failure.
                if (error)
                {
                    error->code = ERROR_CODE_CREATION_FAILED;
                    error->message = strdup("Failed to allocate memory for RSInferTensor");
                }
                rs_infer_uchararray_destroy(data_array);
                rs_infer_llarray_destroy(shape_array);
                rs_infer_tensorarray_destroy(output_array);
                return nullptr;
            }
        }

        return output_array;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get inference results: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

RSInferIntArray* rs_inference_runner_get_input_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<int> shape = tensorrtrunner->get_input_shape(index);

        RSInferIntArray* shape_array = rs_infer_intarray_create(shape.size());
        if (!shape_array)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferIntArray");
            }
            return nullptr;
        }
        memcpy(shape_array->data, shape.data(), shape.size() * sizeof(int));

        return shape_array;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get input shape: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

RSInferIntArray* rs_inference_runner_get_output_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<int> shape = tensorrtrunner->get_output_shape(index);

        RSInferIntArray* shape_array = rs_infer_intarray_create(shape.size());
        if (!shape_array)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferIntArray");
            }
            return nullptr;
        }
        memcpy(shape_array->data, shape.data(), shape.size() * sizeof(int));

        return shape_array;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get output shape: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

RSInferDataType rs_inference_runner_get_tensor_data_type(RSInferenceRunner* runner, const char* tensor_name, RSInferError* error)
{
    if (!runner || !tensor_name)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner or tensor name cannot be null");
        }
        return RSInferDataType_Unknown;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        nvinfer1::DataType data_type = tensorrtrunner->get_tensor_data_type(tensor_name);
        return tensorrt_data_type_to_public(data_type);
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get tensor data type: " + std::string(e.what())).c_str());
        }
        return RSInferDataType_Unknown;
    }
}

size_t rs_inference_runner_get_tensor_data_size(RSInferenceRunner* runner, RSInferDataType data_type, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return 0;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        nvinfer1::DataType trt_data_type;
        if (!public_data_type_to_tensorrt(data_type, trt_data_type))
        {
            if (error)
            {
                error->code = ERROR_CODE_INVALID_ARGUMENTS;
                error->message = strdup("Unsupported tensor data type");
            }
            return 0;
        }
        return tensorrtrunner->get_tensor_data_size(trt_data_type);
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get tensor data size: " + std::string(e.what())).c_str());
        }
        return 0;
    }
}

const char* rs_inference_runner_get_input_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::string name = tensorrtrunner->get_input_tensor_name(index);
        return strdup(name.c_str());
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get input tensor name: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

const char* rs_inference_runner_get_output_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::string name = tensorrtrunner->get_output_tensor_name(index);
        return strdup(name.c_str());
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get output tensor name: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}
