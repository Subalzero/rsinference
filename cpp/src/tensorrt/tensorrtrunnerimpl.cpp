#include "rsinference.h"
#include "tensorrtrunner.h"

enum ErrorCodes
{
    ERROR_CODE_SUCCESS = 0,
    ERROR_CODE_INVALID_ARGUMENTS = 1,
    ERROR_CODE_CREATION_FAILED = 2,
    ERROR_CODE_INFERENCE_FAILED = 3,
    ERROR_CODE_UNKNOWN_BACKEND = 4
};

RSInferenceRunner* rs_inference_runner_create(const char* model_path, const char* config_path, RSInferError* error)
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

    try
    {
        rsinfer::RunnerParams params;
        params.onnxFileName = model_path;
        if (config_path)
        {
            std::ifstream config_file(config_path);
            if (!config_file.is_open())
            {
                error->code = ERROR_CODE_INVALID_ARGUMENTS;
                error->message = strdup(("Failed to open config file: " + std::string(config_path)).c_str());
                return nullptr;
            }
            std::string line;
            while (std::getline(config_file, line))
            {
                // Parse the config file and set parameters accordingly
                // For example, you can parse lines like "batchSize=4" and set params.batchSize = 4
                // Implement your own parsing logic here based on your config file format
            }
        }

        RSInferenceRunner* runner = (RSInferenceRunner*)malloc(sizeof(RSInferenceRunner));
        if (!runner)
        {
            error->code = ERROR_CODE_CREATION_FAILED;
            error->message = strdup("Failed to allocate memory for RSInferenceRunner");
            return nullptr;
        }

        auto tensorrtrunner = new rsinfer::TensorRTRunner(params);
        runner->runner = static_cast<void*>(tensorrtrunner);
        runner->backend = RSInferenceBackend_TensorRT;

        if (!tensorrtrunner->build())
        {
            delete tensorrtrunner;
            free(runner);
            error->code = ERROR_CODE_CREATION_FAILED;
            error->message = strdup("Failed to build the TensorRT engine");
            return nullptr;
        }

        return runner;
    }
    catch (const std::exception& e)
    {
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
            t.dtype = static_cast<rsinfer::DataType>(tensor->dtype);
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
            t.dtype = static_cast<rsinfer::DataType>(tensor->dtype);
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
            memcpy(shape_array->data, tensor.shape.data(), tensor.shape.size());
            output_array->tensors[i] = rs_infer_tensor_create(tensor.name.c_str(), tensor.name.size(), static_cast<RSInferDataType>(tensor.dtype), data_array, shape_array);
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
        return static_cast<RSInferDataType>(data_type);
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
        nvinfer1::DataType trt_data_type = static_cast<nvinfer1::DataType>(data_type);
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