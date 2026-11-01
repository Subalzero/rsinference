#ifndef RS_INFERENCE_H
#define RS_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#ifdef _WIN32
    #define BACKEND_EXPORT __declspec(dllexport)
#else
    #define BACKEND_EXPORT __attribute__((visibility("default")))
#endif

typedef struct _llarray {
    long long* data;
    size_t length;
} RSInferLLArray;

// RSInferLLArray helpers
BACKEND_EXPORT RSInferLLArray* rs_infer_llarray_create(size_t length);
BACKEND_EXPORT void rs_infer_llarray_destroy(RSInferLLArray* array);

typedef struct _fparray {
    float* data;
    size_t length;
} RSInferFPArray;

// RSInferFPArray helpers
BACKEND_EXPORT RSInferFPArray* rs_infer_fparray_create(size_t length);
BACKEND_EXPORT void rs_infer_fparray_destroy(RSInferFPArray* array);

typedef struct _uchararray {
    unsigned char* data;
    size_t length;
} RSInferUCharArray;

// RSInferUCharArray helpers
BACKEND_EXPORT RSInferUCharArray* rs_infer_uchararray_create(size_t length);
BACKEND_EXPORT void rs_infer_uchararray_destroy(RSInferUCharArray* array);

typedef struct _intarray {
    int* data;
    size_t length;
} RSInferIntArray;

// RSInferIntArray helpers
BACKEND_EXPORT RSInferIntArray* rs_infer_intarray_create(size_t length);
BACKEND_EXPORT void rs_infer_intarray_destroy(RSInferIntArray* array);

typedef struct _rsinfer_error {
    int code = 0;
    char* message = NULL;
} RSInferError;

// RSInferError helpers
BACKEND_EXPORT void rs_infer_error_cleanmsg(RSInferError* error);

enum RSInferDataType {
    RSInferDataType_Float32 = 0,
    RSInferDataType_Int32 = 1,
    RSInferDataType_Int64 = 2,
    RSInferDataType_UInt8 = 3,
    RSInferDataType_Bool = 4,
    RSInferDataType_Half = 5,
    RSInferDataType_FP8 = 6,
    RSInferDataType_E8M0 = 7,
    RSInferDataType_BF16 = 8,
    RSInferDataType_Int4 = 9,
    RSInferDataType_FP4 = 10,
    RSInferDataType_Unknown = 11
};

typedef struct _tensor {
    char* name;
    size_t name_length;
    RSInferDataType dtype;
    RSInferUCharArray* data;
    RSInferLLArray* shape;
} RSInferTensor;

BACKEND_EXPORT RSInferTensor* rs_infer_tensor_create(const char* name, size_t name_length, RSInferDataType dtype, RSInferUCharArray* data, RSInferLLArray* shape);
BACKEND_EXPORT void rs_infer_tensor_destroy(RSInferTensor* tensor);

typedef struct _tensorarray {
    RSInferTensor** tensors;
    size_t length;
} RSInferTensorArray;

// RSInferTensorArray helpers
BACKEND_EXPORT RSInferTensorArray* rs_infer_tensorarray_create(size_t length);
BACKEND_EXPORT void rs_infer_tensorarray_destroy(RSInferTensorArray* array);

// ----------------------------------------------------------------------------
// RSInferenceRunner
// ----------------------------------------------------------------------------
enum RSInferenceBackend {
    RSInferenceBackend_TensorRT = 0,
    RSInferenceBackend_ONNXRuntime = 1,
    RSInferenceBackend_OpenVINO = 2,
    RSInferenceBackend_OpenCV = 3
};

/**
 * RSInferenceRunner is a typedef for a structure that encapsulates an inference runner instance, 
 * containing a pointer to the actual runner (such as TensorRTRunner) and an associated inference backend 
 * of type RSInferenceBackend. 
 * This structure is likely used to manage and execute inference tasks in a machine learning context.
 */
typedef struct _inference_runner {
    void* runner; // Pointer to the actual inference runner instance (e.g., TensorRTRunner)
    RSInferenceBackend backend;
} RSInferenceRunner;

/**
 * Creates an RSInferenceRunner instance based on the provided model and configuration paths.
 * @param model_path: Path to the model file (e.g., ONNX, TensorRT engine).
 * @param config_path: Path to the configuration file (if applicable).
 * @param error: Pointer to an RSInferError structure to capture any errors during creation.
 * @return Pointer to the created RSInferenceRunner instance, or nullptr if creation fails.
 * 
 * @attention Implementation Note: This function is called by the Rust engine to create an inference runner.
 * In creating your own inference runner, you should implement this function to initialize the runner 
 * based on the specified model and configuration.
 * DO NOT CONSUME or FREE the model_path, config_path, or error parameter; they are managed by the Rust side. 
 * You should only populate the error parameter with error information if needed.
 * THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 */
BACKEND_EXPORT RSInferenceRunner* rs_inference_runner_create(const char* model_path, const char* config_path, RSInferError* error);
/**
 * Destroys an RSInferenceRunner instance, releasing any associated resources.
 * @param runner: Pointer to the RSInferenceRunner instance to be destroyed.
 * 
 * @attention Implementation Note: This function is called by the Rust engine to destroy an inference runner.
 * In creating your own inference runner, you should implement this function to clean up and free 
 * any resources associated with the runner.
 * THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 */
BACKEND_EXPORT void rs_inference_runner_destroy(RSInferenceRunner* runner);
/**
 * Performs inference using the provided RSInferenceRunner instance and input tensors.
 * @param runner: Pointer to the RSInferenceRunner instance to be used for inference.
 * @param input_tensors: Pointer to an RSInferTensorArray containing the input tensors for inference.
 * @param error: Pointer to an RSInferError structure to capture any errors during inference.
 * @return An integer status code indicating the success or failure of the inference operation.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * 
 * @attention Implementation Note: This function is called by the Rust engine to perform inference.
 * In creating your own inference runner, you should implement this function to execute the inference process
 * using the provided input tensors and return the appropriate status code.
 * input_tensors and error are passed in from the Rust side, and you should NOT CONSUME or FREE them.
 * The Rust side will handle their memory management. You should only read from these structures and not attempt to free or modify them.
 * If you need to return results, use the rs_inference_runner_get_results function to retrieve the inference results.
 * If you need to return an error message, populate the RSInferError structure pointed to by the error parameter, 
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT int rs_inference_runner_infer(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error);
/**
 * Performs asynchronous inference using the provided RSInferenceRunner instance and input tensors.
 * @param runner: Pointer to the RSInferenceRunner instance to be used for inference.
 * @param input_tensors: Pointer to an RSInferTensorArray containing the input tensors for
 * @param error: Pointer to an RSInferError structure to capture any errors during inference.
 * @return An integer status code indicating the success or failure of the asynchronous inference operation.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to perform asynchronous inference.
 * In creating your own inference runner, you should implement this function to execute the inference process
 * using the provided input tensors and return the appropriate status code.
 * input_tensors and error are passed in from the Rust side, and you should NOT CONSUME or FREE them.
 * The Rust side will handle their memory management. You should only read from these structures and not attempt to free or modify them.
 * If you need to return results, use the rs_inference_runner_get_results function to retrieve the inference results.
 * If you need to return an error message, populate the RSInferError structure pointed to by the error parameter, 
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT int rs_inference_runner_infer_async(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error);
/**
 * Retrieves the inference results from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve results.
 * @return Pointer to an RSInferTensorArray containing the inference results, or nullptr if retrieval fails.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 * 
 * Implementation Note: This function is called by the Rust engine to retrieve inference results.
 * In creating your own inference runner, you should implement this function to return the results of the inference process.
 * The returned RSInferTensorArray should be allocated and populated with the inference results.
 * The Rust side will handle the memory management of the returned RSInferTensorArray, so you should not free it.
 * If you encounter an error while retrieving results, you can populate the RSInferError structure pointed to by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT RSInferTensorArray* rs_inference_runner_get_results(RSInferenceRunner* runner, RSInferError* error);
/**
 * Retrieves the input shape of a specific tensor from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve the input shape.
 * @param index: The index of the input tensor for which to retrieve the shape.
 * @param error: Pointer to an RSInferError structure to capture any errors during shape retrieval
 * @return Pointer to an RSInferIntArray containing the input shape, or nullptr if retrieval fails.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 * 
 * @attention Implementation Note: This function is called by the Rust engine to retrieve the input shape of a specific tensor.
 * In creating your own inference runner, you should implement this function to return the shape of the specified input tensor.
 * The returned RSInferIntArray should be allocated and populated with the input shape.
 * The Rust side will handle the memory management of the returned RSInferIntArray, so you should not free it.
 * If you encounter an error while retrieving the input shape, you can populate the RSInferError structure pointed to by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT RSInferIntArray* rs_inference_runner_get_input_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error);
/**
 * Retrieves the output shape of a specific tensor from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve the output shape.
 * @param index: The index of the output tensor for which to retrieve the shape.
 * @param error: Pointer to an RSInferError structure to capture any errors during shape retrieval.
 * @return Pointer to an RSInferIntArray containing the output shape, or nullptr if retrieval fails.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to retrieve the output shape of a specific tensor.
 * In creating your own inference runner, you should implement this function to return the shape of the specified output tensor.
 * The returned RSInferIntArray should be allocated and populated with the output shape.
 * The Rust side will handle the memory management of the returned RSInferIntArray, so you should not free it.
 * If you encounter an error while retrieving the output shape, you can populate the RSInferError structure pointed to by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT RSInferIntArray* rs_inference_runner_get_output_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error);
/**
 * Retrieves the data type of a specific tensor from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve the tensor data type.
 * @param tensor_name: The name of the tensor for which to retrieve the data type.
 * @param error: Pointer to an RSInferError structure to capture any errors during data type retrieval.
 * @return The RSInferDataType of the specified tensor, or RSInferDataType_Unknown if retrieval fails.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to retrieve the data type of a specific tensor.
 * In creating your own inference runner, you should implement this function to return the data type of the specified tensor.
 * If you encounter an error while retrieving the tensor data type, you can populate the RSInferError structure pointed to by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT RSInferDataType rs_inference_runner_get_tensor_data_type(RSInferenceRunner* runner, const char* tensor_name, RSInferError* error);
/**
 * Retrieves the size in bytes of a specific data type from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve the data size.
 * @param data_type: The RSInferDataType for which to retrieve the size in bytes.
 * @param error: Pointer to an RSInferError structure to capture any errors during data size retrieval.
 * @return The size in bytes of the specified data type, or 0 if retrieval fails.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to retrieve the size in bytes of a specific data type.
 * In creating your own inference runner, you should implement this function to return the size in bytes of the specified data type.
 * If you encounter an error while retrieving the data size, you can populate the RSInferError structure pointed to by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT size_t rs_inference_runner_get_tensor_data_size(RSInferenceRunner* runner, RSInferDataType data_type, RSInferError* error);
/**
 * Retrieves the name of a specific input tensor from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve the input tensor name.
 * @param index: The index of the input tensor for which to retrieve the name.
 * @return Pointer to a null-terminated string containing the input tensor name, or nullptr if retrieval fails.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to retrieve the name of a specific input tensor.
 * In creating your own inference runner, you should implement this function to return the name of the specified input tensor.
 * The returned string should be a null-terminated C-style string. The Rust side will handle the memory management of the 
 * returned string, so you should not free it.
 * If you encounter an error while retrieving the input tensor name, you can populate the RSInferError structure pointed 
 * to by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT const char* rs_inference_runner_get_input_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error);
/**
 * Retrieves the name of a specific output tensor from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve the output tensor name.
 * @param index: The index of the output tensor for which to retrieve the name.
 * @param error: Pointer to an RSInferError structure to capture any errors during name retrieval.
 * @return Pointer to a null-terminated string containing the output tensor name, or nullptr if retrieval fails.
 * 
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to retrieve the name of a specific output tensor.
 * In creating your own inference runner, you should implement this function to return the name of the specified output tensor.
 * The returned string should be a null-terminated C-style string. The Rust side will handle the memory management of the
 * returned string, so you should not free it.
 * If you encounter an error while retrieving the output tensor name, you can populate the RSInferError structure pointed to 
 * by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT const char* rs_inference_runner_get_output_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error);




#ifdef __cplusplus
}
#endif

#endif // RS_INFERENCE_H