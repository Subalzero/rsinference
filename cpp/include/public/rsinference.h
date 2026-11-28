#ifndef RS_INFERENCE_H
#define RS_INFERENCE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

#ifdef _WIN32
    #if defined(INFERTENSORRT_STATIC) || defined(INFEROPENVINO_STATIC)
        #define BACKEND_EXPORT
    #elif defined(INFERTENSORRT_BUILD) || defined(INFEROPENVINO_BUILD)
        #define BACKEND_EXPORT __declspec(dllexport)
    #else
        #define BACKEND_EXPORT __declspec(dllimport)
    #endif
#else
    #define BACKEND_EXPORT __attribute__((visibility("default")))
#endif

// Increment when a change alters public struct layout or function contracts.
#define RSINFERENCE_ABI_VERSION 3u
BACKEND_EXPORT unsigned int rs_inference_abi_version(void);

// Array helpers below: a zero-length array is valid and has data == NULL (and
// RSInferTensorArray::tensors == NULL), so never treat a NULL data pointer as an
// error when length is 0. A NULL return from a *_create helper means allocation failed.

typedef struct RSInferLLArray {
    long long* data;
    size_t length;
} RSInferLLArray;

// RSInferLLArray helpers
BACKEND_EXPORT RSInferLLArray* rs_infer_llarray_create(size_t length);
BACKEND_EXPORT void rs_infer_llarray_destroy(RSInferLLArray* array);

typedef struct RSInferFPArray {
    float* data;
    size_t length;
} RSInferFPArray;

// RSInferFPArray helpers
BACKEND_EXPORT RSInferFPArray* rs_infer_fparray_create(size_t length);
BACKEND_EXPORT void rs_infer_fparray_destroy(RSInferFPArray* array);

typedef struct RSInferUCharArray {
    unsigned char* data;
    size_t length;
} RSInferUCharArray;

// RSInferUCharArray helpers
BACKEND_EXPORT RSInferUCharArray* rs_infer_uchararray_create(size_t length);
BACKEND_EXPORT void rs_infer_uchararray_destroy(RSInferUCharArray* array);

typedef struct RSInferIntArray {
    int* data;
    size_t length;
} RSInferIntArray;

// RSInferIntArray helpers
BACKEND_EXPORT RSInferIntArray* rs_infer_intarray_create(size_t length);
BACKEND_EXPORT void rs_infer_intarray_destroy(RSInferIntArray* array);

typedef struct RSInferError {
    int code;
    char* message;
} RSInferError;

// Initialize with {0} before passing to an API call. Release any error message
// through this helper, then reset the fields before reusing the structure.
BACKEND_EXPORT void rs_infer_error_cleanmsg(RSInferError* error);

// Frees a string returned by the bridge (for example the JSON from
// rs_inference_runner_get_json_config). Strings must be released with
// this function, not with the host's free(), because the bridge may use a
// different C runtime.
BACKEND_EXPORT void rs_infer_string_free(const char* string);

typedef enum RSInferDataType {
    RSInferDataType_Float32 = 0,
    RSInferDataType_Int32 = 1,
    RSInferDataType_Int64 = 2,
    RSInferDataType_UInt8 = 3,
    RSInferDataType_Bool = 4,
    RSInferDataType_Half = 5,
    // 8-bit float, E4M3 layout (TensorRT kFP8, OpenVINO f8e4m3).
    RSInferDataType_FP8 = 6,
    RSInferDataType_E8M0 = 7,
    RSInferDataType_BF16 = 8,
    // Packed 4-bit types: two elements share one byte, so n elements take
    // (n + 1) / 2 bytes.
    RSInferDataType_Int4 = 9,
    RSInferDataType_FP4 = 10,
    RSInferDataType_Int8 = 11,
    // 8-bit float, E5M2 layout (OpenVINO f8e5m2).
    RSInferDataType_FP8E5M2 = 12,
    RSInferDataType_Unknown = 13
} RSInferDataType;

typedef struct RSInferTensor {
    char* name;
    size_t name_length;
    RSInferDataType dtype;
    RSInferUCharArray* data;
    RSInferLLArray* shape;
} RSInferTensor;

BACKEND_EXPORT RSInferTensor* rs_infer_tensor_create(const char* name, size_t name_length, RSInferDataType dtype, RSInferUCharArray* data, RSInferLLArray* shape);
BACKEND_EXPORT void rs_infer_tensor_destroy(RSInferTensor* tensor);

typedef struct RSInferTensorArray {
    RSInferTensor** tensors;
    size_t length;
} RSInferTensorArray;

// RSInferTensorArray helpers
BACKEND_EXPORT RSInferTensorArray* rs_infer_tensorarray_create(size_t length);
BACKEND_EXPORT void rs_infer_tensorarray_destroy(RSInferTensorArray* array);

// ----------------------------------------------------------------------------
// RSInferenceRunner
// ----------------------------------------------------------------------------
typedef enum RSInferenceBackend {
    RSInferenceBackend_TensorRT = 0,
    RSInferenceBackend_ONNXRuntime = 1,
    RSInferenceBackend_OpenVINO = 2,
    RSInferenceBackend_OpenCV = 3
} RSInferenceBackend;

/**
 * RSInferenceRunner is a typedef for a structure that encapsulates an inference runner instance, 
 * containing a pointer to the actual runner (such as TensorRTRunner) and an associated inference backend 
 * of type RSInferenceBackend. 
 * This structure is likely used to manage and execute inference tasks in a machine learning context.
 */
typedef struct RSInferenceRunner {
    void* runner; // Pointer to the actual inference runner instance (e.g., TensorRTRunner)
    RSInferenceBackend backend;
} RSInferenceRunner;

/**
 * Creates an RSInferenceRunner instance based on the provided model and configuration paths.
 * @param model_path: Path to the model file (e.g., ONNX, TensorRT engine).
 * @param json_config_str: JSON string containing the configuration (if applicable).
 * @param error: Pointer to an RSInferError structure to capture any errors during creation.
 * @return Pointer to the created RSInferenceRunner instance, or nullptr if creation fails.
 * 
 * @attention Implementation Note: This function is called by the Rust engine to create an inference runner.
 * In creating your own inference runner, you should implement this function to initialize the runner 
 * based on the specified model and configuration.
 * DO NOT CONSUME or FREE the model_path, json_config_str, or error parameter; they are managed by the Rust side. 
 * You should only populate the error parameter with error information if needed.
 * THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 */
BACKEND_EXPORT RSInferenceRunner* rs_inference_runner_create(const char* model_path, const char* json_config_str, RSInferError* error);
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
 * Retrieves the input tensor information from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve input tensor information
 * @param error: Pointer to an RSInferError structure to capture any errors during retrieval
 * @return Pointer to an RSInferTensorArray containing the input tensor information, or nullptr if retrieval fails.
 *
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to describe every model input, in model order.
 * Return one RSInferTensor per input with its name, data type and shape. Use -1 for a dynamic dimension, and an empty
 * shape when even the rank is unknown. Leave each tensor's data array empty (length 0); it carries no tensor data.
 * The returned RSInferTensorArray must be freshly allocated with the rs_infer_* helpers; the caller takes ownership and
 * frees it with rs_infer_tensorarray_destroy, so do not free or reuse it.
 * If you encounter an error, return nullptr and populate the RSInferError structure pointed to by the error parameter,
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT RSInferTensorArray* rs_inference_runner_get_input_tensors_info(RSInferenceRunner* runner, RSInferError* error);
/**
 * Retrieves the output tensor information from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve output tensor information
 * @param error: Pointer to an RSInferError structure to capture any errors during retrieval
 * @return Pointer to an RSInferTensorArray containing the output tensor information, or nullptr if retrieval fails.
 *
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: Same contract as rs_inference_runner_get_input_tensors_info, for the model outputs:
 * one RSInferTensor per output with name, data type and shape (-1 for dynamic dimensions, empty for an unknown rank),
 * an empty data array, and a freshly allocated array that the caller frees with rs_infer_tensorarray_destroy.
 */
BACKEND_EXPORT RSInferTensorArray* rs_inference_runner_get_output_tensors_info(RSInferenceRunner* runner, RSInferError* error);
/**
 * Retrieves the JSON configuration string from the provided RSInferenceRunner instance.
 * @param runner: Pointer to the RSInferenceRunner instance from which to retrieve the JSON configuration string
 * @param error: Pointer to an RSInferError structure to capture any errors during retrieval
 * @return Pointer to a dynamically allocated string containing the JSON configuration, or nullptr if retrieval fails.
 *
 * @attention THIS FUNCTION MUST BE THREAD-SAFE: It may be called from multiple threads concurrently, so ensure that your implementation is safe for concurrent execution.
 *
 * @attention Implementation Note: This function is called by the Rust engine to retrieve the JSON configuration string.
 * In creating your own inference runner, you should implement this function to return the JSON configuration string associated with the runner.
 * The returned string should be dynamically allocated and must be freed by the caller using rs_infer_string_free.
 * If you encounter an error while retrieving the JSON configuration, you can populate the RSInferError structure pointed to by the error parameter, 
 * but do not free it. The Rust side will handle the memory management of the error structure.
 */
BACKEND_EXPORT char* rs_inference_runner_get_json_config(RSInferenceRunner* runner, RSInferError* error);

#ifdef __cplusplus
}
#endif

#endif // RS_INFERENCE_H
