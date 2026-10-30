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

typedef struct _inference_runner {
    void* runner; // Pointer to the actual inference runner instance (e.g., TensorRTRunner)
    RSInferenceBackend backend;
} RSInferenceRunner;

BACKEND_EXPORT RSInferenceRunner* rs_inference_runner_create(const char* model_path, const char* config_path, RSInferError* error);
BACKEND_EXPORT void rs_inference_runner_destroy(RSInferenceRunner* runner);
BACKEND_EXPORT int rs_inference_runner_infer(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error);
BACKEND_EXPORT int rs_inference_runner_infer_async(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error);
BACKEND_EXPORT RSInferTensorArray* rs_inference_runner_get_results(RSInferenceRunner* runner, RSInferError* error);
BACKEND_EXPORT RSInferIntArray* rs_inference_runner_get_input_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error);
BACKEND_EXPORT RSInferIntArray* rs_inference_runner_get_output_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error);
BACKEND_EXPORT RSInferDataType rs_inference_runner_get_tensor_data_type(RSInferenceRunner* runner, const char* tensor_name, RSInferError* error);
BACKEND_EXPORT size_t rs_inference_runner_get_tensor_data_size(RSInferenceRunner* runner, RSInferDataType data_type, RSInferError* error);
BACKEND_EXPORT const char* rs_inference_runner_get_input_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error);
BACKEND_EXPORT const char* rs_inference_runner_get_output_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error);




#ifdef __cplusplus
}
#endif

#endif // RS_INFERENCE_H