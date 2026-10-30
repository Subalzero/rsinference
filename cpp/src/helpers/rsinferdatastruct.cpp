#include "rsinference.h"

#include <stdlib.h>
#include <string.h>

RSInferLLArray* rs_infer_llarray_create(size_t length)
{
    RSInferLLArray* array = (RSInferLLArray*)malloc(sizeof(RSInferLLArray));
    if (!array) {
        return NULL;
    }
    array->data = (long long*)malloc(length * sizeof(long long));
    if (!array->data) {
        free(array);
        return NULL;
    }
    array->length = length;
    return array;
}

void rs_infer_llarray_destroy(RSInferLLArray* array)
{
    if (array) {
        free(array->data);
        free(array);
    }
}

RSInferFPArray* rs_infer_fparray_create(size_t length)
{
    RSInferFPArray* array = (RSInferFPArray*)malloc(sizeof(RSInferFPArray));
    if (!array) {
        return NULL;
    }
    array->data = (float*)malloc(length * sizeof(float));
    if (!array->data) {
        free(array);
        return NULL;
    }
    array->length = length;
    return array;
}

void rs_infer_fparray_destroy(RSInferFPArray* array)
{
    if (array) {
        free(array->data);
        free(array);
    }
}

RSInferIntArray* rs_infer_intarray_create(size_t length)
{
    RSInferIntArray* array = (RSInferIntArray*)malloc(sizeof(RSInferIntArray));
    if (!array) {
        return NULL;
    }
    array->data = (int*)malloc(length * sizeof(int));
    if (!array->data) {
        free(array);
        return NULL;
    }
    array->length = length;
    return array;
}

void rs_infer_intarray_destroy(RSInferIntArray* array)
{
    if (array) {
        free(array->data);
        free(array);
    }
}

void rs_infer_error_cleanmsg(RSInferError* error)
{
    if (error && error->message) {
        free(error->message);
        error->message = NULL;
    }
}

RSInferTensor* rs_infer_tensor_create(const char* name, size_t name_length, RSInferDataType dtype, RSInferUCharArray* data, RSInferLLArray* shape)
{
    RSInferTensor* tensor = (RSInferTensor*)malloc(sizeof(RSInferTensor));
    if (!tensor) {
        return NULL;
    }
    tensor->name = (char*)malloc(name_length + 1);
    if (!tensor->name) {
        free(tensor);
        return NULL;
    }
    memcpy(tensor->name, name, name_length);
    tensor->name[name_length] = '\0';
    tensor->name_length = name_length;
    tensor->dtype = dtype;
    tensor->data = data;
    tensor->shape = shape;
    return tensor;
}

void rs_infer_tensor_destroy(RSInferTensor* tensor)
{
    if (tensor) {
        free(tensor->name);
        rs_infer_uchararray_destroy(tensor->data);
        rs_infer_llarray_destroy(tensor->shape);
        free(tensor);
    }
}

RSInferTensorArray* rs_infer_tensorarray_create(size_t length)
{
    RSInferTensorArray* array = (RSInferTensorArray*)malloc(sizeof(RSInferTensorArray));
    if (!array) {
        return NULL;
    }
    array->tensors = (RSInferTensor**)malloc(length * sizeof(RSInferTensor*));
    if (!array->tensors) {
        free(array);
        return NULL;
    }
    array->length = length;
    return array;
}

void rs_infer_tensorarray_destroy(RSInferTensorArray* array)
{
    if (array) {
        for (size_t i = 0; i < array->length; ++i) {
            rs_infer_tensor_destroy(array->tensors[i]);
        }
        free(array->tensors);
        free(array);
    }
}