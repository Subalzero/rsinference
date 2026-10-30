// Unit tests for the C-compatible FFI layer exported from
// infertensorrt_cpp.dll (see include/public/rsinference.h).
//
// These tests exercise the exported data-structure helpers and the runner's
// argument-validation paths, which work without a GPU or an engine file.

#include <gtest/gtest.h>

#include "rsinference.h"

#include <cstring>
#include <string>
#include <vector>

// The impl's error codes are defined in its translation unit, not the public
// header; the contract tested here is that invalid-argument paths report 1.
enum TestErrorCodes
{
    kInvalidArguments = 1,
    kSuccess = 0
};

// ---------------------------------------------------------------------------
// RSInferLLArray (long long)
// ---------------------------------------------------------------------------

TEST(LLArray, CreateInitializesFields)
{
    RSInferLLArray* array = rs_infer_llarray_create(4);
    ASSERT_NE(array, nullptr);
    EXPECT_NE(array->data, nullptr);
    EXPECT_EQ(array->length, 4U);
    for (size_t i = 0; i < array->length; ++i) {
        array->data[i] = static_cast<long long>(i * 10);
    }
    EXPECT_EQ(array->data[3], 30);
    rs_infer_llarray_destroy(array);
}

TEST(LLArray, CreateZeroLengthIsUsable)
{
    RSInferLLArray* array = rs_infer_llarray_create(0);
    ASSERT_NE(array, nullptr);
    EXPECT_EQ(array->length, 0U);
    rs_infer_llarray_destroy(array);
}

TEST(LLArray, DestroyNullIsNoOp)
{
    rs_infer_llarray_destroy(nullptr);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// RSInferFPArray (float)
// ---------------------------------------------------------------------------

TEST(FPArray, CreateRoundTripsValues)
{
    RSInferFPArray* array = rs_infer_fparray_create(3);
    ASSERT_NE(array, nullptr);
    EXPECT_EQ(array->length, 3U);
    array->data[0] = 1.5F;
    array->data[1] = -2.25F;
    array->data[2] = 0.0F;
    EXPECT_FLOAT_EQ(array->data[0], 1.5F);
    EXPECT_FLOAT_EQ(array->data[1], -2.25F);
    rs_infer_fparray_destroy(array);
}

TEST(FPArray, DestroyNullIsNoOp)
{
    rs_infer_fparray_destroy(nullptr);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// RSInferUCharArray (unsigned char)
// ---------------------------------------------------------------------------

TEST(UCharArray, CreateInitializesFields)
{
    RSInferUCharArray* array = rs_infer_uchararray_create(8);
    ASSERT_NE(array, nullptr);
    EXPECT_NE(array->data, nullptr);
    EXPECT_EQ(array->length, 8U);
    memset(array->data, 0xAB, array->length);
    EXPECT_EQ(array->data[0], 0xAB);
    EXPECT_EQ(array->data[7], 0xAB);
    rs_infer_uchararray_destroy(array);
}

TEST(UCharArray, DestroyNullIsNoOp)
{
    rs_infer_uchararray_destroy(nullptr);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// RSInferIntArray (int)
// ---------------------------------------------------------------------------

TEST(IntArray, CreateRoundTripsValues)
{
    RSInferIntArray* array = rs_infer_intarray_create(2);
    ASSERT_NE(array, nullptr);
    EXPECT_EQ(array->length, 2U);
    array->data[0] = 42;
    array->data[1] = -7;
    EXPECT_EQ(array->data[0], 42);
    EXPECT_EQ(array->data[1], -7);
    rs_infer_intarray_destroy(array);
}

TEST(IntArray, DestroyNullIsNoOp)
{
    rs_infer_intarray_destroy(nullptr);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// RSInferError
// ---------------------------------------------------------------------------

TEST(Error, CleanMsgFreesMessageAndNullsIt)
{
    RSInferError error{};
    error.message = strdup("boom");
    ASSERT_NE(error.message, nullptr);
    rs_infer_error_cleanmsg(&error);
    EXPECT_EQ(error.message, nullptr);
}

TEST(Error, CleanMsgNullMessageIsSafe)
{
    RSInferError error{};
    error.message = nullptr;
    rs_infer_error_cleanmsg(&error);
    EXPECT_EQ(error.message, nullptr);
}

TEST(Error, CleanMsgNullErrorIsSafe)
{
    rs_infer_error_cleanmsg(nullptr);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// RSInferTensor
// ---------------------------------------------------------------------------

TEST(Tensor, CreateCopiesNameAndAdoptsBuffers)
{
    RSInferUCharArray* data = rs_infer_uchararray_create(12);
    RSInferLLArray* shape = rs_infer_llarray_create(4);
    shape->data[0] = 1;
    shape->data[1] = 3;
    shape->data[2] = 2;
    shape->data[3] = 2;

    const char* name = "input";
    RSInferTensor* tensor =
        rs_infer_tensor_create(name, std::strlen(name), RSInferDataType_Float32, data, shape);
    ASSERT_NE(tensor, nullptr);

    EXPECT_STREQ(tensor->name, "input");
    EXPECT_EQ(tensor->name_length, std::strlen("input"));
    EXPECT_EQ(tensor->dtype, RSInferDataType_Float32);
    EXPECT_EQ(tensor->data, data);
    EXPECT_EQ(tensor->shape, shape);
    EXPECT_EQ(tensor->shape->data[1], 3);

    // Destroying the tensor must also free the adopted buffers.
    rs_infer_tensor_destroy(tensor);
}

TEST(Tensor, CreateWithNullBuffersIsUsable)
{
    RSInferTensor* tensor =
        rs_infer_tensor_create("t", 1, RSInferDataType_Int32, nullptr, nullptr);
    ASSERT_NE(tensor, nullptr);
    EXPECT_STREQ(tensor->name, "t");
    EXPECT_EQ(tensor->data, nullptr);
    EXPECT_EQ(tensor->shape, nullptr);
    rs_infer_tensor_destroy(tensor);
}

TEST(Tensor, DestroyNullIsNoOp)
{
    rs_infer_tensor_destroy(nullptr);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// RSInferTensorArray
// ---------------------------------------------------------------------------

TEST(TensorArray, CreateInitializesFields)
{
    RSInferTensorArray* array = rs_infer_tensorarray_create(2);
    ASSERT_NE(array, nullptr);
    EXPECT_NE(array->tensors, nullptr);
    EXPECT_EQ(array->length, 2U);

    // Fill both slots with owned tensors, then destroy the array: it must
    // free every tensor it holds.
    for (size_t i = 0; i < array->length; ++i) {
        array->tensors[i] =
            rs_infer_tensor_create("x", 1, RSInferDataType_UInt8, nullptr, nullptr);
        ASSERT_NE(array->tensors[i], nullptr);
    }
    rs_infer_tensorarray_destroy(array);
}

TEST(TensorArray, DestroyNullIsNoOp)
{
    rs_infer_tensorarray_destroy(nullptr);
    SUCCEED();
}

// ---------------------------------------------------------------------------
// rs_inference_runner_* argument validation (no GPU required)
// ---------------------------------------------------------------------------

TEST(RunnerCreate, NullModelPathSetsErrorAndReturnsNull)
{
    RSInferError error{};
    RSInferenceRunner* runner = rs_inference_runner_create(nullptr, nullptr, &error);
    EXPECT_EQ(runner, nullptr);
    EXPECT_EQ(error.code, kInvalidArguments);
    ASSERT_NE(error.message, nullptr);
    EXPECT_STREQ(error.message, "Model path cannot be null");
    rs_infer_error_cleanmsg(&error);
}

TEST(RunnerCreate, NullModelPathWithoutErrorPointerStillReturnsNull)
{
    RSInferenceRunner* runner = rs_inference_runner_create(nullptr, nullptr, nullptr);
    EXPECT_EQ(runner, nullptr);
}

TEST(RunnerEntryPoints, NullRunnerIsRejected)
{
    RSInferError error{};

    EXPECT_EQ(rs_inference_runner_infer(nullptr, nullptr, &error), kInvalidArguments);
    EXPECT_EQ(error.code, kInvalidArguments);
    rs_infer_error_cleanmsg(&error);

    error = RSInferError{};
    EXPECT_EQ(rs_inference_runner_infer_async(nullptr, nullptr, &error), kInvalidArguments);
    EXPECT_EQ(error.code, kInvalidArguments);
    rs_infer_error_cleanmsg(&error);

    error = RSInferError{};
    EXPECT_EQ(rs_inference_runner_get_results(nullptr, &error), nullptr);
    EXPECT_EQ(error.code, kInvalidArguments);
    rs_infer_error_cleanmsg(&error);

    error = RSInferError{};
    EXPECT_EQ(rs_inference_runner_get_input_shape(nullptr, 0, &error), nullptr);
    EXPECT_EQ(error.code, kInvalidArguments);
    rs_infer_error_cleanmsg(&error);

    error = RSInferError{};
    EXPECT_EQ(rs_inference_runner_get_output_shape(nullptr, 0, &error), nullptr);
    EXPECT_EQ(error.code, kInvalidArguments);
    rs_infer_error_cleanmsg(&error);

    error = RSInferError{};
    EXPECT_EQ(rs_inference_runner_get_tensor_data_type(nullptr, "input", &error),
              RSInferDataType_Unknown);
    EXPECT_EQ(error.code, kInvalidArguments);
    rs_infer_error_cleanmsg(&error);

    EXPECT_EQ(rs_inference_runner_get_tensor_data_size(nullptr, RSInferDataType_Float32, &error), 0U);
    EXPECT_EQ(rs_inference_runner_get_input_tensor_name(nullptr, 0, &error), nullptr);
    EXPECT_EQ(rs_inference_runner_get_output_tensor_name(nullptr, 0, &error), nullptr);
}

TEST(RunnerEntryPoints, NullTensorNameIsRejected)
{
    RSInferError error{};
    EXPECT_EQ(rs_inference_runner_get_tensor_data_type(nullptr, nullptr, &error),
              RSInferDataType_Unknown);
    EXPECT_EQ(error.code, kInvalidArguments);
    rs_infer_error_cleanmsg(&error);
}

// ---------------------------------------------------------------------------
// Dynamic-loading verification: exercise GetProcAddress against the DLL,
// confirming every BACKEND_EXPORT declared in rsinference.h resolves.
// ---------------------------------------------------------------------------
#ifdef _WIN32
#include <windows.h>

static bool verify_export(const char* name)
{
    HMODULE dll = LoadLibraryEx(
        "infertensorrt_cpp.dll",
        nullptr,
        LOAD_LIBRARY_SEARCH_SYSTEM32 // prefers the DLL that was already located
    );
    if (!dll) return false;
    FARPROC fn = GetProcAddress(dll, name);
    FreeLibrary(dll);
    return fn != nullptr;
}

TEST(DynamicLoad, AllHeaderExportsPresentInDll)
{
    // Gather the exact symbol names from the public header.
    // We hand-craft the list so the test stays independent of a regex parser.
    const char* expected[] = {
        "rs_infer_llarray_create",
        "rs_infer_llarray_destroy",
        "rs_infer_fparray_create",
        "rs_infer_fparray_destroy",
        "rs_infer_uchararray_create",
        "rs_infer_uchararray_destroy",
        "rs_infer_intarray_create",
        "rs_infer_intarray_destroy",
        "rs_infer_error_cleanmsg",
        "rs_infer_tensor_create",
        "rs_infer_tensor_destroy",
        "rs_infer_tensorarray_create",
        "rs_infer_tensorarray_destroy",
        "rs_inference_runner_create",
        "rs_inference_runner_destroy",
        "rs_inference_runner_infer",
        "rs_inference_runner_infer_async",
        "rs_inference_runner_get_results",
        "rs_inference_runner_get_input_shape",
        "rs_inference_runner_get_output_shape",
        "rs_inference_runner_get_tensor_data_type",
        "rs_inference_runner_get_tensor_data_size",
        "rs_inference_runner_get_input_tensor_name",
        "rs_inference_runner_get_output_tensor_name",
    };
    constexpr int kCount = sizeof(expected) / sizeof(expected[0]);

    for (int i = 0; i < kCount; ++i) {
        const char* sym = expected[i];
        EXPECT_TRUE(verify_export(sym))
            << "Export '" << sym << "' not found in infertensorrt_cpp.dll";
    }
}
#endif // _WIN32
