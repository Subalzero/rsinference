#pragma once

#include <cstddef>

#include "rsinference.h"
#include "common/tensor.h"

// Conversions between the public C enum (RSInferDataType) and the internal
// rsinfer::DataType. The two enums use different value orders, so they must be
// translated through these functions, never cast. Every backend adapter shares
// them so a new type only has to be added in one place.
namespace rsinfer {

    inline DataType from_public_data_type(RSInferDataType type) noexcept
    {
        switch (type)
        {
        case RSInferDataType_Float32: return DataType::FP32;
        case RSInferDataType_Int32:   return DataType::INT32;
        case RSInferDataType_Int64:   return DataType::INT64;
        case RSInferDataType_UInt8:   return DataType::UINT8;
        case RSInferDataType_Int8:    return DataType::INT8;
        case RSInferDataType_Bool:    return DataType::BOOL;
        case RSInferDataType_Half:    return DataType::FP16;
        case RSInferDataType_FP8:     return DataType::FP8;
        case RSInferDataType_FP8E5M2: return DataType::FP8E5M2;
        case RSInferDataType_E8M0:    return DataType::E8M0;
        case RSInferDataType_BF16:    return DataType::BF16;
        case RSInferDataType_Int4:    return DataType::INT4;
        case RSInferDataType_FP4:     return DataType::FP4;
        default:                      return DataType::UNKNOWN;
        }
    }

    inline RSInferDataType to_public_data_type(DataType type) noexcept
    {
        switch (type)
        {
        case DataType::FP32:    return RSInferDataType_Float32;
        case DataType::INT32:   return RSInferDataType_Int32;
        case DataType::INT64:   return RSInferDataType_Int64;
        case DataType::UINT8:   return RSInferDataType_UInt8;
        case DataType::INT8:    return RSInferDataType_Int8;
        case DataType::BOOL:    return RSInferDataType_Bool;
        case DataType::FP16:    return RSInferDataType_Half;
        case DataType::FP8:     return RSInferDataType_FP8;
        case DataType::FP8E5M2: return RSInferDataType_FP8E5M2;
        case DataType::E8M0:    return RSInferDataType_E8M0;
        case DataType::BF16:    return RSInferDataType_BF16;
        case DataType::INT4:    return RSInferDataType_Int4;
        case DataType::FP4:     return RSInferDataType_FP4;
        default:                return RSInferDataType_Unknown;
        }
    }

    // Bytes per element. Packed 4-bit types report 1, their one-byte storage
    // unit; use packed_byte_count for the size of a buffer. Returns 0 for
    // RSInferDataType_Unknown and out-of-range values.
    inline size_t data_type_element_size(DataType type) noexcept
    {
        switch (type)
        {
        case DataType::INT64:   return 8;
        case DataType::FP32:
        case DataType::INT32:   return 4;
        case DataType::FP16:
        case DataType::BF16:    return 2;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::BOOL:
        case DataType::FP8:
        case DataType::FP8E5M2:
        case DataType::E8M0:
        case DataType::INT4:
        case DataType::FP4:     return 1;
        default:                return 0;
        }
    }

    // Bytes needed for element_count elements, accounting for 4-bit packing.
    // Returns 0 for unknown types.
    inline size_t packed_byte_count(DataType type, size_t element_count) noexcept
    {
        if (type == DataType::INT4 || type == DataType::FP4)
            return (element_count + 1) / 2;
        return element_count * data_type_element_size(type);
    }
}
