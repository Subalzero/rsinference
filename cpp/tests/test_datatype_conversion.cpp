// Unit tests for the public <-> internal data type conversions shared by every
// backend adapter (see include/common/datatypeconversion.h).

#include <gtest/gtest.h>

#include "common/datatypeconversion.h"

#include <iterator>
#include <set>

namespace
{
    struct Mapping
    {
        RSInferDataType public_type;
        rsinfer::DataType internal_type;
        size_t element_size;
    };

    // Every public type except Unknown. Keep in sync with rsinference.h.
    const Mapping kMappings[] = {
        {RSInferDataType_Float32, rsinfer::DataType::FP32, 4},
        {RSInferDataType_Int32, rsinfer::DataType::INT32, 4},
        {RSInferDataType_Int64, rsinfer::DataType::INT64, 8},
        {RSInferDataType_UInt8, rsinfer::DataType::UINT8, 1},
        {RSInferDataType_Bool, rsinfer::DataType::BOOL, 1},
        {RSInferDataType_Half, rsinfer::DataType::FP16, 2},
        {RSInferDataType_FP8, rsinfer::DataType::FP8, 1},
        {RSInferDataType_E8M0, rsinfer::DataType::E8M0, 1},
        {RSInferDataType_BF16, rsinfer::DataType::BF16, 2},
        {RSInferDataType_Int4, rsinfer::DataType::INT4, 1},
        {RSInferDataType_FP4, rsinfer::DataType::FP4, 1},
        {RSInferDataType_Int8, rsinfer::DataType::INT8, 1},
        {RSInferDataType_FP8E5M2, rsinfer::DataType::FP8E5M2, 1},
    };
}

TEST(DataTypeConversion, EveryPublicTypeIsCovered)
{
    // One entry per public value below Unknown, so a new enum value without a
    // mapping fails here.
    EXPECT_EQ(std::size(kMappings), static_cast<size_t>(RSInferDataType_Unknown));
}

TEST(DataTypeConversion, PublicToInternal)
{
    for (const auto& m : kMappings)
        EXPECT_EQ(rsinfer::from_public_data_type(m.public_type), m.internal_type) << m.public_type;
}

TEST(DataTypeConversion, InternalToPublic)
{
    for (const auto& m : kMappings)
        EXPECT_EQ(rsinfer::to_public_data_type(m.internal_type), m.public_type) << m.public_type;
}

TEST(DataTypeConversion, MappingIsOneToOne)
{
    std::set<rsinfer::DataType> seen;
    for (const auto& m : kMappings)
        EXPECT_TRUE(seen.insert(rsinfer::from_public_data_type(m.public_type)).second) << m.public_type;
}

TEST(DataTypeConversion, UnknownAndOutOfRangeValues)
{
    EXPECT_EQ(rsinfer::from_public_data_type(RSInferDataType_Unknown), rsinfer::DataType::UNKNOWN);
    EXPECT_EQ(rsinfer::from_public_data_type(static_cast<RSInferDataType>(-1)), rsinfer::DataType::UNKNOWN);
    EXPECT_EQ(rsinfer::from_public_data_type(static_cast<RSInferDataType>(1000)), rsinfer::DataType::UNKNOWN);
    EXPECT_EQ(rsinfer::to_public_data_type(rsinfer::DataType::UNKNOWN), RSInferDataType_Unknown);
}

TEST(DataTypeConversion, ElementSizes)
{
    for (const auto& m : kMappings)
        EXPECT_EQ(rsinfer::data_type_element_size(m.internal_type), m.element_size) << m.public_type;
    EXPECT_EQ(rsinfer::data_type_element_size(rsinfer::DataType::UNKNOWN), 0U);
}

TEST(DataTypeConversion, PackedByteCount)
{
    EXPECT_EQ(rsinfer::packed_byte_count(rsinfer::DataType::FP32, 3), 12U);
    EXPECT_EQ(rsinfer::packed_byte_count(rsinfer::DataType::INT64, 2), 16U);
    EXPECT_EQ(rsinfer::packed_byte_count(rsinfer::DataType::INT4, 4), 2U);
    EXPECT_EQ(rsinfer::packed_byte_count(rsinfer::DataType::INT4, 5), 3U);
    EXPECT_EQ(rsinfer::packed_byte_count(rsinfer::DataType::FP4, 0), 0U);
    EXPECT_EQ(rsinfer::packed_byte_count(rsinfer::DataType::UNKNOWN, 10), 0U);
}
