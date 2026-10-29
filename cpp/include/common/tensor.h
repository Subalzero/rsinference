#pragma once

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <vector>
#include <queue>
#include <concepts>
#include <utility>
#include <typeinfo>
#include <unordered_map>

namespace inferrs {

    enum DataType {
        FP16,
        FP32,
        INT8,
        BF16,
        INT4,
        FP4,
        UNKNOWN
    };

    struct Tensor {
        std::vector<int64_t> shape;
        std::vector<unsigned char> data;
        std::string name;
        DataType dtype;

         // Constructor
        Tensor()
            : dtype(DataType::FP32) {}

        Tensor(const std::string& name, 
            const std::vector<int64_t>& shape, 
            std::vector<unsigned char>&& data, 
            const DataType& data_type)
            : name(name), shape(shape), data(std::move(data)), dtype(data_type) {}

        Tensor(const std::string& name, 
            const std::vector<int64_t>& shape, 
            const std::vector<unsigned char>& data, 
            const DataType& data_type)
            : name(name), shape(shape), data(data), dtype(data_type) {}

        // Move constructor
        Tensor(Tensor&& other) noexcept
            : name(std::move(other.name)), shape(std::move(other.shape)), data(std::move(other.data)), dtype(other.dtype) {}

        // Move assignment operator
        Tensor& operator=(Tensor&& other) noexcept {
            if (this != &other) {
                name = std::move(other.name);
                shape = std::move(other.shape);
                data = std::move(other.data);
                dtype = other.dtype;
            }
            return *this;
        }

        // Copy constructor
        Tensor(const Tensor& other)
            : name(other.name), shape(other.shape), data(other.data), dtype(other.dtype) {}

        // Copy assignment operator
        Tensor& operator=(const Tensor& other) {
            if (this != &other) {
                name = other.name;
                shape = other.shape;
                data = other.data;
                dtype = other.dtype;
            }
            return *this;
        }
    };
}