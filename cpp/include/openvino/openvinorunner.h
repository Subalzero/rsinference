#ifndef OPENVINO_RUNNER_H
#define OPENVINO_RUNNER_H

#include "common/tensor.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <optional>
#include <queue>
#include <string>
#include <variant>
#include <vector>
#include <mutex>

#include "openvino/openvino.hpp"

namespace rsinfer
{
    struct OpenVINORunnerParams
    {
        std::string modelXmlPath; //!< Required OpenVINO IR XML model path.
        std::string modelWeightsPath; //!< Optional BIN weights path when it is not beside the XML.
        std::string device{ "CPU" }; //!< Device name, e.g. CPU, GPU, NPU, AUTO, or HETERO:...
        uint32_t numInferRequests{ 0 }; //!< 0 uses OpenVINO's optimal request count.

        // Portable performance and execution hints.
        std::optional<ov::hint::PerformanceMode> performanceMode{ ov::hint::PerformanceMode::THROUGHPUT };
        std::optional<ov::element::Type> inferencePrecision{ ov::element::f32 };
        std::optional<ov::hint::ExecutionMode> executionMode;
        std::optional<ov::hint::Priority> modelPriority;
        std::optional<uint32_t> performanceHintNumRequests;

        // Common scheduling, profiling, and cache settings. Support depends on device/plugin.
        std::optional<ov::streams::Num> numStreams;
        std::optional<int32_t> inferenceNumThreads;
        std::optional<int32_t> compilationNumThreads;
        std::optional<ov::hint::SchedulingCoreType> schedulingCoreType;
        std::optional<bool> enableCpuPinning;
        std::optional<bool> enableHyperThreading;
        std::optional<bool> enableProfiling;
        std::optional<std::string> cacheDir;

        // Any additional common, plugin-specific, or future compile property.
        // Explicit fields above override matching entries in this map.
        ov::AnyMap compileProperties;
    };

    class OpenVINORunner
    {
    public:
        OpenVINORunner();
        ~OpenVINORunner();

        explicit OpenVINORunner(const OpenVINORunnerParams& params);
        OpenVINORunner(const std::string& xml_path, const std::string& default_device);

        OpenVINORunner(OpenVINORunner&& temp) noexcept;
        OpenVINORunner& operator=(OpenVINORunner&& temp) noexcept;

        OpenVINORunner(const OpenVINORunner& copy) = delete;
        OpenVINORunner& operator=(const OpenVINORunner& copy) = delete;

        std::vector<int> get_input_shape(uint32_t index) const;
        std::vector<int> get_output_shape(uint32_t index) const;
        std::string get_input_tensor_name(uint32_t index) const;
        std::string get_output_tensor_name(uint32_t index) const;
        size_t get_number_of_inputs() const;
        size_t get_number_of_outputs() const;
        ov::element::Type get_tensor_element_type(const std::string& tensor_name) const;
        rsinfer::DataType get_tensor_data_type(const std::string& tensor_name) const;
        size_t get_tensor_data_size(const rsinfer::DataType& data_type) const;

        void process(const std::vector<rsinfer::Tensor>& input_tensors);
        void process_async(const std::vector<rsinfer::Tensor>& input_tensors);

        void get(std::vector<rsinfer::Tensor>& output_tensors);

        static std::vector<std::string> get_available_devices();
        static std::string get_device_name(const std::string& device);

    private:
        // Process-wide Core, intentionally never destroyed. A static ov::Core
        // would be destroyed from DllMain(DLL_PROCESS_DETACH) under the loader
        // lock; its destructor unloads device plugins (e.g. the GPU plugin) and
        // joins their worker threads, which deadlocks on Windows.
        static ov::Core& core();

        using Completion = std::variant<std::vector<rsinfer::Tensor>, std::string>;

        void begin_request();
        void finish_request() noexcept;
        std::vector<rsinfer::Tensor> process_impl(const std::vector<rsinfer::Tensor>& input_tensors);

        OpenVINORunnerParams _params;

        ov::CompiledModel _compiled_model;

        std::vector<ov::InferRequest> _infer_requests;
        std::queue<ov::InferRequest*> _idle_requests;

        std::queue<Completion> _completions;

        std::mutex _idle_requests_mut;
        std::condition_variable _idle_requests_cond;

        std::mutex _completions_mut;
        std::condition_variable _completions_cond;
        size_t _pending_requests{ 0 };
        size_t _active_calls{ 0 };
    };
}


#endif // OPENVINO_RUNNER_H
