#ifndef OPENVINO_RUNNER_H
#define OPENVINO_RUNNER_H

#include "common/orderedcompletions.h"
#include "common/tensor.h"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>
#include <mutex>
#include <expected>

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

    //! Borrowed view of one input tensor. process() runs inference directly on
    //! `data`; process_async() copies it into the request's own buffer before
    //! returning. Either way the caller may free it once the call returns.
    //! An empty name matches the input at the view's position.
    struct OpenVINOInputView
    {
        std::string_view name;
        ov::Shape shape;
        rsinfer::DataType dtype{ rsinfer::DataType::FP32 };
        std::span<const unsigned char> data;
    };

    class OpenVINORunner
    {
    public:
        OpenVINORunner();
        ~OpenVINORunner();

        explicit OpenVINORunner(const OpenVINORunnerParams& params);
        OpenVINORunner(const std::string& xml_path, const std::string& default_device);

        // Not copyable or movable: each infer request's completion callback
        // points at its slot, which points back at this runner.
        OpenVINORunner(OpenVINORunner&& temp) = delete;
        OpenVINORunner& operator=(OpenVINORunner&& temp) = delete;
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

        std::expected<std::vector<rsinfer::Tensor>, std::string> get_input_tensors_info() const;
        std::expected<std::vector<rsinfer::Tensor>, std::string> get_output_tensors_info() const;

        OpenVINORunnerParams get_config() const;

        //! Runs inference on the caller's buffers (no input copy) and waits for it.
        void process(std::span<const OpenVINOInputView> input_tensors);
        void process(const std::vector<rsinfer::Tensor>& input_tensors);
        //! Validates the inputs, copies them into a free infer request's own
        //! buffers and starts it with start_async before returning. Waits for a
        //! free request when all of them are busy.
        void process_async(std::span<const OpenVINOInputView> input_tensors);
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

        using Completions = OrderedCompletions<std::vector<rsinfer::Tensor>>;
        using Completion = Completions::Completion;

        //! One infer request plus the input buffers it owns for async runs.
        struct Slot
        {
            OpenVINORunner* owner{ nullptr };
            ov::InferRequest request;
            //! Per model input; reused while the shape and type stay the same.
            std::vector<ov::Tensor> owned_inputs;
            //! Ticket of the running async request. The completion callback only
            //! acts while `async_pending` is set, so a callback raised by a
            //! synchronous infer() (if the plugin does that) is ignored.
            uint64_t ticket{ 0 };
            bool async_pending{ false };
            //! start_async was called since the slot was last waited on.
            bool started_async{ false };
        };

        //! Takes the request's completion ticket and counts it as an active call.
        uint64_t begin_request();
        void finish_request() noexcept;
        void complete_request(uint64_t ticket, Completion completion) noexcept;
        void cancel_request(uint64_t ticket) noexcept;

        Slot* acquire_slot();
        void return_slot(Slot* slot) noexcept;
        //! Validates the inputs and binds them to the slot's request: wrapping
        //! the caller's memory (borrow) or copying into the slot's own tensors.
        void bind_inputs(Slot& slot, std::span<const OpenVINOInputView> input_tensors, bool borrow);
        std::vector<rsinfer::Tensor> collect_outputs(Slot& slot) const;
        void on_async_complete(Slot& slot, std::exception_ptr error) noexcept;
        static std::vector<OpenVINOInputView> views_of(const std::vector<rsinfer::Tensor>& tensors);

        OpenVINORunnerParams _params;

        ov::CompiledModel _compiled_model;

        std::vector<std::unique_ptr<Slot>> _slots;
        std::queue<Slot*> _idle_slots;

        Completions _completions; //!< Results in submission order; guarded by _completions_mut.

        std::mutex _idle_slots_mut;
        std::condition_variable _idle_slots_cond;

        std::mutex _completions_mut;
        std::condition_variable _completions_cond;
        size_t _active_calls{ 0 };
    };
}


#endif // OPENVINO_RUNNER_H
