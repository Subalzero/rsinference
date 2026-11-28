#pragma once

#include "common/tensor.h"

#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <variant>
#include <expected>

namespace rsinfer
{
    using namespace nvinfer1;

    struct RunnerParams
    {
        int32_t dlaCore{ -1 };               //!< Specify the DLA core to run network on.
        // Legacy compatibility fields. TensorRT 11 uses strongly typed networks,
        // so these do not override the ONNX graph's tensor precision.
        bool int8{ false };
        bool fp16{ false };
        bool bf16{ false };
        std::vector<std::string> dataDirs; //!< Directory paths where sample data files are stored
        std::vector<std::string> inputTensorNames;
        std::vector<std::string> outputTensorNames;
        std::string timingCacheFile; //!< Path to timing cache file
        std::string onnxFileName;  //!< REQUIRED: Model file in ONNX format.
        uint32_t nContexts{ 1 };  //!< Number of contexts tensorrt will generate
        bool dynamic{ false };
        //! Record each context's copy-in, execution and copy-out as a CUDA graph
        //! and replay it while the input shapes stay the same. A context whose
        //! engine cannot be captured falls back to plain enqueueV3.
        bool cudaGraph{ true };

        // TensorRT builder configuration. Empty optionals preserve TensorRT defaults.
        std::vector<nvinfer1::BuilderFlag> builderFlags;
        std::optional<int32_t> builderOptimizationLevel;
        std::optional<int32_t> avgTimingIterations;
        std::optional<int32_t> maxAuxStreams;
        std::optional<int32_t> maxThreads;
        std::optional<uint32_t> tacticSources;
        std::optional<nvinfer1::ProfilingVerbosity> profilingVerbosity;
        std::optional<nvinfer1::EngineCapability> engineCapability;
        std::optional<nvinfer1::HardwareCompatibilityLevel> hardwareCompatibilityLevel;
        std::optional<nvinfer1::DeviceType> defaultDeviceType;
        std::optional<nvinfer1::RuntimePlatform> runtimePlatform;
        std::optional<nvinfer1::TilingOptimizationLevel> tilingOptimizationLevel;
        std::optional<int64_t> tilingL2Limit;
        std::vector<std::pair<nvinfer1::PreviewFeature, bool>> previewFeatures;
        std::vector<std::string> pluginsToSerialize;
        std::map<std::string, nvinfer1::DeviceType> layerDeviceTypes;
        bool allowGpuFallback{ true };
        std::map<nvinfer1::MemoryPoolType, size_t> memoryPoolLimits;

        // Optional engine cache path overrides the default model-derived filename.
        std::string engineFile;

        //! Optimization profile shapes per input tensor, as given in the "profile" config.
        //! Kept so the effective configuration can be reported back.
        struct ProfileDims
        {
            nvinfer1::Dims min, opt, max;
        };
        std::map<std::string, ProfileDims> profile;
    };

    //! Borrowed view of one input tensor. The runner reads `data` while the call
    //! that receives the view runs (it copies the bytes into pinned memory before
    //! returning, also for process_async), so the caller may free it afterwards.
    struct TensorView
    {
        std::string_view name;
        std::span<const int64_t> shape;
        rsinfer::DataType dtype{ rsinfer::DataType::FP32 };
        std::span<const unsigned char> data;
    };

    class TensorRTRunner
    {
    public:
        TensorRTRunner();
        ~TensorRTRunner();

        TensorRTRunner(const RunnerParams& params);

        TensorRTRunner(const TensorRTRunner& copy) = delete;
        TensorRTRunner& operator=(const TensorRTRunner& copy) = delete;

        TensorRTRunner(TensorRTRunner&& temp) noexcept;
        TensorRTRunner& operator=(TensorRTRunner&& temp) noexcept;

        bool build(const std::function<bool(IOptimizationProfile*)>& optimization = {});

        void process(std::span<const TensorView> input_tensors);
        void process(const std::vector<Tensor>& input_tensors);

        //! Validates the inputs, copies them into an execution slot's pinned
        //! buffers and queues the work on the slot's stream before returning.
        //! Waits for a free slot when all n_contexts slots are busy.
        void process_async(std::span<const TensorView> input_tensors);
        void process_async(const std::vector<Tensor>& input_tensors);

        void get(std::vector<Tensor>& result);

        std::vector<int> get_input_shape(uint32_t index) const;
        std::vector<int> get_output_shape(uint32_t index) const;
        nvinfer1::DataType get_tensor_data_type(const std::string& name) const;
        size_t get_tensor_data_size(const nvinfer1::DataType& data_type) const;
        std::string get_input_tensor_name(uint32_t index) const;
        std::string get_output_tensor_name(uint32_t index) const;
        size_t get_number_of_inputs() const;
        size_t get_number_of_outputs() const;

        std::expected<std::vector<Tensor>, std::string> get_input_tensors_info() const;
        std::expected<std::vector<Tensor>, std::string> get_output_tensors_info() const;

        RunnerParams get_config() const;

        static nvinfer1::Dims vector_to_dims(const std::vector<int>& vec);

        static std::vector<int> dims_to_vector(const nvinfer1::Dims& dims);

    private:
        RunnerParams _params;

        std::shared_ptr<nvinfer1::IRuntime> _runtime;   //!< The TensorRT runtime used to deserialize the engine
        std::shared_ptr<nvinfer1::ICudaEngine> _engine; //!< The TensorRT engine used to run the network

        //! One engine I/O tensor, in engine order.
        struct IOTensor
        {
            std::string name;
            bool is_input{ false };
            nvinfer1::DataType type{ nvinfer1::DataType::kFLOAT };
        };

        //! Device buffer plus its pinned host staging buffer. Grown, never
        //! shrunk, so steady-state requests do no allocation.
        struct IOBuffer
        {
            void* device{ nullptr };
            void* host{ nullptr };
            size_t capacity{ 0 }; //!< Bytes allocated in both buffers.
            size_t bytes{ 0 };    //!< Bytes used by the current request.
            nvinfer1::Dims shape{}; //!< Resolved shape of the current request.
        };

        //! Everything one in-flight request needs: an execution context, its own
        //! stream, and persistent buffers for every I/O tensor (indexed like _io).
        struct Slot
        {
            TensorRTRunner* owner{ nullptr };
            std::unique_ptr<nvinfer1::IExecutionContext> context;
            cudaStream_t stream{ nullptr };
            std::vector<IOBuffer> buffers;

            //! Replayable copy-in + enqueueV3 + copy-out for `graph_key`.
            cudaGraphExec_t graph{ nullptr };
            //! Input shapes the graph (or the last direct run) was made for.
            std::vector<int64_t> graph_key;
            //! The last request ran directly with `graph_key`, so TensorRT has done
            //! its deferred work for these shapes and the next one may be captured.
            bool ran_direct{ false };
            //! Capture failed once; this context always runs directly.
            bool graph_unsupported{ false };
        };

        std::vector<IOTensor> _io;
        std::vector<size_t> _input_order;  //!< _io indices in get_input_tensor_name order.
        std::vector<size_t> _output_order; //!< _io indices in get_output_tensor_name order.

        std::vector<std::unique_ptr<Slot>> _slots;
        std::queue<Slot*> _idle_slots;
        std::mutex _idle_slots_mut;
        std::condition_variable _idle_slots_cond;

        using Completion = std::variant<std::vector<Tensor>, std::string>;
        std::queue<Completion> _completions;
        std::mutex _completions_mut;
        std::condition_variable _completions_cond;
        size_t _pending_requests{ 0 };

        //! Requests whose stream callback has not run yet; the destructor waits for them.
        int _inflight = 0;
        std::mutex _inflight_mut;
        std::condition_variable _inflight_cond;

        bool init_slots();
        void release_slots() noexcept;
        Slot* acquire_slot();
        void return_slot(Slot* slot) noexcept;
        //! Validates the inputs and queues copy-in, execution and copy-out on a
        //! slot's stream. Returns the slot, still owned by the caller.
        Slot* enqueue(std::span<const TensorView> input_tensors);
        void ensure_capacity(IOBuffer& buffer, size_t bytes);
        //! Queues copy-in, enqueueV3 and copy-out on the slot's stream.
        void record_work(Slot& slot);
        //! Runs the queued request through the slot's graph when possible.
        void launch(Slot& slot, std::vector<int64_t>&& key, bool buffers_moved);
        bool capture_graph(Slot& slot);
        static void destroy_graph(Slot& slot) noexcept;
        std::vector<Tensor> collect_outputs(const Slot& slot) const;
        void push_completion(Completion completion);
        void finish_request();
        static void CUDART_CB on_stream_complete(cudaStream_t stream, cudaError_t status, void* user_data);
        static std::vector<TensorView> views_of(const std::vector<Tensor>& tensors);

        bool construct_network(std::unique_ptr<nvinfer1::IBuilder>& builder, std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
            std::unique_ptr<nvonnxparser::IParser>& parser, std::unique_ptr<nvinfer1::ITimingCache>& timingCache);

        //!
        //! \brief Load plan file
        //! 
        bool load_engine(std::vector<char>& data);

        //!
        //! \brief Save engine to .plan file
        //! 
        bool save_engine_to_plan_file(nvinfer1::IHostMemory* plan);
    };
}

