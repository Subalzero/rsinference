#include "rsinference.h"
#include "tensorrtrunner.h"

#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

enum ErrorCodes
{
    ERROR_CODE_SUCCESS = 0,
    ERROR_CODE_INVALID_ARGUMENTS = 1,
    ERROR_CODE_CREATION_FAILED = 2,
    ERROR_CODE_INFERENCE_FAILED = 3,
    ERROR_CODE_UNKNOWN_BACKEND = 4
};

using json = nlohmann::json;

// The public C enum and the C++ rsinfer::DataType enum use different value
// orders, so they must be translated rather than cast.
static rsinfer::DataType to_rsinfer_data_type(RSInferDataType dtype)
{
    switch (dtype)
    {
    case RSInferDataType_Float32: return rsinfer::DataType::FP32;
    case RSInferDataType_Half:    return rsinfer::DataType::FP16;
    case RSInferDataType_BF16:    return rsinfer::DataType::BF16;
    case RSInferDataType_Int4:    return rsinfer::DataType::INT4;
    case RSInferDataType_FP4:     return rsinfer::DataType::FP4;
    // The public enum has no signed 8-bit type; UInt8 is the only 8-bit slot.
    case RSInferDataType_UInt8:   return rsinfer::DataType::INT8;
    default:                      return rsinfer::DataType::UNKNOWN;
    }
}

static RSInferDataType to_public_data_type(rsinfer::DataType dtype)
{
    switch (dtype)
    {
    case rsinfer::DataType::FP32: return RSInferDataType_Float32;
    case rsinfer::DataType::FP16: return RSInferDataType_Half;
    case rsinfer::DataType::BF16: return RSInferDataType_BF16;
    case rsinfer::DataType::INT8: return RSInferDataType_UInt8;
    case rsinfer::DataType::INT4: return RSInferDataType_Int4;
    case rsinfer::DataType::FP4:  return RSInferDataType_FP4;
    default:                      return RSInferDataType_Unknown;
    }
}

static RSInferDataType tensorrt_data_type_to_public(nvinfer1::DataType dtype)
{
    switch (dtype)
    {
    case nvinfer1::DataType::kFLOAT: return RSInferDataType_Float32;
    case nvinfer1::DataType::kHALF:  return RSInferDataType_Half;
    case nvinfer1::DataType::kINT32: return RSInferDataType_Int32;
    case nvinfer1::DataType::kBOOL:  return RSInferDataType_Bool;
    case nvinfer1::DataType::kUINT8: return RSInferDataType_UInt8;
    case nvinfer1::DataType::kFP8:   return RSInferDataType_FP8;
    case nvinfer1::DataType::kE8M0:  return RSInferDataType_E8M0;
    case nvinfer1::DataType::kBF16:  return RSInferDataType_BF16;
    case nvinfer1::DataType::kINT64: return RSInferDataType_Int64;
    case nvinfer1::DataType::kINT4:  return RSInferDataType_Int4;
    case nvinfer1::DataType::kFP4:   return RSInferDataType_FP4;
    // No signed 8-bit slot in the public enum; UInt8 is size-compatible.
    case nvinfer1::DataType::kINT8:  return RSInferDataType_UInt8;
    default:                         return RSInferDataType_Unknown;
    }
}

static bool public_data_type_to_tensorrt(RSInferDataType dtype, nvinfer1::DataType& result)
{
    switch (dtype)
    {
    case RSInferDataType_Float32: result = nvinfer1::DataType::kFLOAT; return true;
    case RSInferDataType_Int32:   result = nvinfer1::DataType::kINT32; return true;
    case RSInferDataType_Int64:   result = nvinfer1::DataType::kINT64; return true;
    case RSInferDataType_UInt8:   result = nvinfer1::DataType::kUINT8; return true;
    case RSInferDataType_Bool:    result = nvinfer1::DataType::kBOOL; return true;
    case RSInferDataType_Half:    result = nvinfer1::DataType::kHALF; return true;
    case RSInferDataType_FP8:     result = nvinfer1::DataType::kFP8; return true;
    case RSInferDataType_E8M0:    result = nvinfer1::DataType::kE8M0; return true;
    case RSInferDataType_BF16:    result = nvinfer1::DataType::kBF16; return true;
    case RSInferDataType_Int4:    result = nvinfer1::DataType::kINT4; return true;
    case RSInferDataType_FP4:     result = nvinfer1::DataType::kFP4; return true;
    default:                      return false;
    }
}

struct ProfileEntry
{
    std::string name;
    nvinfer1::Dims mn, opt, mx;
};

static bool to_dims(const json& j, nvinfer1::Dims& d)
{
    if (!j.is_array() || j.empty() || j.size() > nvinfer1::Dims::MAX_DIMS) return false;
    d.nbDims = static_cast<int32_t>(j.size());
    for (int32_t i = 0; i < d.nbDims; ++i)
    {
        if (!j[i].is_number_integer()) return false;
        d.d[i] = j[i].get<int64_t>();
    }
    return true;
}

static bool parse_builder_flag(const std::string& name, nvinfer1::BuilderFlag& flag)
{
    static const std::map<std::string, nvinfer1::BuilderFlag> flags{
        {"DEBUG", nvinfer1::BuilderFlag::kDEBUG},
        {"GPU_FALLBACK", nvinfer1::BuilderFlag::kGPU_FALLBACK},
        {"REFIT", nvinfer1::BuilderFlag::kREFIT},
        {"DISABLE_TIMING_CACHE", nvinfer1::BuilderFlag::kDISABLE_TIMING_CACHE},
        {"TF32", nvinfer1::BuilderFlag::kTF32},
        {"SPARSE_WEIGHTS", nvinfer1::BuilderFlag::kSPARSE_WEIGHTS},
        {"VERSION_COMPATIBLE", nvinfer1::BuilderFlag::kVERSION_COMPATIBLE},
        {"EXCLUDE_LEAN_RUNTIME", nvinfer1::BuilderFlag::kEXCLUDE_LEAN_RUNTIME},
        {"ERROR_ON_TIMING_CACHE_MISS", nvinfer1::BuilderFlag::kERROR_ON_TIMING_CACHE_MISS},
        {"DISABLE_COMPILATION_CACHE", nvinfer1::BuilderFlag::kDISABLE_COMPILATION_CACHE},
        {"STRIP_PLAN", nvinfer1::BuilderFlag::kSTRIP_PLAN},
        {"REFIT_IDENTICAL", nvinfer1::BuilderFlag::kREFIT_IDENTICAL},
        {"WEIGHT_STREAMING", nvinfer1::BuilderFlag::kWEIGHT_STREAMING},
        {"REFIT_INDIVIDUAL", nvinfer1::BuilderFlag::kREFIT_INDIVIDUAL},
        {"STRICT_NANS", nvinfer1::BuilderFlag::kSTRICT_NANS},
        {"MONITOR_MEMORY", nvinfer1::BuilderFlag::kMONITOR_MEMORY},
        {"EDITABLE_TIMING_CACHE", nvinfer1::BuilderFlag::kEDITABLE_TIMING_CACHE},
        {"DISTRIBUTIVE_INDEPENDENCE", nvinfer1::BuilderFlag::kDISTRIBUTIVE_INDEPENDENCE}
    };
    const std::string key = name.size() > 1 && name[0] == 'k' ? name.substr(1) : name;
    auto it = flags.find(key);
    if (it == flags.end()) return false;
    flag = it->second;
    return true;
}

static bool parse_memory_pool(const std::string& name, nvinfer1::MemoryPoolType& pool)
{
    static const std::map<std::string, nvinfer1::MemoryPoolType> pools{
        {"WORKSPACE", nvinfer1::MemoryPoolType::kWORKSPACE},
        {"DLA_MANAGED_SRAM", nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM},
        {"DLA_LOCAL_DRAM", nvinfer1::MemoryPoolType::kDLA_LOCAL_DRAM},
        {"DLA_GLOBAL_DRAM", nvinfer1::MemoryPoolType::kDLA_GLOBAL_DRAM},
        {"TACTIC_DRAM", nvinfer1::MemoryPoolType::kTACTIC_DRAM},
        {"TACTIC_SHARED_MEMORY", nvinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY}
    };
    auto it = pools.find(name);
    if (it == pools.end()) return false;
    pool = it->second;
    return true;
}

// Returns an empty string on success, otherwise a message describing the problem.
static std::string load_config(const std::string& json_config, rsinfer::RunnerParams& params, std::vector<ProfileEntry>& entries)
{
    json j = json::parse(json_config, nullptr, /*allow_exceptions=*/false);
    if (j.is_discarded() || !j.is_object())
        return "json_config_str is not a valid JSON object";

    if (j.contains("n_contexts"))
    {
        if (!j["n_contexts"].is_number_integer() || j["n_contexts"].get<int64_t>() < 1 ||
            static_cast<uint64_t>(j["n_contexts"].get<int64_t>()) > std::numeric_limits<uint32_t>::max())
            return "n_contexts must be an integer >= 1";
        params.nContexts = j["n_contexts"].get<uint32_t>();
    }

    if (j.contains("timing_cache"))
    {
        if (!j["timing_cache"].is_string())
            return "timing_cache must be a string";
        params.timingCacheFile = j["timing_cache"].get<std::string>();
    }

    if (j.contains("dla_core"))
    {
        if (!j["dla_core"].is_number_integer() ||
            j["dla_core"].get<int64_t>() < std::numeric_limits<int32_t>::min() ||
            j["dla_core"].get<int64_t>() > std::numeric_limits<int32_t>::max())
            return "dla_core must be an integer";
        params.dlaCore = j["dla_core"].get<int32_t>();
    }

    if (j.contains("batch_size"))
        return "batch_size is unsupported; pass the desired batch dimension in each input tensor shape";

    for (const auto& [key, target] : std::vector<std::pair<const char*, bool*>>{
        {"int8", &params.int8}, {"fp16", &params.fp16}, {"bf16", &params.bf16}})
    {
        if (j.contains(key))
        {
            if (!j[key].is_boolean()) return std::string(key) + " must be a boolean";
            *target = j[key].get<bool>();
        }
    }

    auto string_array = [&j](const char* key, std::vector<std::string>& target)
    {
        if (!j.contains(key)) return true;
        if (!j[key].is_array()) return false;
        for (const auto& item : j[key])
        {
            if (!item.is_string()) return false;
            target.push_back(item.get<std::string>());
        }
        return true;
    };
    if (!string_array("input_tensor_names", params.inputTensorNames))
        return "input_tensor_names must be an array of strings";
    if (!string_array("output_tensor_names", params.outputTensorNames))
        return "output_tensor_names must be an array of strings";

    if (j.contains("builder_flags"))
    {
        if (!j["builder_flags"].is_array()) return "builder_flags must be an array of flag names";
        for (const auto& item : j["builder_flags"])
        {
            nvinfer1::BuilderFlag flag;
            if (!item.is_string() || !parse_builder_flag(item.get<std::string>(), flag))
                return "builder_flags contains an unknown TensorRT 11 builder flag";
            params.builderFlags.push_back(flag);
        }
    }

    auto integer_option = [&j](const char* key, std::optional<int32_t>& out, int32_t minimum, int32_t maximum)
    {
        if (!j.contains(key)) return true;
        if (j[key].is_number_unsigned())
        {
            const uint64_t value = j[key].get<uint64_t>();
            if (minimum < 0 || value > static_cast<uint64_t>(maximum)) return false;
            out = static_cast<int32_t>(value);
            return true;
        }
        if (!j[key].is_number_integer()) return false;
        const int64_t value = j[key].get<int64_t>();
        if (value < minimum || value > maximum) return false;
        out = static_cast<int32_t>(value);
        return true;
    };
    if (!integer_option("builder_optimization_level", params.builderOptimizationLevel, 0, 5))
        return "builder_optimization_level must be an integer from 0 to 5";
    if (!integer_option("avg_timing_iterations", params.avgTimingIterations, 1, std::numeric_limits<int32_t>::max()))
        return "avg_timing_iterations must be a positive integer";
    if (!integer_option("max_aux_streams", params.maxAuxStreams, 0, std::numeric_limits<int32_t>::max()))
        return "max_aux_streams must be an integer >= 0";
    if (!integer_option("max_threads", params.maxThreads, 1, std::numeric_limits<int32_t>::max()))
        return "max_threads must be a positive integer";

    if (j.contains("tactic_sources"))
    {
        if ((!j["tactic_sources"].is_number_unsigned() && !j["tactic_sources"].is_number_integer()) ||
            (j["tactic_sources"].is_number_unsigned() &&
                j["tactic_sources"].get<uint64_t>() > std::numeric_limits<uint32_t>::max()) ||
            (j["tactic_sources"].is_number_integer() &&
                (j["tactic_sources"].get<int64_t>() < 0 ||
                    j["tactic_sources"].get<int64_t>() > std::numeric_limits<uint32_t>::max())))
            return "tactic_sources must be an unsigned 32-bit bitmask";
        params.tacticSources = j["tactic_sources"].get<uint32_t>();
    }

    if (j.contains("profiling_verbosity"))
    {
        if (!j["profiling_verbosity"].is_string()) return "profiling_verbosity must be a string";
        const auto value = j["profiling_verbosity"].get<std::string>();
        if (value == "LAYER_NAMES_ONLY") params.profilingVerbosity = nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY;
        else if (value == "NONE") params.profilingVerbosity = nvinfer1::ProfilingVerbosity::kNONE;
        else if (value == "DETAILED") params.profilingVerbosity = nvinfer1::ProfilingVerbosity::kDETAILED;
        else return "profiling_verbosity must be NONE, LAYER_NAMES_ONLY, or DETAILED";
    }

    if (j.contains("engine_capability"))
    {
        if (!j["engine_capability"].is_string()) return "engine_capability must be a string";
        const auto value = j["engine_capability"].get<std::string>();
        if (value == "STANDARD") params.engineCapability = nvinfer1::EngineCapability::kSTANDARD;
        else if (value == "SAFETY") params.engineCapability = nvinfer1::EngineCapability::kSAFETY;
        else if (value == "DLA_STANDALONE") params.engineCapability = nvinfer1::EngineCapability::kDLA_STANDALONE;
        else return "engine_capability must be STANDARD, SAFETY, or DLA_STANDALONE";
    }

    if (j.contains("hardware_compatibility_level"))
    {
        if (!j["hardware_compatibility_level"].is_string()) return "hardware_compatibility_level must be a string";
        const auto value = j["hardware_compatibility_level"].get<std::string>();
        if (value == "NONE") params.hardwareCompatibilityLevel = nvinfer1::HardwareCompatibilityLevel::kNONE;
        else if (value == "AMPERE_PLUS") params.hardwareCompatibilityLevel = nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS;
        else if (value == "SAME_COMPUTE_CAPABILITY") params.hardwareCompatibilityLevel = nvinfer1::HardwareCompatibilityLevel::kSAME_COMPUTE_CAPABILITY;
        else return "hardware_compatibility_level must be NONE, AMPERE_PLUS, or SAME_COMPUTE_CAPABILITY";
    }

    if (j.contains("default_device_type"))
    {
        if (!j["default_device_type"].is_string()) return "default_device_type must be GPU or DLA";
        const auto value = j["default_device_type"].get<std::string>();
        if (value == "GPU") params.defaultDeviceType = nvinfer1::DeviceType::kGPU;
        else if (value == "DLA") params.defaultDeviceType = nvinfer1::DeviceType::kDLA;
        else return "default_device_type must be GPU or DLA";
    }

    if (j.contains("layer_device_types"))
    {
        if (!j["layer_device_types"].is_object()) return "layer_device_types must map layer names to GPU or DLA";
        for (auto it = j["layer_device_types"].begin(); it != j["layer_device_types"].end(); ++it)
        {
            if (!it.value().is_string()) return "layer_device_types values must be GPU or DLA";
            const auto value = it.value().get<std::string>();
            if (value == "GPU") params.layerDeviceTypes[it.key()] = nvinfer1::DeviceType::kGPU;
            else if (value == "DLA") params.layerDeviceTypes[it.key()] = nvinfer1::DeviceType::kDLA;
            else return "layer_device_types values must be GPU or DLA";
        }
    }

    if (j.contains("runtime_platform"))
    {
        if (!j["runtime_platform"].is_string()) return "runtime_platform must be SAME_AS_BUILD or WINDOWS_AMD64";
        const auto value = j["runtime_platform"].get<std::string>();
        if (value == "SAME_AS_BUILD") params.runtimePlatform = nvinfer1::RuntimePlatform::kSAME_AS_BUILD;
        else if (value == "WINDOWS_AMD64") params.runtimePlatform = nvinfer1::RuntimePlatform::kWINDOWS_AMD64;
        else return "runtime_platform must be SAME_AS_BUILD or WINDOWS_AMD64";
    }

    if (j.contains("tiling_optimization_level"))
    {
        if (!j["tiling_optimization_level"].is_string()) return "tiling_optimization_level must be a string";
        const auto value = j["tiling_optimization_level"].get<std::string>();
        if (value == "NONE") params.tilingOptimizationLevel = nvinfer1::TilingOptimizationLevel::kNONE;
        else if (value == "FAST") params.tilingOptimizationLevel = nvinfer1::TilingOptimizationLevel::kFAST;
        else if (value == "MODERATE") params.tilingOptimizationLevel = nvinfer1::TilingOptimizationLevel::kMODERATE;
        else if (value == "FULL") params.tilingOptimizationLevel = nvinfer1::TilingOptimizationLevel::kFULL;
        else return "tiling_optimization_level must be NONE, FAST, MODERATE, or FULL";
    }

    if (j.contains("tiling_l2_limit"))
    {
        if (!j["tiling_l2_limit"].is_number_integer() || j["tiling_l2_limit"].get<int64_t>() < 0)
            return "tiling_l2_limit must be a non-negative byte count";
        params.tilingL2Limit = j["tiling_l2_limit"].get<int64_t>();
    }

    if (j.contains("preview_features"))
    {
        if (!j["preview_features"].is_object()) return "preview_features must be an object mapping feature names to booleans";
        for (auto it = j["preview_features"].begin(); it != j["preview_features"].end(); ++it)
        {
            if (!it.value().is_boolean()) return "preview_features values must be booleans";
            nvinfer1::PreviewFeature feature;
            if (it.key() == "ALIASED_PLUGIN_IO_10_03") feature = nvinfer1::PreviewFeature::kALIASED_PLUGIN_IO_10_03;
            else if (it.key() == "RUNTIME_ACTIVATION_RESIZE_10_10") feature = nvinfer1::PreviewFeature::kRUNTIME_ACTIVATION_RESIZE_10_10;
            else return "preview_features contains an unknown TensorRT 11 preview feature";
            params.previewFeatures.emplace_back(feature, it.value().get<bool>());
        }
    }

    if (!string_array("plugins_to_serialize", params.pluginsToSerialize))
        return "plugins_to_serialize must be an array of strings";

    if (j.contains("allow_gpu_fallback"))
    {
        if (!j["allow_gpu_fallback"].is_boolean()) return "allow_gpu_fallback must be a boolean";
        params.allowGpuFallback = j["allow_gpu_fallback"].get<bool>();
    }

    if (j.contains("workspace_size"))
    {
        if (!j["workspace_size"].is_number_unsigned() &&
            (!j["workspace_size"].is_number_integer() || j["workspace_size"].get<int64_t>() < 0))
            return "workspace_size must be a non-negative byte count";
        if (j["workspace_size"].get<uint64_t>() > std::numeric_limits<size_t>::max())
            return "workspace_size exceeds the platform size limit";
        params.memoryPoolLimits[nvinfer1::MemoryPoolType::kWORKSPACE] = j["workspace_size"].get<size_t>();
    }

    if (j.contains("memory_pool_limits"))
    {
        if (!j["memory_pool_limits"].is_object()) return "memory_pool_limits must be an object of pool names to byte counts";
        for (auto it = j["memory_pool_limits"].begin(); it != j["memory_pool_limits"].end(); ++it)
        {
            nvinfer1::MemoryPoolType pool;
            if (!parse_memory_pool(it.key(), pool)) return "memory_pool_limits contains an unknown pool name";
            if (!it.value().is_number_unsigned() &&
                (!it.value().is_number_integer() || it.value().get<int64_t>() < 0))
                return "memory_pool_limits values must be non-negative byte counts";
            if (it.value().get<uint64_t>() > std::numeric_limits<size_t>::max())
                return "memory_pool_limits value exceeds the platform size limit";
            params.memoryPoolLimits[pool] = it.value().get<size_t>();
        }
    }

    if (j.contains("engine_file"))
    {
        if (!j["engine_file"].is_string()) return "engine_file must be a string";
        params.engineFile = j["engine_file"].get<std::string>();
    }

    if (j.contains("data_dirs"))
    {
        if (!j["data_dirs"].is_array())
            return "data_dirs must be an array of strings";
        for (const auto& d : j["data_dirs"])
        {
            if (!d.is_string())
                return "data_dirs must be an array of strings";
            params.dataDirs.push_back(d.get<std::string>());
        }
    }

    if (j.contains("profile"))
    {
        if (!j["profile"].is_object())
            return "profile must be an object";
        for (auto it = j["profile"].begin(); it != j["profile"].end(); ++it)
        {
            ProfileEntry e;
            e.name = it.key();
            const json& v = it.value();
            if (!v.is_object() || !v.contains("min") || !v.contains("opt") || !v.contains("max") ||
                !to_dims(v["min"], e.mn) || !to_dims(v["opt"], e.opt) || !to_dims(v["max"], e.mx))
                return "Bad profile entry for tensor: " + e.name;
            entries.push_back(std::move(e));
        }
    }

    params.dynamic = !entries.empty();
    return {};
}

RSInferenceRunner* rs_inference_runner_create(const char* model_path, const char* json_config_str, RSInferError* error)
{
    if (!model_path)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Model path cannot be null");
        }
        return nullptr;
    }

    // Declared outside the try so the catch block can free them.
    RSInferenceRunner* runner = nullptr;
    rsinfer::TensorRTRunner* tensorrtrunner = nullptr;

    try
    {
        rsinfer::RunnerParams params;
        params.onnxFileName = model_path;
        // `locateFile()` requires at least one directory entry. An empty entry
        // means "use onnxFileName exactly as given", matching the C++ sample.
        params.dataDirs = { "" };

        std::vector<ProfileEntry> entries;
        if (json_config_str && *json_config_str)
        {
            std::string msg = load_config(json_config_str, params, entries);
            if (!msg.empty())
            {
                if (error)
                {
                    error->code = ERROR_CODE_INVALID_ARGUMENTS;
                    error->message = strdup(msg.c_str());
                }
                return nullptr;
            }
        }

        runner = (RSInferenceRunner*)malloc(sizeof(RSInferenceRunner));
        if (!runner)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferenceRunner");
            }
            return nullptr;
        }

        tensorrtrunner = new rsinfer::TensorRTRunner(params);
        runner->runner = static_cast<void*>(tensorrtrunner);
        runner->backend = RSInferenceBackend_TensorRT;

        std::function<bool(nvinfer1::IOptimizationProfile*)> profile_cb;
        if (!entries.empty())
        {
            profile_cb = [&entries](nvinfer1::IOptimizationProfile* p)
            {
                for (const auto& e : entries)
                {
                    if (!p->setDimensions(e.name.c_str(), nvinfer1::OptProfileSelector::kMIN, e.mn) ||
                        !p->setDimensions(e.name.c_str(), nvinfer1::OptProfileSelector::kOPT, e.opt) ||
                        !p->setDimensions(e.name.c_str(), nvinfer1::OptProfileSelector::kMAX, e.mx))
                        return false;
                }
                return true;
            };
        }

        if (!tensorrtrunner->build(profile_cb))
        {
            delete tensorrtrunner;
            free(runner);
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to build the TensorRT engine");
            }
            return nullptr;
        }

        return runner;
    }
    catch (const std::exception& e)
    {
        delete tensorrtrunner; // deleting nullptr is a no-op
        free(runner);
        if (error)
        {
            error->code = ERROR_CODE_CREATION_FAILED;
            error->message = strdup(("Failed to create RSInferenceRunner: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

void rs_inference_runner_destroy(RSInferenceRunner* runner)
{
    if (runner)
    {
        if (runner->runner)
        {
            auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
            delete tensorrtrunner;
        }
        free(runner);
    }
}

int rs_inference_runner_infer(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error)
{
    if (!runner || !input_tensors)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner or input tensors cannot be null");
        }
        return ERROR_CODE_INVALID_ARGUMENTS;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<rsinfer::Tensor> inputs;

        for (size_t i = 0; i < input_tensors->length; ++i)
        {
            RSInferTensor* tensor = input_tensors->tensors[i];
            // The Rust side is allowed to hand us a tensor whose payload is
            // missing; report it instead of dereferencing null below.
            if (!tensor || !tensor->shape || !tensor->data ||
                (!tensor->shape->data && tensor->shape->length > 0) ||
                (!tensor->data->data && tensor->data->length > 0))
            {
                if (error)
                {
                    error->code = ERROR_CODE_INVALID_ARGUMENTS;
                    error->message = strdup("Input tensor is missing data or shape");
                }
                return ERROR_CODE_INVALID_ARGUMENTS;
            }
            rsinfer::Tensor t;
            t.name = tensor->name;
            t.shape.assign(tensor->shape->data, tensor->shape->data + tensor->shape->length);
            t.data.assign(tensor->data->data, tensor->data->data + tensor->data->length);
            t.dtype = to_rsinfer_data_type(tensor->dtype);
            inputs.push_back(std::move(t));
        }

        tensorrtrunner->process(inputs);

        return ERROR_CODE_SUCCESS;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Inference failed: " + std::string(e.what())).c_str());
        }
        return ERROR_CODE_INFERENCE_FAILED;
    }
}

int rs_inference_runner_infer_async(RSInferenceRunner* runner, RSInferTensorArray* input_tensors, RSInferError* error)
{
    if (!runner || !input_tensors)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner or input tensors cannot be null");
        }
        return ERROR_CODE_INVALID_ARGUMENTS;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<rsinfer::Tensor> inputs;

        for (size_t i = 0; i < input_tensors->length; ++i)
        {
            RSInferTensor* tensor = input_tensors->tensors[i];
            // The Rust side is allowed to hand us a tensor whose payload is
            // missing; report it instead of dereferencing null below.
            if (!tensor || !tensor->shape || !tensor->data ||
                (!tensor->shape->data && tensor->shape->length > 0) ||
                (!tensor->data->data && tensor->data->length > 0))
            {
                if (error)
                {
                    error->code = ERROR_CODE_INVALID_ARGUMENTS;
                    error->message = strdup("Input tensor is missing data or shape");
                }
                return ERROR_CODE_INVALID_ARGUMENTS;
            }
            rsinfer::Tensor t;
            t.name = tensor->name;
            t.shape.assign(tensor->shape->data, tensor->shape->data + tensor->shape->length);
            t.data.assign(tensor->data->data, tensor->data->data + tensor->data->length);
            t.dtype = to_rsinfer_data_type(tensor->dtype);
            inputs.push_back(std::move(t));
        }

        tensorrtrunner->process_async(inputs);

        return ERROR_CODE_SUCCESS;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Asynchronous inference failed: " + std::string(e.what())).c_str());
        }
        return ERROR_CODE_INFERENCE_FAILED;
    }
}

RSInferTensorArray* rs_inference_runner_get_results(RSInferenceRunner* runner, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<rsinfer::Tensor> outputs;
        tensorrtrunner->get(outputs);

        RSInferTensorArray* output_array = rs_infer_tensorarray_create(outputs.size());
        if (!output_array)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferTensorArray");
            }
            return nullptr;
        }

        for (size_t i = 0; i < outputs.size(); ++i)
        {
            const auto& tensor = outputs[i];
            RSInferUCharArray* data_array = rs_infer_uchararray_create(tensor.data.size());
            if (!data_array)
            {
                if (error)
                {
                    error->code = ERROR_CODE_CREATION_FAILED;
                    error->message = strdup("Failed to allocate memory for RSInferUCharArray");
                }
                rs_infer_tensorarray_destroy(output_array);
                return nullptr;
            }
            memcpy(data_array->data, tensor.data.data(), tensor.data.size());
            RSInferLLArray* shape_array = rs_infer_llarray_create(tensor.shape.size());
            if (!shape_array)
            {
                if (error)
                {
                    error->code = ERROR_CODE_CREATION_FAILED;
                    error->message = strdup("Failed to allocate memory for RSInferLLArray");
                }
                rs_infer_uchararray_destroy(data_array);
                rs_infer_tensorarray_destroy(output_array);
                return nullptr;
            }
            for (size_t d = 0; d < tensor.shape.size(); ++d)
                shape_array->data[d] = static_cast<long long>(tensor.shape[d]);
            output_array->tensors[i] = rs_infer_tensor_create(tensor.name.c_str(), tensor.name.size(), to_public_data_type(tensor.dtype), data_array, shape_array);
            if (!output_array->tensors[i])
            {
                // `rs_infer_tensor_create` did not take ownership on failure.
                if (error)
                {
                    error->code = ERROR_CODE_CREATION_FAILED;
                    error->message = strdup("Failed to allocate memory for RSInferTensor");
                }
                rs_infer_uchararray_destroy(data_array);
                rs_infer_llarray_destroy(shape_array);
                rs_infer_tensorarray_destroy(output_array);
                return nullptr;
            }
        }

        return output_array;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get inference results: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

RSInferIntArray* rs_inference_runner_get_input_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<int> shape = tensorrtrunner->get_input_shape(index);

        RSInferIntArray* shape_array = rs_infer_intarray_create(shape.size());
        if (!shape_array)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferIntArray");
            }
            return nullptr;
        }
        memcpy(shape_array->data, shape.data(), shape.size() * sizeof(int));

        return shape_array;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get input shape: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

RSInferIntArray* rs_inference_runner_get_output_shape(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::vector<int> shape = tensorrtrunner->get_output_shape(index);

        RSInferIntArray* shape_array = rs_infer_intarray_create(shape.size());
        if (!shape_array)
        {
            if (error)
            {
                error->code = ERROR_CODE_CREATION_FAILED;
                error->message = strdup("Failed to allocate memory for RSInferIntArray");
            }
            return nullptr;
        }
        memcpy(shape_array->data, shape.data(), shape.size() * sizeof(int));

        return shape_array;
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get output shape: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

RSInferDataType rs_inference_runner_get_tensor_data_type(RSInferenceRunner* runner, const char* tensor_name, RSInferError* error)
{
    if (!runner || !tensor_name)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner or tensor name cannot be null");
        }
        return RSInferDataType_Unknown;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        nvinfer1::DataType data_type = tensorrtrunner->get_tensor_data_type(tensor_name);
        return tensorrt_data_type_to_public(data_type);
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get tensor data type: " + std::string(e.what())).c_str());
        }
        return RSInferDataType_Unknown;
    }
}

size_t rs_inference_runner_get_tensor_data_size(RSInferenceRunner* runner, RSInferDataType data_type, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return 0;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        nvinfer1::DataType trt_data_type;
        if (!public_data_type_to_tensorrt(data_type, trt_data_type))
        {
            if (error)
            {
                error->code = ERROR_CODE_INVALID_ARGUMENTS;
                error->message = strdup("Unsupported tensor data type");
            }
            return 0;
        }
        return tensorrtrunner->get_tensor_data_size(trt_data_type);
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get tensor data size: " + std::string(e.what())).c_str());
        }
        return 0;
    }
}

const char* rs_inference_runner_get_input_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::string name = tensorrtrunner->get_input_tensor_name(index);
        return strdup(name.c_str());
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get input tensor name: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}

const char* rs_inference_runner_get_output_tensor_name(RSInferenceRunner* runner, unsigned index, RSInferError* error)
{
    if (!runner)
    {
        if (error)
        {
            error->code = ERROR_CODE_INVALID_ARGUMENTS;
            error->message = strdup("Runner cannot be null");
        }
        return nullptr;
    }

    try
    {
        auto tensorrtrunner = static_cast<rsinfer::TensorRTRunner*>(runner->runner);
        std::string name = tensorrtrunner->get_output_tensor_name(index);
        return strdup(name.c_str());
    }
    catch (const std::exception& e)
    {
        if (error)
        {
            error->code = ERROR_CODE_INFERENCE_FAILED;
            error->message = strdup(("Failed to get output tensor name: " + std::string(e.what())).c_str());
        }
        return nullptr;
    }
}
