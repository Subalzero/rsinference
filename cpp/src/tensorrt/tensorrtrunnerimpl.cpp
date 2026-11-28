#include "rsinference.h"
#include "tensorrtrunner.h"
#include "common/datatypeconversion.h"

#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
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

static const std::map<std::string, nvinfer1::BuilderFlag>& builder_flag_names()
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
    return flags;
}

static bool parse_builder_flag(const std::string& name, nvinfer1::BuilderFlag& flag)
{
    const auto& flags = builder_flag_names();
    const std::string key = name.size() > 1 && name[0] == 'k' ? name.substr(1) : name;
    auto it = flags.find(key);
    if (it == flags.end()) return false;
    flag = it->second;
    return true;
}

static const std::map<std::string, nvinfer1::MemoryPoolType>& memory_pool_names()
{
    static const std::map<std::string, nvinfer1::MemoryPoolType> pools{
        {"WORKSPACE", nvinfer1::MemoryPoolType::kWORKSPACE},
        {"DLA_MANAGED_SRAM", nvinfer1::MemoryPoolType::kDLA_MANAGED_SRAM},
        {"DLA_LOCAL_DRAM", nvinfer1::MemoryPoolType::kDLA_LOCAL_DRAM},
        {"DLA_GLOBAL_DRAM", nvinfer1::MemoryPoolType::kDLA_GLOBAL_DRAM},
        {"TACTIC_DRAM", nvinfer1::MemoryPoolType::kTACTIC_DRAM},
        {"TACTIC_SHARED_MEMORY", nvinfer1::MemoryPoolType::kTACTIC_SHARED_MEMORY}
    };
    return pools;
}

static bool parse_memory_pool(const std::string& name, nvinfer1::MemoryPoolType& pool)
{
    const auto& pools = memory_pool_names();
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
            params.profile[e.name] = { e.mn, e.opt, e.mx };
            entries.push_back(std::move(e));
        }
    }

    params.dynamic = !entries.empty();
    return {};
}

template <typename Map, typename Value>
static const std::string* name_of(const Map& names, Value value)
{
    for (const auto& [name, v] : names)
        if (v == value) return &name;
    return nullptr;
}

static json dims_to_json(const nvinfer1::Dims& d)
{
    json arr = json::array();
    for (int32_t i = 0; i < d.nbDims; ++i) arr.push_back(d.d[i]);
    return arr;
}

//! Inverse of load_config: emits the same keys so the result can be fed back to
//! rs_inference_runner_create. Unset optional settings are omitted.
static json config_to_json(const rsinfer::RunnerParams& params)
{
    json j = json::object();
    j["n_contexts"] = params.nContexts;
    if (!params.timingCacheFile.empty()) j["timing_cache"] = params.timingCacheFile;
    j["dla_core"] = params.dlaCore;
    j["int8"] = params.int8;
    j["fp16"] = params.fp16;
    j["bf16"] = params.bf16;
    if (!params.inputTensorNames.empty()) j["input_tensor_names"] = params.inputTensorNames;
    if (!params.outputTensorNames.empty()) j["output_tensor_names"] = params.outputTensorNames;

    if (!params.builderFlags.empty())
    {
        json flags = json::array();
        for (auto flag : params.builderFlags)
            if (const auto* name = name_of(builder_flag_names(), flag)) flags.push_back(*name);
        j["builder_flags"] = std::move(flags);
    }

    if (params.builderOptimizationLevel) j["builder_optimization_level"] = *params.builderOptimizationLevel;
    if (params.avgTimingIterations) j["avg_timing_iterations"] = *params.avgTimingIterations;
    if (params.maxAuxStreams) j["max_aux_streams"] = *params.maxAuxStreams;
    if (params.maxThreads) j["max_threads"] = *params.maxThreads;
    if (params.tacticSources) j["tactic_sources"] = *params.tacticSources;

    if (params.profilingVerbosity)
    {
        switch (*params.profilingVerbosity)
        {
        case nvinfer1::ProfilingVerbosity::kLAYER_NAMES_ONLY: j["profiling_verbosity"] = "LAYER_NAMES_ONLY"; break;
        case nvinfer1::ProfilingVerbosity::kNONE: j["profiling_verbosity"] = "NONE"; break;
        case nvinfer1::ProfilingVerbosity::kDETAILED: j["profiling_verbosity"] = "DETAILED"; break;
        default: break;
        }
    }
    if (params.engineCapability)
    {
        switch (*params.engineCapability)
        {
        case nvinfer1::EngineCapability::kSTANDARD: j["engine_capability"] = "STANDARD"; break;
        case nvinfer1::EngineCapability::kSAFETY: j["engine_capability"] = "SAFETY"; break;
        case nvinfer1::EngineCapability::kDLA_STANDALONE: j["engine_capability"] = "DLA_STANDALONE"; break;
        default: break;
        }
    }
    if (params.hardwareCompatibilityLevel)
    {
        switch (*params.hardwareCompatibilityLevel)
        {
        case nvinfer1::HardwareCompatibilityLevel::kNONE: j["hardware_compatibility_level"] = "NONE"; break;
        case nvinfer1::HardwareCompatibilityLevel::kAMPERE_PLUS: j["hardware_compatibility_level"] = "AMPERE_PLUS"; break;
        case nvinfer1::HardwareCompatibilityLevel::kSAME_COMPUTE_CAPABILITY: j["hardware_compatibility_level"] = "SAME_COMPUTE_CAPABILITY"; break;
        default: break;
        }
    }

    auto device_name = [](nvinfer1::DeviceType type) { return type == nvinfer1::DeviceType::kDLA ? "DLA" : "GPU"; };
    if (params.defaultDeviceType) j["default_device_type"] = device_name(*params.defaultDeviceType);
    if (!params.layerDeviceTypes.empty())
    {
        json layers = json::object();
        for (const auto& [layer, type] : params.layerDeviceTypes) layers[layer] = device_name(type);
        j["layer_device_types"] = std::move(layers);
    }

    if (params.runtimePlatform)
    {
        switch (*params.runtimePlatform)
        {
        case nvinfer1::RuntimePlatform::kSAME_AS_BUILD: j["runtime_platform"] = "SAME_AS_BUILD"; break;
        case nvinfer1::RuntimePlatform::kWINDOWS_AMD64: j["runtime_platform"] = "WINDOWS_AMD64"; break;
        default: break;
        }
    }
    if (params.tilingOptimizationLevel)
    {
        switch (*params.tilingOptimizationLevel)
        {
        case nvinfer1::TilingOptimizationLevel::kNONE: j["tiling_optimization_level"] = "NONE"; break;
        case nvinfer1::TilingOptimizationLevel::kFAST: j["tiling_optimization_level"] = "FAST"; break;
        case nvinfer1::TilingOptimizationLevel::kMODERATE: j["tiling_optimization_level"] = "MODERATE"; break;
        case nvinfer1::TilingOptimizationLevel::kFULL: j["tiling_optimization_level"] = "FULL"; break;
        default: break;
        }
    }
    if (params.tilingL2Limit) j["tiling_l2_limit"] = *params.tilingL2Limit;

    if (!params.previewFeatures.empty())
    {
        json features = json::object();
        for (const auto& [feature, enabled] : params.previewFeatures)
        {
            if (feature == nvinfer1::PreviewFeature::kALIASED_PLUGIN_IO_10_03) features["ALIASED_PLUGIN_IO_10_03"] = enabled;
            else if (feature == nvinfer1::PreviewFeature::kRUNTIME_ACTIVATION_RESIZE_10_10) features["RUNTIME_ACTIVATION_RESIZE_10_10"] = enabled;
        }
        j["preview_features"] = std::move(features);
    }

    if (!params.pluginsToSerialize.empty()) j["plugins_to_serialize"] = params.pluginsToSerialize;
    j["allow_gpu_fallback"] = params.allowGpuFallback;

    // workspace_size is reported through memory_pool_limits.WORKSPACE.
    if (!params.memoryPoolLimits.empty())
    {
        json pools = json::object();
        for (const auto& [pool, limit] : params.memoryPoolLimits)
            if (const auto* name = name_of(memory_pool_names(), pool)) pools[*name] = limit;
        j["memory_pool_limits"] = std::move(pools);
    }

    if (!params.engineFile.empty()) j["engine_file"] = params.engineFile;

    // rs_inference_runner_create always seeds dataDirs with "" (use the model
    // path as given); only report the directories the caller added.
    json dirs = json::array();
    for (const auto& dir : params.dataDirs)
        if (!dir.empty()) dirs.push_back(dir);
    if (!dirs.empty()) j["data_dirs"] = std::move(dirs);

    if (!params.profile.empty())
    {
        json profile = json::object();
        for (const auto& [name, dims] : params.profile)
            profile[name] = { {"min", dims_to_json(dims.min)}, {"opt", dims_to_json(dims.opt)}, {"max", dims_to_json(dims.max)} };
        j["profile"] = std::move(profile);
    }
    return j;
}

static void set_error(RSInferError* error, int code, const std::string& message)
{
    if (!error) return;
    error->code = code;
    error->message = strdup(message.c_str());
}

//! Converts tensors to a caller-owned RSInferTensorArray. Returns nullptr and sets
//! `error` on allocation failure.
static RSInferTensorArray* make_tensor_array(const std::vector<rsinfer::Tensor>& tensors, RSInferError* error)
{
    RSInferTensorArray* array = rs_infer_tensorarray_create(tensors.size());
    if (!array)
    {
        set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate memory for RSInferTensorArray");
        return nullptr;
    }

    for (size_t i = 0; i < tensors.size(); ++i)
    {
        const auto& tensor = tensors[i];
        RSInferUCharArray* data_array = rs_infer_uchararray_create(tensor.data.size());
        if (!data_array)
        {
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate memory for RSInferUCharArray");
            rs_infer_tensorarray_destroy(array);
            return nullptr;
        }
        if (!tensor.data.empty())
            memcpy(data_array->data, tensor.data.data(), tensor.data.size());
        RSInferLLArray* shape_array = rs_infer_llarray_create(tensor.shape.size());
        if (!shape_array)
        {
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate memory for RSInferLLArray");
            rs_infer_uchararray_destroy(data_array);
            rs_infer_tensorarray_destroy(array);
            return nullptr;
        }
        for (size_t d = 0; d < tensor.shape.size(); ++d)
            shape_array->data[d] = static_cast<long long>(tensor.shape[d]);
        array->tensors[i] = rs_infer_tensor_create(tensor.name.c_str(), tensor.name.size(), rsinfer::to_public_data_type(tensor.dtype), data_array, shape_array);
        if (!array->tensors[i])
        {
            // `rs_infer_tensor_create` did not take ownership on failure.
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate memory for RSInferTensor");
            rs_infer_uchararray_destroy(data_array);
            rs_infer_llarray_destroy(shape_array);
            rs_infer_tensorarray_destroy(array);
            return nullptr;
        }
    }
    return array;
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
            t.dtype = rsinfer::from_public_data_type(tensor->dtype);
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
            t.dtype = rsinfer::from_public_data_type(tensor->dtype);
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
        return make_tensor_array(outputs, error);
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

static rsinfer::TensorRTRunner* get_tensorrt_runner(RSInferenceRunner* runner)
{
    if (!runner || !runner->runner || runner->backend != RSInferenceBackend_TensorRT)
        throw std::invalid_argument("Runner is null or is not a TensorRT runner");
    return static_cast<rsinfer::TensorRTRunner*>(runner->runner);
}

RSInferTensorArray* rs_inference_runner_get_input_tensors_info(RSInferenceRunner* runner, RSInferError* error)
{
    try
    {
        const auto infos = get_tensorrt_runner(runner)->get_input_tensors_info();
        if (!infos.has_value())
        {
            set_error(error, ERROR_CODE_INFERENCE_FAILED, "Failed to get TensorRT input tensors info: " + infos.error());
            return nullptr;
        }
        // Info tensors carry name, dtype and shape only; their data arrays are empty.
        // Dynamic dimensions are reported as -1.
        return make_tensor_array(infos.value(), error);
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return nullptr;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get TensorRT input tensors info: ") + e.what());
        return nullptr;
    }
}

RSInferTensorArray* rs_inference_runner_get_output_tensors_info(RSInferenceRunner* runner, RSInferError* error)
{
    try
    {
        const auto infos = get_tensorrt_runner(runner)->get_output_tensors_info();
        if (!infos.has_value())
        {
            set_error(error, ERROR_CODE_INFERENCE_FAILED, "Failed to get TensorRT output tensors info: " + infos.error());
            return nullptr;
        }
        // Info tensors carry name, dtype and shape only; their data arrays are empty.
        // Dynamic dimensions are reported as -1.
        return make_tensor_array(infos.value(), error);
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return nullptr;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get TensorRT output tensors info: ") + e.what());
        return nullptr;
    }
}

char* rs_inference_runner_get_json_config(RSInferenceRunner* runner, RSInferError* error)
{
    try
    {
        char* result = strdup(config_to_json(get_tensorrt_runner(runner)->get_config()).dump().c_str());
        if (!result)
            set_error(error, ERROR_CODE_CREATION_FAILED, "Failed to allocate TensorRT JSON config string");
        return result; // freed by the caller with rs_infer_string_free
    }
    catch (const std::invalid_argument& e)
    {
        set_error(error, ERROR_CODE_INVALID_ARGUMENTS, e.what());
        return nullptr;
    }
    catch (const std::exception& e)
    {
        set_error(error, ERROR_CODE_INFERENCE_FAILED, std::string("Failed to get TensorRT JSON config: ") + e.what());
        return nullptr;
    }
}
