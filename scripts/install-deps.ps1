[CmdletBinding()]
param(
    [Parameter(Position = 0, ValueFromRemainingArguments)]
    [ValidateSet('cuda', 'tensorrt', 'openvino', 'toolchain')]
    [string[]] $Component = @('toolchain', 'cuda', 'tensorrt', 'openvino'),

    # Where TensorRT and OpenVINO are extracted.
    [string] $DepsDir = $(if ($env:DEPS_DIR) { $env:DEPS_DIR } else { 'C:\deps' }),

    [string] $CudaVersion = $(if ($env:CUDA_VERSION) { $env:CUDA_VERSION } else { '13.2.2' }),
    [string] $TensorRtVersion = $(if ($env:TENSORRT_VERSION) { $env:TENSORRT_VERSION } else { '11.0.0.114' }),
    [string] $OpenVinoVersion = $(if ($env:OPENVINO_VERSION) { $env:OPENVINO_VERSION } else { '2026.4.1' }),
    [string] $OpenVinoBuild = $(if ($env:OPENVINO_BUILD) { $env:OPENVINO_BUILD } else { '22982.07f9c262b05' })
)

<#
.SYNOPSIS
Download and install the native SDKs needed to build the C++ bridge on Windows.

.DESCRIPTION
Components: cuda, tensorrt, openvino, toolchain (default: all).
  cuda       CUDA Toolkit (nvcc + cudart) via the NVIDIA network installer (needs admin)
  tensorrt   TensorRT zip from developer.download.nvidia.com
  openvino   OpenVINO runtime zip from storage.openvinotoolkit.org
  toolchain  CMake, Ninja, LLVM (libclang for bindgen) via winget/choco when missing

Prints, and appends to $env:GITHUB_ENV when set, CUDA_TOOLKIT_ROOT_DIR, TENSORRT_DIR,
OPENVINO_DIR and LIBCLANG_PATH. Visual Studio Build Tools (MSVC) are not installed.
#>

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Set-BuildVar([string] $Name, [string] $Value) {
    Write-Host "$Name=$Value"
    Set-Item -Path "Env:$Name" -Value $Value
    if ($env:GITHUB_ENV) { Add-Content -LiteralPath $env:GITHUB_ENV -Value "$Name=$Value" }
}

function Get-Download([string] $Url, [string] $Destination) {
    Write-Host "Downloading $Url"
    # curl.exe with retries copes with the multi-GB archives better than Invoke-WebRequest.
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -fL --retry 5 --retry-delay 10 --retry-all-errors --connect-timeout 30 -o $Destination $Url
        if ($LASTEXITCODE -ne 0) { throw "Download failed with exit code ${LASTEXITCODE}: $Url" }
    }
    else {
        Invoke-WebRequest -Uri $Url -OutFile $Destination -UseBasicParsing
    }
}

function Expand-Into([string] $Archive, [string] $Destination) {
    New-Item -ItemType Directory -Force -Path $Destination | Out-Null
    Expand-Archive -LiteralPath $Archive -DestinationPath $Destination -Force
}

function Install-Tool([string] $Command, [string] $WingetId, [string] $ChocoId) {
    if (Get-Command $Command -ErrorAction SilentlyContinue) { return }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --id $WingetId -e --accept-package-agreements --accept-source-agreements
    }
    elseif (Get-Command choco -ErrorAction SilentlyContinue) {
        & choco install $ChocoId -y
    }
    else {
        throw "$Command is missing and neither winget nor choco is available to install it."
    }
    if ($LASTEXITCODE -ne 0) { throw "Installing $Command failed with exit code $LASTEXITCODE." }
}

New-Item -ItemType Directory -Force -Path $DepsDir | Out-Null
$DepsDir = (Resolve-Path $DepsDir).Path
$tmp = Join-Path ([IO.Path]::GetTempPath()) ([Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null

try {
    if ($Component -contains 'toolchain') {
        Install-Tool 'cmake' 'Kitware.CMake' 'cmake'
        Install-Tool 'ninja' 'Ninja-build.Ninja' 'ninja'
        $llvmBin = 'C:\Program Files\LLVM\bin'
        if (-not (Test-Path (Join-Path $llvmBin 'libclang.dll'))) {
            Install-Tool 'clang' 'LLVM.LLVM' 'llvm'
        }
        if (-not (Test-Path (Join-Path $llvmBin 'libclang.dll'))) {
            throw "libclang.dll not found under '$llvmBin' after installing LLVM."
        }
        Set-BuildVar 'LIBCLANG_PATH' $llvmBin
    }

    if ($Component -contains 'cuda') {
        $short = ($CudaVersion -split '\.')[0..1] -join '.'
        $root = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$short"
        if (-not (Test-Path (Join-Path $root 'include\cuda.h'))) {
            $installer = Join-Path $tmp 'cuda_network.exe'
            Get-Download "https://developer.download.nvidia.com/compute/cuda/$CudaVersion/network_installers/cuda_${CudaVersion}_windows_network.exe" $installer
            # Compiler and runtime only; skips the driver and Visual Studio integration.
            $proc = Start-Process -FilePath $installer -ArgumentList '-s', '-n', "nvcc_$short", "cudart_$short", "crt_$short" -Wait -PassThru
            if ($proc.ExitCode -ne 0) { throw "CUDA installer failed with exit code $($proc.ExitCode)." }
        }
        if (-not (Test-Path (Join-Path $root 'include\cuda.h'))) {
            throw "CUDA headers not found under '$root' after installation."
        }
        Set-BuildVar 'CUDA_TOOLKIT_ROOT_DIR' $root
        $env:PATH = "$root\bin;$env:PATH"
        if ($env:GITHUB_PATH) { Add-Content -LiteralPath $env:GITHUB_PATH -Value "$root\bin" }
    }

    if ($Component -contains 'tensorrt') {
        $cudaMajorMinor = if ($env:CUDA_TOOLKIT_ROOT_DIR) { (Split-Path $env:CUDA_TOOLKIT_ROOT_DIR -Leaf).TrimStart('v') } else { ($CudaVersion -split '\.')[0..1] -join '.' }
        $trtRoot = Join-Path $DepsDir 'TensorRT'
        if (-not (Test-Path (Join-Path $trtRoot 'include\NvInferVersion.h'))) {
            $zip = Join-Path $tmp 'tensorrt.zip'
            Get-Download "https://developer.download.nvidia.com/compute/machine-learning/tensorrt/$($TensorRtVersion -replace '^(\d+\.\d+\.\d+)\..*', '$1')/zip/TensorRT-Enterprise-$TensorRtVersion-Windows-amd64-cuda-$cudaMajorMinor-Release-external.zip" $zip
            $extract = Join-Path $tmp 'tensorrt'
            Expand-Into $zip $extract
            # The archive holds a single top-level TensorRT-* directory.
            $inner = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
            if (Test-Path -LiteralPath $trtRoot) { Remove-Item -Recurse -Force -LiteralPath $trtRoot }
            Move-Item -LiteralPath $inner.FullName -Destination $trtRoot
        }
        if (-not (Test-Path (Join-Path $trtRoot 'include\NvInferVersion.h'))) {
            throw "NvInferVersion.h not found under '$trtRoot\include'."
        }
        Set-BuildVar 'TENSORRT_DIR' $trtRoot
        if ($env:GITHUB_PATH) { Add-Content -LiteralPath $env:GITHUB_PATH -Value "$trtRoot\bin" }
    }

    if ($Component -contains 'openvino') {
        $archive = "openvino_toolkit_windows_$OpenVinoVersion.${OpenVinoBuild}_x86_64"
        $ovRoot = Join-Path $DepsDir 'openvino'
        if (-not (Test-Path (Join-Path $ovRoot 'runtime\cmake'))) {
            $zip = Join-Path $tmp 'openvino.zip'
            Get-Download "https://storage.openvinotoolkit.org/repositories/openvino/packages/$OpenVinoVersion/windows/$archive.zip" $zip
            $extract = Join-Path $tmp 'openvino'
            Expand-Into $zip $extract
            $inner = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
            if (Test-Path -LiteralPath $ovRoot) { Remove-Item -Recurse -Force -LiteralPath $ovRoot }
            Move-Item -LiteralPath $inner.FullName -Destination $ovRoot
        }
        # CMake looks for ${OPENVINO_DIR}/cmake, which lives under runtime/.
        Set-BuildVar 'OPENVINO_DIR' (Join-Path $ovRoot 'runtime')
        if ($env:GITHUB_PATH) {
            Add-Content -LiteralPath $env:GITHUB_PATH -Value (Join-Path $ovRoot 'runtime\bin\intel64\Release')
            Add-Content -LiteralPath $env:GITHUB_PATH -Value (Join-Path $ovRoot 'runtime\3rdparty\tbb\bin')
        }
    }
}
finally {
    Remove-Item -Recurse -Force -LiteralPath $tmp -ErrorAction SilentlyContinue
}
