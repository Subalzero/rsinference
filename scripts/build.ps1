[CmdletBinding()]
param(
    [Parameter()]
    [string] $CudaRoot = $env:CUDA_TOOLKIT_ROOT_DIR,

    [Parameter()]
    [string] $TensorRtRoot = $env:TENSORRT_DIR,

    [Parameter()]
    [string] $VcpkgRoot = (Join-Path $PSScriptRoot '..\cpp\thirdparty\vcpkg'),

    [Parameter()]
    [string] $LlvmBin = 'C:\Program Files\LLVM\bin',

    [Parameter()]
    [string] $OpenCvDir,

    [ValidateSet('Debug', 'Release')]
    [string] $Configuration = 'Release',

    [switch] $BuildFaceDetection
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$cppRoot = Join-Path $repoRoot 'cpp'
$buildDir = Join-Path $cppRoot "build/x64-$($Configuration.ToLowerInvariant())"
$toolchainFile = Join-Path $VcpkgRoot 'scripts/buildsystems/vcpkg.cmake'

foreach ($required in @(
    @{ Name = 'CUDA Toolkit'; Path = $CudaRoot; Sentinel = 'include/cuda.h' },
    @{ Name = 'TensorRT'; Path = $TensorRtRoot; Sentinel = 'include/NvInferVersion.h' },
    @{ Name = 'vcpkg toolchain'; Path = $toolchainFile; Sentinel = $null },
    @{ Name = 'LLVM bin'; Path = $LlvmBin; Sentinel = 'clang.exe' }
)) {
    if ([string]::IsNullOrWhiteSpace($required.Path) -or -not (Test-Path -LiteralPath $required.Path)) {
        throw "Path for $($required.Name) does not exist: '$($required.Path)'. Pass the corresponding script parameter."
    }
    if ($required.Sentinel -and -not (Test-Path -LiteralPath (Join-Path $required.Path $required.Sentinel))) {
        throw "Could not find $($required.Sentinel) under $($required.Name) path '$($required.Path)'."
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $LlvmBin 'libclang.dll'))) {
    throw "Could not find libclang.dll under LLVM bin path '$LlvmBin'. Install LLVM with libclang or pass its bin directory."
}

$configureArgs = @(
    '-S', $cppRoot,
    '-B', $buildDir,
    '-G', 'Ninja',
    "-DCMAKE_BUILD_TYPE=$Configuration",
    "-DCMAKE_TOOLCHAIN_FILE=$toolchainFile",
    "-DCUDA_TOOLKIT_ROOT_DIR=$CudaRoot",
    "-DCUDAToolkit_ROOT=$CudaRoot",
    "-DTENSORRT_DIR=$TensorRtRoot"
)
if ($OpenCvDir) {
    $configureArgs += "-DOpenCV_DIR=$OpenCvDir"
}

Push-Location $repoRoot
try {
    & cmake @configureArgs
    if ($LASTEXITCODE -ne 0) { throw "CMake configure failed with exit code $LASTEXITCODE." }

    & cmake --build $buildDir --config $Configuration
    if ($LASTEXITCODE -ne 0) { throw "C++ build failed with exit code $LASTEXITCODE." }

    $env:CLANG_PATH = Join-Path $LlvmBin 'clang.exe'
    $env:LIBCLANG_PATH = $LlvmBin
    $env:INFERTENSORRT_BIN_DIR = $buildDir
    if ($Configuration -eq 'Release') { & cargo build --release }
    else { & cargo build }
    if ($LASTEXITCODE -ne 0) { throw "Rust build failed with exit code $LASTEXITCODE." }

    if ($BuildFaceDetection) {
        $vcpkgInstalled = Join-Path $buildDir 'vcpkg_installed/x64-windows'
        $env:OPENCV_LINK_PATHS = Join-Path $vcpkgInstalled 'lib'
        $env:OPENCV_INCLUDE_PATHS = Join-Path $vcpkgInstalled 'include/opencv4'
        $env:OPENCV_LINK_LIBS = 'opencv_core4,opencv_imgproc4,opencv_videoio4,opencv_highgui4'
        Push-Location (Join-Path $repoRoot 'examples/facedetection')
        try {
            if ($Configuration -eq 'Release') { & cargo build --release }
            else { & cargo build }
            if ($LASTEXITCODE -ne 0) { throw "Face detection example build failed with exit code $LASTEXITCODE." }
        }
        finally { Pop-Location }
    }
}
finally { Pop-Location }
