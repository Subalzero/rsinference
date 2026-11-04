# Local emulation of the cpp-windows and rust-windows jobs in .github/workflows/develop.yml.
param([ValidateSet('all', 'cpp', 'rust')] [string] $Job = 'all')
$ErrorActionPreference = 'Stop'

function Step([string] $Name, [scriptblock] $Body) {
    Write-Host "`n##### STEP: $Name #####" -ForegroundColor Cyan
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $global:LASTEXITCODE = 0
    & $Body
    if ($LASTEXITCODE -ne 0) { throw "STEP FAILED: $Name (exit $LASTEXITCODE)" }
    Write-Host "##### OK: $Name ($([int]$sw.Elapsed.TotalSeconds)s) #####" -ForegroundColor Green
}

# Apply GITHUB_ENV / GITHUB_PATH the way the runner does between steps.
function Sync-GitHubEnv {
    if (Test-Path $env:GITHUB_ENV) {
        foreach ($l in Get-Content $env:GITHUB_ENV) { if ($l -match '^([^=]+)=(.*)$') { Set-Item "Env:$($Matches[1])" $Matches[2] } }
    }
    if (Test-Path $env:GITHUB_PATH) {
        foreach ($p in Get-Content $env:GITHUB_PATH) { if ($p -and ($env:PATH -notlike "*$p*")) { $env:PATH = "$p;$env:PATH" } }
    }
}

Step 'checkout (copy working tree)' {
    robocopy C:\src C:\work /E /NFL /NDL /NJH /NP /MT:16 `
        /XD C:\src\target C:\src\cpp\build C:\src\cpp\vcpkg_installed C:\src\cpp\install C:\src\.vcpkg-cache C:\src\bridge C:\src\scripts\docker-ci\out `
            C:\src\cpp\thirdparty\vcpkg\buildtrees C:\src\cpp\thirdparty\vcpkg\downloads C:\src\cpp\thirdparty\vcpkg\packages `
            C:\src\cpp\thirdparty\vcpkg\installed C:\src\cpp\thirdparty\vcpkg\vcpkg.exe | Out-Null
    if ($LASTEXITCODE -lt 8) { $global:LASTEXITCODE = 0 }
    Remove-Item -Force C:\work\cpp\thirdparty\vcpkg\vcpkg.exe -ErrorAction SilentlyContinue
    git config --global --add safe.directory '*'
}

$ws = 'C:\work'
$env:GITHUB_ENV = 'C:\ci\github_env'; $env:GITHUB_PATH = 'C:\ci\github_path'
New-Item -Force -ItemType File $env:GITHUB_ENV, $env:GITHUB_PATH | Out-Null
$env:CARGO_TERM_COLOR = 'always'

if ($Job -in 'all', 'cpp') {
    $env:VCPKG_ROOT = "$ws/cpp/thirdparty/vcpkg"
    $env:VCPKG_DEFAULT_BINARY_CACHE = "$ws/.vcpkg-cache"
    $env:DEPS_DIR = 'C:\deps'
    $env:BUILD_DIR = 'cpp/build/x64-release'

    Step 'Install dependencies' { pwsh -NoProfile -File ./scripts/install-deps.ps1 }
    Sync-GitHubEnv

    Step 'Bootstrap vcpkg' {
        New-Item -ItemType Directory -Force $env:VCPKG_DEFAULT_BINARY_CACHE | Out-Null
        & "$env:VCPKG_ROOT/bootstrap-vcpkg.bat" -disableMetrics
    }

    Step 'msvc-dev-cmd' {
        & C:\BuildTools\Common7\Tools\Launch-VsDevShell.ps1 -Arch amd64 -HostArch amd64 -SkipAutomaticLocation | Out-Null
        Get-Command cl.exe | Select-Object -Expand Source
    }

    Step 'Configure' {
        cmake -S cpp -B $env:BUILD_DIR -G Ninja `
            -DCMAKE_BUILD_TYPE=Release `
            "-DCMAKE_C_COMPILER=cl.exe" `
            "-DCMAKE_CXX_COMPILER=cl.exe" `
            -DCMAKE_TOOLCHAIN_FILE="$ws/cpp/thirdparty/vcpkg/scripts/buildsystems/vcpkg.cmake" `
            -DCUDA_TOOLKIT_ROOT_DIR="$env:CUDA_TOOLKIT_ROOT_DIR" `
            -DCUDAToolkit_ROOT="$env:CUDA_TOOLKIT_ROOT_DIR" `
            -DTENSORRT_DIR="$env:TENSORRT_DIR" `
            -DOPENVINO_DIR="$env:OPENVINO_DIR" `
            -DBUILD_EXAMPLES=OFF
    }

    Step 'Build' { cmake --build $env:BUILD_DIR }

    Step 'Test' {
        $env:PATH = "$env:TENSORRT_DIR\bin;$env:OPENVINO_DIR\bin\intel64\Release;$env:OPENVINO_DIR\3rdparty\tbb\bin;$env:PATH"
        ctest --test-dir $env:BUILD_DIR --output-on-failure
    }

    Step 'Collect bridge libraries' {
        New-Item -ItemType Directory -Force bridge | Out-Null
        Copy-Item "$env:BUILD_DIR/infer*.dll" bridge/
        if (-not (Get-ChildItem bridge/*.dll)) { throw 'if-no-files-found: error — no DLLs collected' }
        Get-ChildItem bridge
        New-Item -ItemType Directory -Force C:\out\bridge | Out-Null
        Copy-Item bridge/*.dll C:\out\bridge\
    }
}
if ($Job -in 'all', 'rust') {
    # In 'all' mode this runs in the same environment as the C++ job (deps on PATH via GITHUB_PATH).
    $env:LIBCLANG_PATH = 'C:\Program Files\LLVM\bin'
    Step 'download-artifact bridge-windows' {
        New-Item -ItemType Directory -Force bridge | Out-Null
        Copy-Item C:\out\bridge\*.dll bridge/
        Get-ChildItem bridge
    }
    $env:INFERTENSORRT_BIN_DIR = "$ws/bridge"
    Step 'Build' { cargo build --verbose }
    Step 'Test' { cargo test --verbose }
}
Write-Host "`n##### JOB $Job SUCCEEDED #####" -ForegroundColor Green
