import ctypes
from ctypes import wintypes

k32 = ctypes.windll.kernel32
k32.LoadLibraryExW.restype = wintypes.HMODULE
k32.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HMODULE, wintypes.DWORD]
k32.GetProcAddress.restype = wintypes.HMODULE  # actually FARPROC; truthiness is enough
k32.GetProcAddress.argtypes = [wintypes.HMODULE, ctypes.c_char_p]
k32.FreeLibrary.argtypes = [wintypes.HMODULE]

path = r"cpp\install\x64-release\bin\infertensorrt_cpp.dll"


def rva_to_offset(data, rva):
    # PE headers: image base-relative RVA -> file offset via section headers.
    e_lfanew = int.from_bytes(data[0x3C:0x40], "little")
    n_sections = int.from_bytes(data[e_lfanew + 6:e_lfanew + 8], "little")
    section_off = e_lfanew + 24 + int.from_bytes(data[e_lfanew + 20:e_lfanew + 22], "little")
    for i in range(n_sections):
        s = data[section_off + i * 40: section_off + (i + 1) * 40]
        vaddr = int.from_bytes(s[0:4], "little")
        vsize = int.from_bytes(s[8:12], "little")
        raw = int.from_bytes(s[16:20], "little")
        if vaddr <= rva < vaddr + vsize:
            return raw + (rva - vaddr)
    return None


def read_u32(data, off):
    return int.from_bytes(data[off:off + 4], "little")


def imports(data):
    e_lfanew = int.from_bytes(data[0x3C:0x40], "little")
    opt_off = e_lfanew + 24
    magic = read_u32(data, opt_off)  # 0x200 = PE32+
    dd_off = opt_off + (112 if magic == 0x200 else 96) + 8 * 1  # IMPORT dir = index 1
    imp_rva = read_u32(data, dd_off)
    names = []
    desc = rva_to_offset(data, imp_rva)
    while desc:
        name_rva = read_u32(data, desc + 12)
        if name_rva == 0:
            break
        name_off = rva_to_offset(data, name_rva)
        end = data.index(b"\0", name_off)
        names.append(data[name_off:end].decode())
        desc += 20
    return names


with open(path, "rb") as fh:
    data = fh.read()

deps = imports(data)
print("imports:", deps)

for dep in deps:
    h = k32.LoadLibraryExW(dep, None, 0)
    print(f"  {dep}: loaded={bool(h)} error={k32.GetLastError()}")

h = k32.LoadLibraryExW(path, None, 0)
err = k32.GetLastError()
print("bridge load: handle =", bool(h), "error =", err)
if h:
    for name in (b"rs_infer_llarray_create", b"rs_inference_runner_create",
                 b"rs_inference_runner_get_results"):
        print(name, bool(k32.GetProcAddress(h, name)))
    k32.FreeLibrary(h)
