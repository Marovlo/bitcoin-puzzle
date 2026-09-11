// amd_power.cpp
//
// Minimal AMD Display Library (ADL) client for two things the OS does not
// expose on Windows:
//
//   1. Reading real GPU power draw, temperature and fan speed (PMLog sensors).
//      Windows' own perf counters have no GPU power counter at all.
//   2. Setting the OverDrive8 power limit and GFX clock ceiling - the exact
//      same knobs the Adrenalin "Performance > Tuning" sliders drive, but
//      reachable from a script.
//
// Everything is resolved from atiadlxx.dll at runtime, so this links against
// nothing but kernel32 and needs no ADL import library.
//
// Usage:
//   amd_power info                     dump adapters, OD8 capabilities and ranges
//   amd_power watch <seconds>          sample power/temp/fan once a second
//   amd_power set-power <pct>          set OverDrive8 power limit, in percent
//   amd_power set-clkmax <mhz>         set GFX clock ceiling, in MHz
//   amd_power set <k=v[,k=v...]>       keys: power, clkmax
//   amd_power reset                    restore default power limit and clock

#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "adl_sdk.h"
#include "adl_structures.h"

// ---------------------------------------------------------------------------
// Entry points. Names and signatures are taken from the ADL SDK's Overdrive8
// sample, which is the reference usage for these APIs.
// ---------------------------------------------------------------------------
typedef int (*ADL2_MAIN_CONTROL_CREATE)(ADL_MAIN_MALLOC_CALLBACK, int, ADL_CONTEXT_HANDLE*);
typedef int (*ADL2_MAIN_CONTROL_DESTROY)(ADL_CONTEXT_HANDLE);
typedef int (*ADL2_ADAPTER_NUMBEROFADAPTERS_GET)(ADL_CONTEXT_HANDLE, int*);
typedef int (*ADL2_ADAPTER_ADAPTERINFO_GET)(ADL_CONTEXT_HANDLE, LPAdapterInfo, int);
typedef int (*ADL2_ADAPTER_ACTIVE_GET)(ADL_CONTEXT_HANDLE, int, int*);
typedef int (*ADL2_OVERDRIVE_CAPS)(ADL_CONTEXT_HANDLE, int, int*, int*, int*);
typedef int (*ADL2_ADAPTER_REGVALUEINT_SET)(ADL_CONTEXT_HANDLE, int, int, char*, char*, int);

typedef int (*ADL2_OVERDRIVE8_INIT_SETTING_GET)(ADL_CONTEXT_HANDLE, int, ADLOD8InitSetting*);
typedef int (*ADL2_OVERDRIVE8_CURRENT_SETTING_GET)(ADL_CONTEXT_HANDLE, int, ADLOD8CurrentSetting*);
typedef int (*ADL2_OVERDRIVE8_SETTING_SET)(ADL_CONTEXT_HANDLE, int, ADLOD8SetSetting*, ADLOD8CurrentSetting*);
typedef int (*ADL2_OVERDRIVE8_INIT_SETTINGX2_GET)(ADL_CONTEXT_HANDLE, int, int*, int*, ADLOD8SingleInitSetting**);
typedef int (*ADL2_OVERDRIVE8_CURRENT_SETTINGX2_GET)(ADL_CONTEXT_HANDLE, int, int*, int**);

typedef int (*ADL2_OVERDRIVE8_PMLOGSENORTYPE_SUPPORT_GET)(ADL_CONTEXT_HANDLE, int, int*, int**);
typedef int (*ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_SUPPORT)(ADL_CONTEXT_HANDLE, int, int*, int);
typedef int (*ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_START)(ADL_CONTEXT_HANDLE, int, int, int, int*,
                                                       ADL_D3DKMT_HANDLE*, void**, int);
typedef int (*ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_READ)(ADL_CONTEXT_HANDLE, int, int, int*, void**,
                                                      ADLPMLogDataOutput*);
typedef int (*ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_STOP)(ADL_CONTEXT_HANDLE, int, ADL_D3DKMT_HANDLE*);
typedef int (*ADL2_DEVICE_PMLOG_DEVICE_CREATE)(ADL_CONTEXT_HANDLE, int, ADL_D3DKMT_HANDLE*);
typedef int (*ADL2_DEVICE_PMLOG_DEVICE_DESTROY)(ADL_CONTEXT_HANDLE, ADL_D3DKMT_HANDLE);

static HINSTANCE g_dll = nullptr;
static ADL_CONTEXT_HANDLE g_ctx = nullptr;

static ADL2_MAIN_CONTROL_CREATE                  p_MainControlCreate = nullptr;
static ADL2_MAIN_CONTROL_DESTROY                 p_MainControlDestroy = nullptr;
static ADL2_ADAPTER_NUMBEROFADAPTERS_GET         p_NumAdapters = nullptr;
static ADL2_ADAPTER_ADAPTERINFO_GET              p_AdapterInfo = nullptr;
static ADL2_ADAPTER_ACTIVE_GET                   p_AdapterActive = nullptr;
static ADL2_OVERDRIVE_CAPS                       p_OverdriveCaps = nullptr;
static ADL2_ADAPTER_REGVALUEINT_SET              p_RegValueIntSet = nullptr;
static ADL2_OVERDRIVE8_INIT_SETTING_GET          p_Od8InitGet = nullptr;
static ADL2_OVERDRIVE8_CURRENT_SETTING_GET       p_Od8CurGet = nullptr;
static ADL2_OVERDRIVE8_SETTING_SET               p_Od8Set = nullptr;
static ADL2_OVERDRIVE8_INIT_SETTINGX2_GET        p_Od8InitGetX2 = nullptr;
static ADL2_OVERDRIVE8_CURRENT_SETTINGX2_GET     p_Od8CurGetX2 = nullptr;
static ADL2_OVERDRIVE8_PMLOGSENORTYPE_SUPPORT_GET p_PmSensorSupport = nullptr;
static ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_SUPPORT p_PmShareSupport = nullptr;
static ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_START   p_PmStart = nullptr;
static ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_READ    p_PmRead = nullptr;
static ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_STOP    p_PmStop = nullptr;
static ADL2_DEVICE_PMLOG_DEVICE_CREATE           p_PmDeviceCreate = nullptr;
static ADL2_DEVICE_PMLOG_DEVICE_DESTROY          p_PmDeviceDestroy = nullptr;

extern "C" void* __stdcall ADL_Main_Memory_Alloc(int size) { return malloc(size); }
static void ADL_Main_Memory_Free(void** buf) {
    if (buf && *buf) { free(*buf); *buf = nullptr; }
}

// ---------------------------------------------------------------------------
// Names for the OD8 rows we care about; OD8_COUNT is 78, so printing the raw
// table without labels is unreadable.
// ---------------------------------------------------------------------------
struct Od8Name { int id; const char* name; };
static const Od8Name kOd8Names[] = {
    { OD8_GFXCLK_FMAX,            "GFXCLK_FMAX (MHz)" },
    { OD8_GFXCLK_FMIN,            "GFXCLK_FMIN (MHz)" },
    { OD8_GFXCLK_FREQ1,           "GFXCLK_FREQ1 (MHz)" },
    { OD8_GFXCLK_VOLTAGE1,        "GFXCLK_VOLTAGE1 (mV)" },
    { OD8_GFXCLK_FREQ2,           "GFXCLK_FREQ2 (MHz)" },
    { OD8_GFXCLK_VOLTAGE2,        "GFXCLK_VOLTAGE2 (mV)" },
    { OD8_GFXCLK_FREQ3,           "GFXCLK_FREQ3 (MHz)" },
    { OD8_GFXCLK_VOLTAGE3,        "GFXCLK_VOLTAGE3 (mV)" },
    { OD8_UCLK_FMAX,              "UCLK_FMAX (MHz)" },
    { OD8_POWER_PERCENTAGE,       "POWER_PERCENTAGE (%)" },
    { OD8_FAN_MIN_SPEED,          "FAN_MIN_SPEED (rpm)" },
    { OD8_FAN_ACOUSTIC_LIMIT,     "FAN_ACOUSTIC_LIMIT (max rpm)" },
    { OD8_FAN_TARGET_TEMP,        "FAN_TARGET_TEMP (C)" },
    { OD8_OPERATING_TEMP_MAX,     "OPERATING_TEMP_MAX (C)" },
    { OD8_OD_VOLTAGE,             "OD_VOLTAGE (mV offset)" },
    { OD8_TDC_PERCENTAGE,         "TDC_PERCENTAGE (%)" },
    { OD8_OPTIMZED_POWER_MODE,    "OPTIMIZED_POWER_MODE" },
};
static const char* od8_name(int id) {
    for (const auto& n : kOd8Names) if (n.id == id) return n.name;
    return nullptr;
}

static void print_caps(int caps) {
    struct Cap { int bit; const char* name; };
    static const Cap kCaps[] = {
        { ADL_OD8_GFXCLK_LIMITS,          "GFXCLK_LIMITS" },
        { ADL_OD8_GFXCLK_CURVE,           "GFXCLK_CURVE" },
        { ADL_OD8_UCLK_MAX,               "UCLK_MAX" },
        { ADL_OD8_POWER_LIMIT,            "POWER_LIMIT" },
        { ADL_OD8_ACOUSTIC_LIMIT_SCLK,    "ACOUSTIC_LIMIT (max fan rpm)" },
        { ADL_OD8_FAN_SPEED_MIN,          "FAN_SPEED_MIN" },
        { ADL_OD8_TEMPERATURE_FAN,        "TEMPERATURE_FAN" },
        { ADL_OD8_FAN_ZERO_RPM_CONTROL,   "FAN_ZERO_RPM" },
        { ADL_OD8_AUTO_UV_ENGINE,         "AUTO_UV_ENGINE" },
        { ADL_OD8_FAN_CURVE,              "FAN_CURVE" },
        { ADL_OD8_OPTIMIZED_GPU_POWER_MODE, "OPTIMIZED_GPU_POWER_MODE" },
        { ADL_OD8_ODVOLTAGE_LIMIT,        "ODVOLTAGE_LIMIT" },
        { ADL_OD8_GFX_VOLTAGE_LIMIT,      "GFX_VOLTAGE_LIMIT" },
        { ADL_OD8_TDC_LIMIT,              "TDC_LIMIT" },
        { ADL_OD8_FULL_CONTROL_MODE,      "FULL_CONTROL_MODE" },
        { ADL_OD8_POWER_GAUGE,            "POWER_GAUGE" },
    };
    printf("  capabilities = 0x%08x\n", (unsigned)caps);
    for (const auto& c : kCaps)
        printf("    [%c] %s\n", (caps & c.bit) ? 'x' : ' ', c.name);
}

// Loads the library and resolves every entry point we may want. Missing
// optional entries are tolerated; missing mandatory ones are fatal.
static bool adl_open() {
    g_dll = LoadLibraryA("atiadlxx.dll");
    if (!g_dll) g_dll = LoadLibraryA("atiadlxy.dll");   // 32-bit library
    if (!g_dll) { printf("FATAL: atiadlxx.dll not found\n"); return false; }

    p_MainControlCreate = (ADL2_MAIN_CONTROL_CREATE)GetProcAddress(g_dll, "ADL2_Main_Control_Create");
    p_MainControlDestroy = (ADL2_MAIN_CONTROL_DESTROY)GetProcAddress(g_dll, "ADL2_Main_Control_Destroy");
    p_NumAdapters = (ADL2_ADAPTER_NUMBEROFADAPTERS_GET)GetProcAddress(g_dll, "ADL2_Adapter_NumberOfAdapters_Get");
    p_AdapterInfo = (ADL2_ADAPTER_ADAPTERINFO_GET)GetProcAddress(g_dll, "ADL2_Adapter_AdapterInfo_Get");
    p_AdapterActive = (ADL2_ADAPTER_ACTIVE_GET)GetProcAddress(g_dll, "ADL2_Adapter_Active_Get");
    p_OverdriveCaps = (ADL2_OVERDRIVE_CAPS)GetProcAddress(g_dll, "ADL2_Overdrive_Caps");
    p_RegValueIntSet = (ADL2_ADAPTER_REGVALUEINT_SET)GetProcAddress(g_dll, "ADL2_Adapter_RegValueInt_Set");
    p_Od8InitGet = (ADL2_OVERDRIVE8_INIT_SETTING_GET)GetProcAddress(g_dll, "ADL2_Overdrive8_Init_Setting_Get");
    p_Od8CurGet = (ADL2_OVERDRIVE8_CURRENT_SETTING_GET)GetProcAddress(g_dll, "ADL2_Overdrive8_Current_Setting_Get");
    p_Od8Set = (ADL2_OVERDRIVE8_SETTING_SET)GetProcAddress(g_dll, "ADL2_Overdrive8_Setting_Set");
    p_Od8InitGetX2 = (ADL2_OVERDRIVE8_INIT_SETTINGX2_GET)GetProcAddress(g_dll, "ADL2_Overdrive8_Init_SettingX2_Get");
    p_Od8CurGetX2 = (ADL2_OVERDRIVE8_CURRENT_SETTINGX2_GET)GetProcAddress(g_dll, "ADL2_Overdrive8_Current_SettingX2_Get");

    p_PmSensorSupport = (ADL2_OVERDRIVE8_PMLOGSENORTYPE_SUPPORT_GET)GetProcAddress(g_dll, "ADL2_Overdrive8_PMLogSenorType_Support_Get");
    p_PmShareSupport = (ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_SUPPORT)GetProcAddress(g_dll, "ADL2_Overdrive8_PMLog_ShareMemory_Support");
    p_PmStart = (ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_START)GetProcAddress(g_dll, "ADL2_Overdrive8_PMLog_ShareMemory_Start");
    p_PmRead = (ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_READ)GetProcAddress(g_dll, "ADL2_Overdrive8_PMLog_ShareMemory_Read");
    p_PmStop = (ADL2_OVERDRIVE8_PMLOG_SHAREMEMORY_STOP)GetProcAddress(g_dll, "ADL2_Overdrive8_PMLog_ShareMemory_Stop");
    p_PmDeviceCreate = (ADL2_DEVICE_PMLOG_DEVICE_CREATE)GetProcAddress(g_dll, "ADL2_Device_PMLog_Device_Create");
    p_PmDeviceDestroy = (ADL2_DEVICE_PMLOG_DEVICE_DESTROY)GetProcAddress(g_dll, "ADL2_Device_PMLog_Device_Destroy");

    if (!p_MainControlCreate || !p_NumAdapters || !p_AdapterInfo) {
        printf("FATAL: atiadlxx.dll is missing the ADL2 core entry points\n");
        return false;
    }
    if (ADL_OK != p_MainControlCreate(ADL_Main_Memory_Alloc, 1, &g_ctx)) {
        printf("FATAL: ADL2_Main_Control_Create failed\n");
        return false;
    }
    return true;
}

static void adl_close() {
    if (g_ctx && p_MainControlDestroy) p_MainControlDestroy(g_ctx);
    g_ctx = nullptr;
    if (g_dll) FreeLibrary(g_dll);
    g_dll = nullptr;
}

// Reads the OD8 range table and current values, preferring the X2 variants the
// sample uses (the non-X2 ones return ADL_ERR on recent drivers).
static bool read_od8(int adapter, ADLOD8InitSetting& init, ADLOD8CurrentSetting& cur) {
    memset(&init, 0, sizeof(init));
    memset(&cur, 0, sizeof(cur));

    bool initOk = false;
    if (p_Od8InitGetX2) {
        int caps = 0, n = 0;
        ADLOD8SingleInitSetting* list = nullptr;
        int rc = p_Od8InitGetX2(g_ctx, adapter, &caps, &n, &list);
        printf("  [dbg] Init_SettingX2_Get  rc=%d caps=0x%08x n=%d\n", rc, (unsigned)caps, n);
        // The X2 call can report success while handing back an empty list; an
        // empty list must fall through to the plain variant rather than being
        // accepted as "no tunables".
        if (rc == ADL_OK && list && n > 0) {
            init.count = (n > OD8_COUNT) ? OD8_COUNT : n;
            init.overdrive8Capabilities = caps;
            for (int i = 0; i < init.count; i++) {
                init.od8SettingTable[i].featureID    = list[i].featureID;
                init.od8SettingTable[i].minValue     = list[i].minValue;
                init.od8SettingTable[i].maxValue     = list[i].maxValue;
                init.od8SettingTable[i].defaultValue = list[i].defaultValue;
            }
            ADL_Main_Memory_Free((void**)&list);
            initOk = true;
        } else if (list) {
            ADL_Main_Memory_Free((void**)&list);
        }
    }
    if (!initOk && p_Od8InitGet) {
        // This driver build treats `count` as the capacity of od8SettingTable on
        // input and does not write it back, so seed it rather than leaving zero.
        init.count = OD8_COUNT;
        int rc = p_Od8InitGet(g_ctx, adapter, &init);
        int rowsWithRange = 0;
        for (int i = 0; i < OD8_COUNT; i++)
            if (init.od8SettingTable[i].maxValue != init.od8SettingTable[i].minValue) rowsWithRange++;
        printf("  [dbg] Init_Setting_Get    rc=%d count=%d caps=0x%08x rows_with_range=%d\n",
               rc, init.count, (unsigned)init.overdrive8Capabilities, rowsWithRange);
        // Trust the table when it actually carries ranges, even if the driver
        // left count at zero.
        if (rc == ADL_OK && (init.count > 0 || rowsWithRange > 0)) {
            if (init.count <= 0) init.count = OD8_COUNT;
            initOk = true;
        }
    }
    if (!initOk) { printf("  OD8 init settings unavailable\n"); return false; }

    bool curOk = false;
    if (p_Od8CurGetX2) {
        int n = 0;
        int* list = nullptr;
        int rc = p_Od8CurGetX2(g_ctx, adapter, &n, &list);
        printf("  [dbg] Current_SettingX2_Get rc=%d n=%d\n", rc, n);
        if (rc == ADL_OK && list && n > 0) {
            cur.count = (n > OD8_COUNT) ? OD8_COUNT : n;
            for (int i = 0; i < cur.count; i++) cur.Od8SettingTable[i] = list[i];
            ADL_Main_Memory_Free((void**)&list);
            curOk = true;
        } else if (list) {
            ADL_Main_Memory_Free((void**)&list);
        }
    }
    if (!curOk && p_Od8CurGet) {
        cur.count = OD8_COUNT;   // capacity on input, same convention as above
        int rc = p_Od8CurGet(g_ctx, adapter, &cur);
        printf("  [dbg] Current_Setting_Get rc=%d count=%d\n", rc, cur.count);
        if (rc == ADL_OK) {
            if (cur.count <= 0) cur.count = OD8_COUNT;
            curOk = true;
        }
    }
    if (!curOk) { printf("  OD8 current settings unavailable\n"); return false; }
    return true;
}

// Mirror of the sample's SetOD8PlusRange: on Navi 21+ the whole GFX clock slider
// group has to be submitted together, and the ASIC must be in Manual power mode
// before the driver will accept a power limit.
static bool od8_power_mode_manual(int adapter, ADLOD8CurrentSetting& cur) {
    if (!(cur.count)) return false;
    ADLOD8SetSetting s;
    memset(&s, 0, sizeof(s));
    s.count = OD8_COUNT;
    s.od8SettingTable[OD8_OPTIMZED_POWER_MODE].requested = 1;
    s.od8SettingTable[OD8_OPTIMZED_POWER_MODE].value = 3;      // 3 = Manual
    int rc = p_Od8Set(g_ctx, adapter, &s, &cur);
    printf("  switch to Manual power mode: %s (rc=%d)\n", (rc == ADL_OK) ? "ok" : "FAILED", rc);
    return rc == ADL_OK;
}

static bool od8_apply(int adapter, const ADLOD8InitSetting& init, ADLOD8CurrentSetting& cur,
                      int settingId, int value, bool reset) {
    if (settingId < 0 || settingId >= OD8_COUNT) return false;

    if (!reset && (value < init.od8SettingTable[settingId].minValue ||
                   value > init.od8SettingTable[settingId].maxValue)) {
        printf("  %s: %d out of driver range [%d, %d]\n", od8_name(settingId) ? od8_name(settingId) : "setting",
               value, init.od8SettingTable[settingId].minValue, init.od8SettingTable[settingId].maxValue);
        return false;
    }

    ADLOD8SetSetting s;
    memset(&s, 0, sizeof(s));
    s.count = OD8_COUNT;

    // Settings the driver insists on receiving as a group, carrying their
    // current values so nothing else is disturbed.
    if (settingId <= OD8_GFXCLK_FMIN && settingId >= OD8_GFXCLK_FMAX) {
        for (int i = OD8_GFXCLK_FMAX; i <= OD8_GFXCLK_FMIN; ++i) {
            s.od8SettingTable[i].requested = 1;
            s.od8SettingTable[i].value = cur.Od8SettingTable[i];
        }
    } else if (settingId == OD8_UCLK_FMIN || settingId == OD8_UCLK_FMAX) {
        for (int i = OD8_UCLK_FMAX; i <= OD8_UCLK_FMIN; ++i) {
            s.od8SettingTable[i].requested = 1;
            s.od8SettingTable[i].value = cur.Od8SettingTable[i];
        }
    } else if (settingId >= OD8_FAN_CURVE_TEMPERATURE_1 && settingId <= OD8_FAN_CURVE_SPEED_5) {
        for (int i = OD8_FAN_CURVE_TEMPERATURE_1; i <= OD8_FAN_CURVE_SPEED_5; ++i) {
            s.od8SettingTable[i].requested = 1;
            s.od8SettingTable[i].value = cur.Od8SettingTable[i];
        }
    }

    s.od8SettingTable[settingId].requested = 1;
    if (reset) {
        s.od8SettingTable[settingId].reset = 1;
        s.od8SettingTable[settingId].value = init.od8SettingTable[settingId].defaultValue;
    } else {
        s.od8SettingTable[settingId].value = value;
        // The clock ceiling/multiplier rows move together on this ASIC.
        if (settingId == OD8_GFXCLK_FMAX) s.od8SettingTable[OD8_GFXCLK_FREQ3].value = value;
        else if (settingId == OD8_GFXCLK_FMIN) s.od8SettingTable[OD8_GFXCLK_FREQ1].value = value;
    }

    int rc = p_Od8Set(g_ctx, adapter, &s, &cur);
    printf("  %-28s -> %-7d : %s (rc=%d)\n",
           od8_name(settingId) ? od8_name(settingId) : "setting",
           reset ? init.od8SettingTable[settingId].defaultValue : value,
           (rc == ADL_OK) ? "ok" : "FAILED", rc);
    if (rc == ADL_OK && p_RegValueIntSet) {
        // Clears the "this is still the factory default" flag, exactly as the
        // Overdrive8 sample does after a successful write.
        p_RegValueIntSet(g_ctx, adapter, 0x00000001, nullptr, (char*)"IsAutoDefault", 0);
    }
    return rc == ADL_OK;
}

// ---------------------------------------------------------------------------
// PMLog. Windows exposes no GPU power counter, so this is the only way to read
// actual watts without a third-party tool.
// ---------------------------------------------------------------------------
struct PmLog {
    ADL_D3DKMT_HANDLE dev = 0;
    void* shared = nullptr;
    int* sensors = nullptr;
    int num = 0;
    bool ok = false;

    bool start(int adapter) {
        if (!p_PmSensorSupport || !p_PmStart || !p_PmDeviceCreate) return false;

        int supported = 0;
        if (p_PmShareSupport && ADL_OK != p_PmShareSupport(g_ctx, adapter, &supported, 0)) return false;

        int n = 0;
        int* list = nullptr;
        if (ADL_OK != p_PmSensorSupport(g_ctx, adapter, &n, &list) || !list || n <= 0) return false;
        sensors = list;                 // ADL allocates this; freed in stop()
        num = n;

        if (ADL_OK != p_PmDeviceCreate(g_ctx, adapter, &dev)) return false;
        // sample rate 1000 ms, option 0
        if (ADL_OK != p_PmStart(g_ctx, adapter, 1000, num, sensors, &dev, &shared, 0)) {
            p_PmDeviceDestroy(g_ctx, dev);
            dev = 0;
            return false;
        }
        ok = true;
        return true;
    }

    bool read(int adapter, ADLPMLogDataOutput& out) {
        if (!ok) return false;
        memset(&out, 0, sizeof(out));
        return ADL_OK == p_PmRead(g_ctx, adapter, num, sensors, &shared, &out);
    }

    void stop(int adapter) {
        if (ok && p_PmStop) p_PmStop(g_ctx, adapter, &dev);
        if (dev && p_PmDeviceDestroy) p_PmDeviceDestroy(g_ctx, dev);
        dev = 0;
        shared = nullptr;
        if (sensors) ADL_Main_Memory_Free((void**)&sensors);
        ok = false;
    }
};

// ASIC power arrives in units of 1/100 W on this family; GFX power in the same
// unit. Printed both ways so a wrong guess is obvious rather than silent.
static void print_snapshot(const ADLPMLogDataOutput& d) {
    auto val = [&](int id) -> int {
        return (id >= 0 && id < ADL_PMLOG_MAX_SENSORS && d.sensors[id].supported) ? d.sensors[id].value : -1;
    };
    auto show = [&](const char* label, int id, const char* unit) {
        int v = val(id);
        if (v < 0) return;
        printf("  %-22s %8d %s\n", label, v, unit);
    };
    printf("  --- sensors ---\n");
    int p = val(ADL_PMLOG_ASIC_POWER);
    if (p >= 0) printf("  %-22s %8d  (= %.2f W if 1/100 W)\n", "ASIC_POWER", p, p / 100.0);
    int bp = val(ADL_PMLOG_BOARD_POWER);
    if (bp >= 0) printf("  %-22s %8d  (= %.2f W if 1/100 W)\n", "BOARD_POWER", bp, bp / 100.0);
    int gp = val(ADL_PMLOG_GFX_POWER);
    if (gp >= 0) printf("  %-22s %8d  (= %.2f W if 1/100 W)\n", "GFX_POWER", gp, gp / 100.0);
    show("TEMPERATURE_EDGE",     ADL_PMLOG_TEMPERATURE_EDGE,     "C");
    show("TEMPERATURE_HOTSPOT",  ADL_PMLOG_TEMPERATURE_HOTSPOT,  "C");
    show("FAN_RPM",              ADL_PMLOG_FAN_RPM,              "rpm");
    show("FAN_PERCENTAGE",       ADL_PMLOG_FAN_PERCENTAGE,       "%");
    show("CLK_GFXCLK",           ADL_PMLOG_CLK_GFXCLK,           "MHz");
    show("CLK_MEMCLK",           ADL_PMLOG_CLK_MEMCLK,           "MHz");
    show("GFX_VOLTAGE",          ADL_PMLOG_GFX_VOLTAGE,          "mV");
}

int main(int argc, char** argv) {
    std::string cmd = (argc > 1) ? argv[1] : "info";

    // A startup/logon scheduled task can fire before the display driver has
    // finished initialising, so give the ADL context a few chances to come up
    // rather than failing the one shot that matters.
    for (int i = 0; i < 20 && g_ctx == nullptr; i++) {
        if (!adl_open()) {
            if (i < 19) { printf("  retrying ADL init in 3s...\n"); Sleep(3000); }
        }
    }
    if (g_ctx == nullptr) { adl_close(); return 1; }

    int num = 0;
    if (ADL_OK != p_NumAdapters(g_ctx, &num) || num <= 0) {
        printf("FATAL: no ADL adapters\n"); adl_close(); return 1;
    }
    LPAdapterInfo ai = (LPAdapterInfo)calloc(num, sizeof(AdapterInfo));
    if (ADL_OK != p_AdapterInfo(g_ctx, ai, (int)(num * sizeof(AdapterInfo)))) {
        printf("FATAL: adapter info query failed\n"); free(ai); adl_close(); return 1;
    }

    printf("=== ADL adapters ===\n");
    int target = -1;
    for (int i = 0; i < num; i++) {
        int active = 0;
        if (p_AdapterActive) p_AdapterActive(g_ctx, ai[i].iAdapterIndex, &active);
        printf("  [%d] idx=%d vendor=%d (0x%x) present=%d active=%d  \"%s\"  (%s)\n",
               i, ai[i].iAdapterIndex, ai[i].iVendorID, ai[i].iVendorID, ai[i].iPresent, active,
               ai[i].strAdapterName, ai[i].strDisplayName);
        // ADL reports the PCI vendor id as decimal 1002 rather than 0x1002.
        // Prefer a present AMD adapter; skip the Microsoft basic render driver.
        // The driver lists one entry per display output, so take the first
        // (the active one) and ignore its clones.
        if (ai[i].iVendorID == 1002 && ai[i].iPresent && target < 0) target = ai[i].iAdapterIndex;
    }
    free(ai);
    if (target < 0) { printf("FATAL: no AMD adapter found\n"); adl_close(); return 1; }
    printf("target adapter index = %d\n", target);

    if (p_OverdriveCaps) {
        int sup = 0, en = 0, ver = 0;
        if (ADL_OK == p_OverdriveCaps(g_ctx, target, &sup, &en, &ver))
            printf("Overdrive: supported=%d enabled=%d version=%d\n", sup, en, ver);
    }

    ADLOD8InitSetting init;
    ADLOD8CurrentSetting cur;
    // Reading OD8 can also come up empty right after boot, but only a mutating
    // command is worth waiting for; info/watch should just report what is there.
    const bool mutating = (cmd == "set-power" || cmd == "set-clkmax" || cmd == "set" || cmd == "reset");
    bool readOk = false;
    for (int i = 0; i < (mutating ? 10 : 1) && !readOk; i++) {
        readOk = read_od8(target, init, cur);
        if (!readOk && mutating) { printf("  retrying OD8 query in 2s...\n"); Sleep(2000); }
    }
    if (!readOk) { adl_close(); return 1; }

    if (cmd == "info") {
        printf("\n=== OverDrive8 ===\n");
        print_caps(init.overdrive8Capabilities);
        printf("\n  %-30s %8s %8s %8s %8s\n", "setting", "min", "max", "default", "current");
        for (int i = 0; i < OD8_COUNT && i < init.count; i++) {
            const char* nm = od8_name(i);
            bool interesting = (nm != nullptr) ||
                               (init.od8SettingTable[i].maxValue != init.od8SettingTable[i].minValue);
            if (!interesting) continue;
            char buf[48];
            if (!nm) { snprintf(buf, sizeof(buf), "[id %d]", i); nm = buf; }
            printf("  %-30s %8d %8d %8d %8d\n", nm,
                   init.od8SettingTable[i].minValue, init.od8SettingTable[i].maxValue,
                   init.od8SettingTable[i].defaultValue, (i < cur.count) ? cur.Od8SettingTable[i] : 0);
        }

        // The two numbers that decide whether a 100 W target is even reachable.
        printf("\n  power limit range : %d%% .. %d%%  (default %d%%)\n",
               init.od8SettingTable[OD8_POWER_PERCENTAGE].minValue,
               init.od8SettingTable[OD8_POWER_PERCENTAGE].maxValue,
               init.od8SettingTable[OD8_POWER_PERCENTAGE].defaultValue);
        printf("  clock ceiling      : %d .. %d MHz  (default %d MHz)\n",
               init.od8SettingTable[OD8_GFXCLK_FMAX].minValue,
               init.od8SettingTable[OD8_GFXCLK_FMAX].maxValue,
               init.od8SettingTable[OD8_GFXCLK_FMAX].defaultValue);

        PmLog pm;
        if (pm.start(target)) {
            ADLPMLogDataOutput d;
            Sleep(1500);
            if (pm.read(target, d)) print_snapshot(d);
            pm.stop(target);
        } else {
            printf("\n  PMLog unavailable - no programmatic power readout\n");
        }
        adl_close();
        return 0;
    }

    if (cmd == "watch") {
        int secs = (argc > 2) ? atoi(argv[2]) : 10;
        PmLog pm;
        if (!pm.start(target)) { printf("PMLog unavailable\n"); adl_close(); return 1; }
        printf("\n=== sampling %d s ===\n", secs);
        printf("  %6s %10s %8s %8s %8s %8s %8s\n", "t(s)", "ASIC_raw", "GFXclk", "edge", "hotspot", "fanRPM", "volt");
        for (int t = 0; t < secs; t++) {
            Sleep(1000);
            ADLPMLogDataOutput d;
            if (!pm.read(target, d)) { printf("  read failed\n"); break; }
            auto g = [&](int id) { return (d.sensors[id].supported) ? d.sensors[id].value : -1; };
            printf("  %6d %10d %8d %8d %8d %8d %8d\n", t + 1,
                   g(ADL_PMLOG_ASIC_POWER), g(ADL_PMLOG_CLK_GFXCLK),
                   g(ADL_PMLOG_TEMPERATURE_EDGE), g(ADL_PMLOG_TEMPERATURE_HOTSPOT),
                   g(ADL_PMLOG_FAN_RPM), g(ADL_PMLOG_GFX_VOLTAGE));
            fflush(stdout);
        }
        pm.stop(target);
        adl_close();
        return 0;
    }

    // ---- mutating commands ----
    printf("\n=== applying ===\n");
    od8_power_mode_manual(target, cur);

    bool allOk = true;
    if (cmd == "set-power" && argc > 2) {
        allOk &= od8_apply(target, init, cur, OD8_POWER_PERCENTAGE, atoi(argv[2]), false);
    } else if (cmd == "set-clkmax" && argc > 2) {
        allOk &= od8_apply(target, init, cur, OD8_GFXCLK_FMAX, atoi(argv[2]), false);
    } else if (cmd == "reset") {
        allOk &= od8_apply(target, init, cur, OD8_POWER_PERCENTAGE, 0, true);
        allOk &= od8_apply(target, init, cur, OD8_GFXCLK_FMAX, 0, true);
    } else if (cmd == "set" && argc > 2) {
        std::string spec = argv[2];
        size_t pos = 0;
        while (pos < spec.size()) {
            size_t comma = spec.find(',', pos);
            std::string kv = spec.substr(pos, (comma == std::string::npos) ? std::string::npos : comma - pos);
            size_t eq = kv.find('=');
            if (eq != std::string::npos) {
                std::string k = kv.substr(0, eq), v = kv.substr(eq + 1);
                if (k == "power")       allOk &= od8_apply(target, init, cur, OD8_POWER_PERCENTAGE, atoi(v.c_str()), false);
                else if (k == "clkmax") allOk &= od8_apply(target, init, cur, OD8_GFXCLK_FMAX, atoi(v.c_str()), false);
                else printf("  unknown key '%s'\n", k.c_str());
            }
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    } else {
        printf("usage: %s info | watch <s> | set-power <pct> | set-clkmax <mhz> | set k=v,.. | reset\n", argv[0]);
        adl_close();
        return 1;
    }

    printf("\n=== readback ===\n");
    ADLOD8InitSetting init2;
    ADLOD8CurrentSetting cur2;
    if (read_od8(target, init2, cur2)) {
        printf("  power limit : %d%%\n", cur2.Od8SettingTable[OD8_POWER_PERCENTAGE]);
        printf("  clock max   : %d MHz\n", cur2.Od8SettingTable[OD8_GFXCLK_FMAX]);
    }

    adl_close();
    return allOk ? 0 : 2;
}
