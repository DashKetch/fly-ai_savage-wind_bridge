#!/usr/bin/env python3
"""Apply the Savage Wind Recomp integration hooks for botw_fly_bridge.

Run from the SavageWindRecomp repository root after copying runtime/ files.
All anchors are validated before any file is written, so a source mismatch does
not leave a partially patched checkout.
"""
from pathlib import Path
import sys


def replace_once(source: str, path: Path, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one integration anchor, found {count}: {old!r}")
    return source.replace(old, new, 1)


def main() -> int:
    root = Path.cwd()
    makefile = root / "Makefile"
    input_c = root / "runtime" / "hle" / "input.c"
    perfmon_c = root / "runtime" / "perfmon.c"
    header = root / "runtime" / "botw_fly_bridge.h"
    cpp = root / "runtime" / "botw_fly_bridge.cpp"

    for path in (makefile, input_c, perfmon_c, header, cpp):
        if not path.is_file():
            print(f"Missing {path}. Copy the bundle's runtime/ files into this repository first.", file=sys.stderr)
            return 2

    make_text = makefile.read_text(encoding="utf-8")
    input_text = input_c.read_text(encoding="utf-8")
    perfmon_text = perfmon_c.read_text(encoding="utf-8")
    if "botw_bridge_begin_frame" in input_text or "BOTW_BRIDGE_OBJ" in make_text:
        print("Bridge hooks already appear to be installed; refusing to patch twice.", file=sys.stderr)
        return 3

    # Preflight and transform in memory. No writes occur until every anchor matches.
    make_text = replace_once(
        make_text, makefile,
        "RT_OBJ   := $(patsubst runtime/%.c,$(B)/rt/%.o,$(RT_SRC))\n",
        "RT_OBJ   := $(patsubst runtime/%.c,$(B)/rt/%.o,$(RT_SRC))\n"
        "BOTW_BRIDGE_OBJ := $(B)/rt/botw_fly_bridge.o\n",
    )
    make_text = replace_once(
        make_text, makefile,
        "$(EXE): $(RT_OBJ) $(BACKEND_OBJ) $(UI_OBJ) $(IMGUI_OBJ)\n",
        "$(EXE): $(RT_OBJ) $(BOTW_BRIDGE_OBJ) $(BACKEND_OBJ) $(UI_OBJ) $(IMGUI_OBJ)\n",
    )
    make_text = replace_once(
        make_text, makefile,
        "runtime-objs: $(RT_OBJ) $(BACKEND_OBJ) $(UI_OBJ)\n",
        "runtime-objs: $(RT_OBJ) $(BOTW_BRIDGE_OBJ) $(BACKEND_OBJ) $(UI_OBJ)\n",
    )

    input_text = replace_once(
        input_text, input_c,
        '#include "../quick_drop_action.h"\n',
        '#include "../quick_drop_action.h"\n#include "../botw_fly_bridge.h"\n',
    )
    input_text = replace_once(
        input_text, input_c,
        "    g_pad_obj = 0; g_inject_plus = 0;\n",
        "    g_pad_obj = 0; g_inject_plus = 0;\n    botw_bridge_reset_input();\n",
    )
    input_text = replace_once(
        input_text, input_c,
        "void input_frame(uint64_t frame) {\n    extern uint64_t renderer_present_count(void);\n",
        "void input_frame(uint64_t frame) {\n"
        "    botw_bridge_begin_frame(frame);\n"
        "    extern uint64_t renderer_present_count(void);\n",
    )
    input_text = replace_once(
        input_text, input_c,
        "    b = quick_drop_action_filter_buttons(b);\n"
        "    // stick directions also appear as buttons (bits 16-23)\n",
        "    b = quick_drop_action_filter_buttons(b);\n"
        "    BotwFlyInput fly_input;\n"
        "    if (botw_bridge_get_input(&fly_input) && (fly_input.flags & BOTW_FLY_CMD_OVERRIDE)) {\n"
        "        bool safe_game_context = !in_menu && !framepace_in_event() && !framepace_dialogue_ui() && !pc_cursor_active();\n"
        "        if (safe_game_context || (fly_input.flags & BOTW_FLY_CMD_ALLOW_UI)) {\n"
        "            b = fly_input.buttons & 0xFFFFull;\n"
        "            lx = clamp_axis(fly_input.left_x);\n"
        "            ly = clamp_axis(fly_input.left_y);\n"
        "            rx = clamp_axis(fly_input.right_x);\n"
        "            ry = clamp_axis(fly_input.right_y);\n"
        "        }\n"
        "    }\n"
        "    // stick directions also appear as buttons (bits 16-23)\n",
    )

    perfmon_text = replace_once(
        perfmon_text, perfmon_c,
        '#include "rt.h"\n',
        '#include "rt.h"\n#include "botw_fly_bridge.h"\n',
    )
    perfmon_text = replace_once(
        perfmon_text, perfmon_c,
        "    pthread_mutex_unlock(&g_lock);\n}\nstatic uint64_t g_gen_at[512];",
        "    pthread_mutex_unlock(&g_lock);\n"
        "    /* PlayerInfo position is already bounds-checked by perfmon_position(). */\n"
        "    float link_position[3];\n"
        "    bool link_valid = perfmon_position(link_position);\n"
        "    botw_bridge_publish_link_state(now, link_position, link_valid ? 1 : 0);\n"
        "}\nstatic uint64_t g_gen_at[512];",
    )

    makefile.write_text(make_text, encoding="utf-8", newline="\n")
    input_c.write_text(input_text, encoding="utf-8", newline="\n")
    perfmon_c.write_text(perfmon_text, encoding="utf-8", newline="\n")
    print("Installed bridge integration hooks. Review the diff, then build with: make runtime-objs && make -j")
    print("Enable the pipe at launch with SAVAGEWIND_BOTW_BRIDGE=1.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Patch failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
