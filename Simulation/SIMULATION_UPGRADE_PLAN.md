# Simulation Upgrade Plan — Professional GUI Overhaul

**Feature:** Improvement 7 — Professional Simulation GUI
**Date:** 2026-03-17
**Status:** Planning
**Goal:** Upgrade the simulation from a dev-tool dashboard to a near-Webots-quality
professional demonstration, while keeping all existing functionality intact.

---

## 1. Current State Assessment

### What We Have (Good)
- Full working simulation with real production code
- 6 completed improvements (multi-speaker, 3D robot, demo console, fall detection,
  voice commands rework, object search + bring me)
- OpenCV-based 1280×800 dashboard with 6 panels
- 3D filled-polygon NAO model with orbit/zoom, posture animations, walk/dance
- Live camera feed with face/pose/YOLO overlays
- Real-time event console, audio bar, hotkey system
- ~30 FPS, pure CPU rendering

### What Needs Improvement
1. **OpenCV text rendering** — blocky HERSHEY fonts, no anti-aliasing, no Unicode
2. **Fixed window size** — 1280×800 hardcoded, no resize support
3. **No proper UI widgets** — no buttons, sliders, dropdowns (hotkeys only)
4. **Robot panel is small** (350×280) — hard to see the 3D model details
5. **No environment** — robot floats on a dark background, no room/floor/walls
6. **No object visualization** — YOLO finds objects in camera but the 3D scene
   doesn't show them
7. **No walk path visualization** — robot walks but you can't see where it's going
8. **Console is text-only** — no charts, graphs, or timeline visualization
9. **Single window** — can't detach panels, can't fullscreen the 3D view

---

## 2. Upgrade Strategy — Layered Approach

Instead of a full rewrite, we upgrade in **5 independent layers**. Each layer
is a standalone improvement that works without the others. This means:
- No risk of breaking the working simulation
- Each layer can be tested independently
- We can stop at any layer and still have improvements

```
Layer 5: 3D Environment (room, objects, path visualization)     ← Wow factor
Layer 4: Charts & Data Visualization (PID graphs, timelines)    ← Professional
Layer 3: Robot Panel Upgrade (larger, better lighting, shadows)  ← Visual quality
Layer 2: Resizable Layout + Panel Detach                        ← Usability
Layer 1: GUI Framework Migration (Dear PyGui)                   ← Foundation
```

**We build bottom-up: Layer 1 first, then each subsequent layer builds on it.**

---

## 3. Layer 1: GUI Framework Migration — Dear PyGui

### Why Dear PyGui (Not PyQt5)

| Feature | OpenCV (current) | PyQt5 | Dear PyGui |
|---------|-----------------|-------|------------|
| GPU acceleration | No | Limited (QOpenGLWidget) | **Yes (DirectX/OpenGL/Vulkan)** |
| Text rendering | Blocky HERSHEY | Good (Qt fonts) | **Excellent (TrueType, anti-aliased)** |
| Widgets | None | Full (buttons, sliders, menus) | **Full (imgui-style, GPU-rendered)** |
| Layout | Manual pixel math | QLayouts (complex) | **Flexible (docking, auto-resize)** |
| Learning curve | N/A (already using) | Medium (signals/slots) | **Low (immediate-mode, Pythonic)** |
| Installation | `pip install opencv-python` | `pip install PyQt5` | **`pip install dearpygui`** |
| Image display | Native (`cv2.imshow`) | QLabel+QPixmap | **`dpg.add_raw_texture` (GPU upload)** |
| OpenCV integration | Native | Need conversion | **Direct NumPy texture upload** |
| 3D rendering | Manual projection | QOpenGLWidget | **Built-in 3D drawlist API** |
| Docking/detach | No | QDockWidget (complex) | **Built-in docking system** |
| Theme/styling | Manual colors | QSS stylesheets | **Built-in themes + custom styling** |
| Frame rate | ~30 FPS (CPU) | ~60 FPS | **60+ FPS (GPU-accelerated)** |

**Dear PyGui wins because:**
1. GPU-accelerated rendering — smooth 60 FPS even with complex scenes
2. Built-in docking system — panels can be dragged, detached, rearranged
3. Immediate-mode paradigm — similar to our current "draw every frame" approach
4. Direct NumPy array → GPU texture upload (perfect for camera frames)
5. Built-in plot/chart widgets (for PID graphs, fall score timeline)
6. Anti-aliased TrueType fonts (professional text)
7. Single `pip install dearpygui` — no Qt/GTK dependencies
8. Python-native API — cleaner than PyQt5's signal/slot system

### What Changes in Layer 1

**Replace OpenCV window with Dear PyGui window:**

```python
# BEFORE (OpenCV):
canvas = np.full((800, 1280, 3), BG_COLOR, dtype=np.uint8)
cv2.putText(canvas, "FSM State:", ...)
cv2.rectangle(canvas, ...)
cv2.imshow("ElderGuard Simulation", canvas)
key = cv2.waitKey(33)

# AFTER (Dear PyGui):
import dearpygui.dearpygui as dpg

dpg.create_context()
dpg.create_viewport(title="ElderGuard Simulation", width=1440, height=900)

with dpg.window(label="ElderGuard Simulation", tag="main_window"):
    with dpg.group(horizontal=True):
        # Camera panel
        with dpg.child_window(width=700, height=480):
            dpg.add_image("camera_texture")
        # State panel
        with dpg.child_window(width=-1, height=480):
            dpg.add_text("FSM State:")
            dpg.add_text("[IDLE]", tag="fsm_badge", color=(0, 200, 80))
            ...

dpg.setup_dearpygui()
dpg.show_viewport()
while dpg.is_dearpygui_running():
    # Update textures, text, colors from SharedSimState
    update_gui(shared_state.snapshot())
    dpg.render_dearpygui_frame()
```

**Key migration steps:**

1. **Camera feed:** OpenCV frame → `dpg.set_value("camera_texture", frame_data)` — GPU texture upload, zero-copy
2. **Text elements:** `cv2.putText` → `dpg.set_value("fsm_text", "[IDLE]")` — clean API
3. **Color-coded badges:** `dpg.configure_item("fsm_badge", color=GREEN)` — dynamic colors
4. **Panels:** OpenCV rectangles → Dear PyGui child windows with borders
5. **Hotkeys:** `cv2.waitKey` → `dpg.set_key_callback` — per-key handlers
6. **Mouse:** `cv2.setMouseCallback` → `dpg.set_item_callback` per widget

### Files Created/Modified

| File | Change |
|------|--------|
| `gui/dpg_dashboard.py` | **NEW** — Dear PyGui main window, layout, update loop |
| `gui/dpg_panels.py` | **NEW** — Panel creation functions (camera, state, robot, console) |
| `gui/dpg_theme.py` | **NEW** — Colors, fonts, styling constants |
| `gui/dashboard.py` | KEEP — renamed to `dashboard_opencv.py` as fallback |
| `gui/panels.py` | KEEP — renamed to `panels_opencv.py` as fallback |
| `run_simulation.py` | Modify — import `dpg_dashboard` instead of `dashboard` |
| `requirements_simulation.txt` | Add `dearpygui>=2.0` |

### Fallback Strategy

The old OpenCV dashboard is kept as `dashboard_opencv.py`. A CLI flag selects:
```bash
python run_simulation.py                  # Uses Dear PyGui (new default)
python run_simulation.py --legacy-gui     # Falls back to OpenCV dashboard
```

---

## 4. Layer 2: Resizable Layout + Docking

### What Changes

Dear PyGui has a built-in **docking system** (like VS Code). Panels become
dockable windows that can be:
- Dragged to new positions
- Detached into separate floating windows
- Resized independently
- Collapsed/expanded
- Tabbed (stack multiple panels in one area)

```python
dpg.configure_app(docking=True, docking_space=True)

# Each panel becomes a dockable window:
with dpg.window(label="Camera Feed", tag="camera_window"):
    dpg.add_image("camera_texture")

with dpg.window(label="Robot State", tag="state_window"):
    ...

with dpg.window(label="3D Robot", tag="robot_window"):
    ...

with dpg.window(label="Console", tag="console_window"):
    ...
```

**Default layout** matches the current 4-panel arrangement but is user-resizable.

### Benefits
- User can enlarge the 3D robot view by dragging the panel border
- Console can be detached to a second monitor
- Camera feed can be fullscreened for face-tracking demos
- Layout is saved between sessions

---

## 5. Layer 3: Robot Panel Upgrade

### What Changes

Replace the OpenCV-projected 3D wireframe with Dear PyGui's **3D drawlist**
or a **PyBullet viewport** embedded in the GUI:

**Option A: Dear PyGui 3D Drawlist (simpler, integrated)**
- Use `dpg.draw_line_3d`, `dpg.draw_triangle_3d` for filled 3D primitives
- Built-in camera orbit, zoom, pan
- Ground plane grid with perspective
- Shadow projection (project joints onto y=0 plane)
- Ambient/directional lighting simulation via face normals

**Option B: PyBullet viewport (realistic physics)**
- `pip install pybullet`
- Load NAO URDF model (available from community repos)
- Real joint articulation with physics
- Render to offscreen buffer → upload as texture to Dear PyGui
- 3D environment with floor, objects, walls

**Recommendation: Option A for Layer 3** (integrated, no external dependency).
Option B can be added as Layer 5.

### Visual Improvements
- **Anti-aliased 3D lines** (GPU-rendered, smooth)
- **Proper lighting** — ambient + directional, face brightness from normals
- **Smooth animations** — 60 FPS interpolation (vs current 30 FPS)
- **Ground plane** — checkered floor with perspective grid
- **Robot shadow** — projected shadow on the ground
- **Coordinate axes** — XYZ arrows showing robot orientation
- **Trail visualization** — dots showing robot's walk path over time
- **Larger default panel** — 600×500 instead of 350×280

---

## 6. Layer 4: Charts & Data Visualization

### What Changes

Dear PyGui has built-in **plot widgets** (based on ImPlot). Add real-time charts:

**Chart 1: PID Error Over Time** (line graph)
```
Shows error_x (blue) and error_y (red) over last 30 seconds.
X-axis: time, Y-axis: normalized error [-1, 1].
Useful for analyzing servo tracking quality.
```

**Chart 2: Fall Score Timeline** (area chart)
```
Shows fall_score (0.0–1.0) over last 60 seconds.
Horizontal line at 0.6 (trigger threshold).
Red shading above threshold.
Shows exactly when falls are detected and recovered.
```

**Chart 3: Audio Level History** (bar graph)
```
Rolling 10-second mic level display.
Shows audio activity patterns.
Highlights when speech is detected.
```

**Chart 4: System State Timeline** (Gantt-style)
```
Horizontal bars showing FSM state durations:
IDLE (green) | LISTENING (yellow) | EXECUTING (blue) | SEARCHING (orange)
Last 60 seconds, scrolling.
```

### Implementation
```python
with dpg.window(label="Analytics", tag="analytics_window"):
    with dpg.plot(label="PID Error", height=200, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
        with dpg.plot_axis(dpg.mvYAxis, label="Error"):
            dpg.add_line_series([], [], label="error_x", tag="pid_x_series")
            dpg.add_line_series([], [], label="error_y", tag="pid_y_series")

    with dpg.plot(label="Fall Score", height=150, width=-1):
        dpg.add_plot_axis(dpg.mvXAxis, label="Time (s)")
        with dpg.plot_axis(dpg.mvYAxis, label="Score"):
            dpg.add_shade_series([], [], [], tag="fall_series")
            dpg.add_hline_series([0.6], label="Threshold")
```

---

## 7. Layer 5: 3D Environment (Room + Objects)

### What Changes

Add a virtual room around the robot for spatial context:

**Room elements:**
- Floor (textured or checkered, 4m × 4m)
- Walls (transparent or wireframe, suggest boundaries)
- Furniture outlines (chair, table — simple boxes)
- Objects on the floor (phone, bottle, cup — when found by YOLO)

**Robot in the room:**
- Robot position tracked from walk commands (x, y, theta)
- Walk path visualized as a dotted trail
- Person position estimated from face detection (relative to robot)
- Object positions placed when YOLO detects them

**How robot position is tracked:**
```python
# In mock_proxies.py or a new tracker:
class RobotPositionTracker:
    def __init__(self):
        self.x = 2.0      # meters, center of room
        self.y = 2.0
        self.theta = 0.0   # radians, facing +Y

    def update_from_walk(self, vx, vy, vtheta, dt):
        """Called from mock_proxies when walk_toward/set_walk_velocity executes."""
        self.theta += vtheta * dt
        self.x += vx * math.cos(self.theta) * dt
        self.y += vx * math.sin(self.theta) * dt
```

**Person visualization:**
- Blue dot showing estimated person position (derived from face bbox + head angle)
- Line from robot to person (tracking line)
- When person is "lost" (face not detected), last known position shown as faded dot

**Object visualization:**
- When YOLO detects an object, place a colored marker in the 3D scene
- Object position estimated from head angle at time of detection
- Shows object name label above the marker
- Marker persists until object is picked up or a new search is done

### Implementation Approach

All 3D scene rendering uses Dear PyGui's drawlist API:
```python
with dpg.drawlist(width=800, height=600, tag="scene_3d"):
    # Floor grid
    for i in range(-4, 5):
        dpg.draw_line_3d([i, -4, 0], [i, 4, 0], color=GRID_COLOR)
        dpg.draw_line_3d([-4, i, 0], [4, i, 0], color=GRID_COLOR)

    # Robot (at tracked position)
    draw_nao_at(robot_pos.x, robot_pos.y, robot_pos.theta, joints)

    # Person (estimated position)
    if person_visible:
        dpg.draw_circle_3d(person_pos, radius=0.15, color=BLUE)

    # Found objects
    for obj in found_objects:
        dpg.draw_circle_3d(obj.pos, radius=0.05, color=CYAN)
        dpg.draw_text_3d(obj.pos + [0,0,0.1], obj.name)
```

---

## 8. Implementation Order

| Layer | What | Effort | Dependencies | Impact |
|-------|------|--------|-------------|--------|
| **1** | Dear PyGui migration (replace OpenCV window) | 3-4 days | None | Foundation for everything |
| **2** | Docking + resizable panels | 1 day | Layer 1 | Usability |
| **3** | Robot panel upgrade (better 3D, lighting) | 2 days | Layer 1 | Visual quality |
| **4** | Charts + data visualization | 1-2 days | Layer 1 | Professional analytics |
| **5** | 3D environment (room, objects, paths) | 2-3 days | Layers 1+3 | Wow factor |

**Total: ~10-12 days for all 5 layers.**

Each layer is independently functional. You can ship after any layer.

---

## 9. Files Plan

### New Files

| File | Layer | Purpose |
|------|-------|---------|
| `gui/dpg_dashboard.py` | 1 | Main Dear PyGui window, viewport, render loop |
| `gui/dpg_panels.py` | 1 | Panel creation (camera, state, console, audio) |
| `gui/dpg_theme.py` | 1 | Colors, fonts, styling, dark theme |
| `gui/dpg_robot_view.py` | 3 | Upgraded 3D robot renderer using Dear PyGui drawlists |
| `gui/dpg_charts.py` | 4 | Real-time charts (PID error, fall score, audio) |
| `gui/dpg_scene.py` | 5 | 3D room environment renderer |
| `gui/robot_position.py` | 5 | Robot world-position tracker from walk commands |

### Modified Files

| File | Layer | Change |
|------|-------|--------|
| `run_simulation.py` | 1 | Switch dashboard import, add `--legacy-gui` flag |
| `requirements_simulation.txt` | 1 | Add `dearpygui>=2.0` |
| `mock_nao/mock_proxies.py` | 5 | Track robot world position from walk commands |

### Preserved Files (Fallback)

| File | Renamed To | Purpose |
|------|-----------|---------|
| `gui/dashboard.py` | `gui/dashboard_opencv.py` | Legacy OpenCV fallback |
| `gui/panels.py` | `gui/panels_opencv.py` | Legacy OpenCV panels |
| `gui/robot_visualizer.py` | `gui/robot_visualizer_opencv.py` | Legacy 3D renderer |

---

## 10. Feature Parity Checklist

Every feature in the current simulation MUST work in the new GUI:

### Camera Panel
- [ ] Live camera feed display (NumPy → GPU texture)
- [ ] Face detection green box overlay
- [ ] YOLO detection cyan boxes (up to 5)
- [ ] Pose skeleton overlay (green/red)
- [ ] Fall detected banner (flashing red)
- [ ] "No Camera Feed" placeholder

### State Panel
- [ ] FSM state badge with color
- [ ] Posture display
- [ ] 4 channel states with color dots (HEAD/LEGS/SPEECH/ARMS)
- [ ] Head yaw/pitch values
- [ ] PID error display
- [ ] Servo mode (OFF/HEAD ONLY/FOLLOWING/SEARCHING)
- [ ] Fall monitor state with color dot
- [ ] Fall score display
- [ ] Voice commands reference list

### Robot Panel
- [ ] 3D NAO model with filled body parts
- [ ] Head rotation from yaw/pitch
- [ ] Standing/sitting/resting/fallen postures
- [ ] Walk animation (leg stride + body sway)
- [ ] Dance animation
- [ ] Wave animation
- [ ] Posture transition animation (lerp)
- [ ] Mouse orbit (drag to rotate view)
- [ ] Mouse zoom (scroll wheel)
- [ ] Ground plane/shadow
- [ ] FALLEN!/RESTING status labels

### Console Panel
- [ ] Scrolling event log from SimEventBus
- [ ] Color-coded categories (STT/VERIFY/FSM/CMD/NAO/SYS)
- [ ] Severity overrides (error=red, warning=yellow)
- [ ] Timestamp + category tag + message format
- [ ] Auto-scroll to latest event

### Audio Bar
- [ ] Mic level bar (green, 0-100%)
- [ ] STT recognized text display
- [ ] Speaker verification score + OK/REJ badge

### Hotkeys
- [ ] All 10 command injection keys (1-0)
- [ ] f/p/r/d/q simulation control keys
- [ ] Mouse orbit restricted to robot panel

### Header
- [ ] Title "ELDERGUARD SIMULATION"
- [ ] FSM state badge
- [ ] Uptime counter (HH:MM:SS)

---

## 11. Dear PyGui Specific Advantages

### Camera Feed (Zero-Copy GPU Upload)
```python
# Create texture once:
with dpg.texture_registry():
    dpg.add_raw_texture(640, 480, default_value=data,
                        format=dpg.mvFormat_Float_rgb, tag="cam_tex")

# Update every frame (NumPy → GPU, no encoding):
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
frame_float = frame_rgb.astype(np.float32) / 255.0
dpg.set_value("cam_tex", frame_float.flatten())
```

### Anti-Aliased Text
```python
# Load a professional font:
with dpg.font_registry():
    dpg.add_font("fonts/Roboto-Regular.ttf", 16, tag="main_font")
    dpg.add_font("fonts/RobotoMono-Regular.ttf", 14, tag="mono_font")
dpg.bind_font("main_font")
```

### Dark Theme (Built-In)
```python
with dpg.theme() as dark_theme:
    with dpg.theme_component(dpg.mvAll):
        dpg.add_theme_color(dpg.mvThemeCol_WindowBg, (30, 30, 30))
        dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (45, 45, 45))
        dpg.add_theme_color(dpg.mvThemeCol_Text, (224, 224, 224))
        dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 70, 70))
dpg.bind_theme(dark_theme)
```

### Docking (Built-In)
```python
dpg.configure_app(docking=True, docking_space=True)
# Panels automatically become dockable — users can rearrange freely
```

---

## 12. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Dear PyGui doesn't support Python 3.9 | Low | High | Check compatibility; PyGui supports 3.8-3.12 |
| Camera texture upload is slow | Low | Medium | Float32 conversion is fast; GPU upload is <1ms for 640×480 |
| 3D drawlist can't match current robot quality | Medium | Medium | Dear PyGui drawlists are GPU-accelerated — should be better |
| Docking system is complex to set up defaults | Low | Low | Save/load layout via `dpg.save_init_file` |
| Migration breaks existing hotkeys | Low | High | Test every hotkey after migration |
| OpenCV still needed for camera/face/YOLO | None | None | OpenCV remains for vision; only the GUI changes |
| Legacy GUI fallback has bit-rot | Medium | Low | Keep both paths tested; CI runs both |

---

## 13. Compatibility

- Dear PyGui runs on **Windows, Linux, macOS** (same as current)
- All production code (`rpi_brain/`, `nao_body/`) is **unchanged** — only GUI layer changes
- `SharedSimState` and `SimEventBus` are **unchanged** — new GUI reads same data
- `run_simulation.py` hooks are **unchanged** — same brain patching
- Camera, face tracking, YOLO, pose estimation all continue using **OpenCV internally**
  (Dear PyGui only replaces the display window, not the vision pipeline)
- `--legacy-gui` flag preserves the old OpenCV dashboard for comparison/fallback
