# --- Deep Live Cam UI - Redesigned ---

import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from cv2_enumerate_cameras import enumerate_cameras
from PIL import Image, ImageOps
import time
import json
import threading
import queue
import torch
import modules.globals
import modules.metadata
from modules.face_analyser import (
    get_one_face,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager
from modules import globals
import platform

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# Global references
ROOT = None
PREVIEW = None
POPUP = None
POPUP_LIVE = None

# Sizing
ROOT_WIDTH = 1000
ROOT_HEIGHT = 700
SIDEBAR_WIDTH = 320
PREVIEW_MIN_WIDTH = 640
PREVIEW_MIN_HEIGHT = 480

# Preview references
preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.poisson_blend = switch_states.get("poisson_blend", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask = switch_states.get("mouth_mask", False)
        modules.globals.show_mouth_mask_box = switch_states.get("show_mouth_mask_box", False)
    except FileNotFoundError:
        pass


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:
    global source_label, target_label, status_label

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.title(f"{modules.metadata.name} {modules.metadata.version} {modules.metadata.edition}")
    root.geometry(f"{ROOT_WIDTH}x{ROOT_HEIGHT}")
    root.minsize(800, 600)
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # Configure grid weights
    root.grid_columnconfigure(0, weight=0, minsize=SIDEBAR_WIDTH)  # Sidebar
    root.grid_columnconfigure(1, weight=1)  # Main preview area
    root.grid_rowconfigure(0, weight=1)

    # === LEFT SIDEBAR ===
    sidebar = ctk.CTkScrollableFrame(root, width=SIDEBAR_WIDTH - 20)
    sidebar.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

    # --- Source/Target Section ---
    section_source = ctk.CTkFrame(sidebar)
    section_source.pack(fill="x", padx=5, pady=5)

    ctk.CTkLabel(section_source, text="Source & Target", font=("", 14, "bold")).pack(pady=(5, 10))

    # Source/Target images frame
    images_frame = ctk.CTkFrame(section_source, fg_color="transparent")
    images_frame.pack(fill="x", padx=5)

    # Source
    source_frame = ctk.CTkFrame(images_frame)
    source_frame.pack(side="left", expand=True, fill="both", padx=2)
    source_label = ctk.CTkLabel(source_frame, text="No Source", width=120, height=120)
    source_label.pack(pady=5)
    ctk.CTkButton(source_frame, text="Select Face", width=120, height=28,
                  command=select_source_path).pack(pady=5)

    # Swap button
    ctk.CTkButton(images_frame, text="⇄", width=30, height=30,
                  command=swap_faces_paths).pack(side="left", padx=5)

    # Target
    target_frame = ctk.CTkFrame(images_frame)
    target_frame.pack(side="left", expand=True, fill="both", padx=2)
    target_label = ctk.CTkLabel(target_frame, text="No Target", width=120, height=120)
    target_label.pack(pady=5)
    ctk.CTkButton(target_frame, text="Select Target", width=120, height=28,
                  command=select_target_path).pack(pady=5)

    # --- Processing Options Section ---
    section_process = ctk.CTkFrame(sidebar)
    section_process.pack(fill="x", padx=5, pady=5)

    ctk.CTkLabel(section_process, text="Processing", font=("", 14, "bold")).pack(pady=(5, 10))

    options_frame = ctk.CTkFrame(section_process, fg_color="transparent")
    options_frame.pack(fill="x", padx=10)

    # Row 1
    row1 = ctk.CTkFrame(options_frame, fg_color="transparent")
    row1.pack(fill="x", pady=2)

    many_faces_var = ctk.BooleanVar(value=modules.globals.many_faces)
    ctk.CTkCheckBox(row1, text="Many Faces", variable=many_faces_var, width=130,
                    command=lambda: (setattr(modules.globals, "many_faces", many_faces_var.get()),
                                    save_switch_states())).pack(side="left")

    map_faces_var = ctk.BooleanVar(value=modules.globals.map_faces)
    ctk.CTkCheckBox(row1, text="Map Faces", variable=map_faces_var, width=130,
                    command=lambda: (setattr(modules.globals, "map_faces", map_faces_var.get()),
                                    save_switch_states(),
                                    close_mapper_window() if not map_faces_var.get() else None)).pack(side="left")

    # Row 2
    row2 = ctk.CTkFrame(options_frame, fg_color="transparent")
    row2.pack(fill="x", pady=2)

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    ctk.CTkCheckBox(row2, text="Mouth Mask", variable=mouth_mask_var, width=130,
                    command=lambda: setattr(modules.globals, "mouth_mask", mouth_mask_var.get())).pack(side="left")

    poisson_var = ctk.BooleanVar(value=modules.globals.poisson_blend)
    ctk.CTkCheckBox(row2, text="Poisson Blend", variable=poisson_var, width=130,
                    command=lambda: (setattr(modules.globals, "poisson_blend", poisson_var.get()),
                                    save_switch_states())).pack(side="left")

    # Row 3
    row3 = ctk.CTkFrame(options_frame, fg_color="transparent")
    row3.pack(fill="x", pady=2)

    color_fix_var = ctk.BooleanVar(value=modules.globals.color_correction)
    ctk.CTkCheckBox(row3, text="Fix Blue Cam", variable=color_fix_var, width=130,
                    command=lambda: (setattr(modules.globals, "color_correction", color_fix_var.get()),
                                    save_switch_states())).pack(side="left")

    show_fps_var = ctk.BooleanVar(value=modules.globals.show_fps)
    ctk.CTkCheckBox(row3, text="Show FPS", variable=show_fps_var, width=130,
                    command=lambda: (setattr(modules.globals, "show_fps", show_fps_var.get()),
                                    save_switch_states())).pack(side="left")

    # Sliders
    sliders_frame = ctk.CTkFrame(options_frame, fg_color="transparent")
    sliders_frame.pack(fill="x", pady=(10, 5))

    # Transparency
    ctk.CTkLabel(sliders_frame, text="Opacity:", width=70, anchor="w").grid(row=0, column=0, sticky="w")
    opacity_slider = ctk.CTkSlider(sliders_frame, from_=0, to=1, width=180,
                                   command=lambda v: setattr(modules.globals, "opacity", float(v)))
    opacity_slider.set(getattr(modules.globals, 'opacity', 1.0))
    opacity_slider.grid(row=0, column=1, padx=5, pady=2)

    # Sharpness
    ctk.CTkLabel(sliders_frame, text="Sharpness:", width=70, anchor="w").grid(row=1, column=0, sticky="w")
    sharpness_slider = ctk.CTkSlider(sliders_frame, from_=0, to=5, width=180,
                                     command=lambda v: setattr(modules.globals, "sharpness", float(v)))
    sharpness_slider.set(getattr(modules.globals, 'sharpness', 0.0))
    sharpness_slider.grid(row=1, column=1, padx=5, pady=2)

    # --- Enhancement Section ---
    section_enhance = ctk.CTkFrame(sidebar)
    section_enhance.pack(fill="x", padx=5, pady=5)

    ctk.CTkLabel(section_enhance, text="Face Enhancement", font=("", 14, "bold")).pack(pady=(5, 10))

    enhance_frame = ctk.CTkFrame(section_enhance, fg_color="transparent")
    enhance_frame.pack(fill="x", padx=10)

    # Enable switch
    enhancer_var = ctk.BooleanVar(value=modules.globals.fp_ui.get("face_enhancer", False))
    ctk.CTkCheckBox(enhance_frame, text="Enable Enhancement", variable=enhancer_var,
                    command=lambda: (update_tumbler("face_enhancer", enhancer_var.get()),
                                    save_switch_states())).pack(anchor="w", pady=2)

    # Enhancer type dropdown
    type_frame = ctk.CTkFrame(enhance_frame, fg_color="transparent")
    type_frame.pack(fill="x", pady=5)

    ctk.CTkLabel(type_frame, text="Type:", width=50, anchor="w").pack(side="left")

    type_map = {"gfpgan_onnx": "GFPGAN_ONNX", "codeformer": "CodeFormer", "gfpgan": "GFPGAN",
                "gpen": "GPEN", "fast": "Fast", "none": "None"}
    type_map_rev = {v: k for k, v in type_map.items()}

    enhancer_dropdown = ctk.CTkOptionMenu(
        type_frame, width=150,
        values=["GFPGAN_ONNX", "CodeFormer", "GFPGAN", "GPEN", "Fast", "None"],
        command=lambda v: setattr(modules.globals, "enhancer_type", type_map_rev.get(v, "gfpgan_onnx"))
    )
    enhancer_dropdown.set(type_map.get(getattr(modules.globals, 'enhancer_type', 'gfpgan_onnx'), "GFPGAN_ONNX"))
    enhancer_dropdown.pack(side="left", padx=5)

    # Blend slider
    blend_frame = ctk.CTkFrame(enhance_frame, fg_color="transparent")
    blend_frame.pack(fill="x", pady=2)

    ctk.CTkLabel(blend_frame, text="Blend:", width=50, anchor="w").pack(side="left")
    blend_slider = ctk.CTkSlider(blend_frame, from_=0, to=1, width=150,
                                 command=lambda v: setattr(modules.globals, "enhancer_blend", float(v)))
    blend_slider.set(getattr(modules.globals, 'enhancer_blend', 0.8))
    blend_slider.pack(side="left", padx=5)

    # FP16 toggle
    fp16_var = ctk.BooleanVar(value=getattr(modules.globals, 'use_fp16', True))
    ctk.CTkCheckBox(enhance_frame, text="FP16 (Faster)", variable=fp16_var,
                    command=lambda: setattr(modules.globals, "use_fp16", fp16_var.get())).pack(anchor="w", pady=2)

    # --- Camera Section ---
    section_camera = ctk.CTkFrame(sidebar)
    section_camera.pack(fill="x", padx=5, pady=5)

    ctk.CTkLabel(section_camera, text="Live Camera", font=("", 14, "bold")).pack(pady=(5, 10))

    camera_frame = ctk.CTkFrame(section_camera, fg_color="transparent")
    camera_frame.pack(fill="x", padx=10, pady=5)

    # Camera dropdown
    camera_indices, camera_names = get_available_cameras()
    camera_var = ctk.StringVar(value=camera_names[0] if camera_names else "No cameras")

    ctk.CTkLabel(camera_frame, text="Camera:", width=60, anchor="w").pack(side="left")
    camera_dropdown = ctk.CTkOptionMenu(camera_frame, variable=camera_var, width=180,
                                        values=camera_names if camera_names else ["No cameras"])
    camera_dropdown.pack(side="left", padx=5)

    # Live options
    live_opts = ctk.CTkFrame(section_camera, fg_color="transparent")
    live_opts.pack(fill="x", padx=10, pady=5)

    mirror_var = ctk.BooleanVar(value=modules.globals.live_mirror)
    ctk.CTkCheckBox(live_opts, text="Mirror", variable=mirror_var, width=80,
                    command=lambda: (setattr(modules.globals, "live_mirror", mirror_var.get()),
                                    save_switch_states())).pack(side="left")

    # --- Output Options Section ---
    section_output = ctk.CTkFrame(sidebar)
    section_output.pack(fill="x", padx=5, pady=5)

    ctk.CTkLabel(section_output, text="Output Options", font=("", 14, "bold")).pack(pady=(5, 10))

    output_frame = ctk.CTkFrame(section_output, fg_color="transparent")
    output_frame.pack(fill="x", padx=10)

    keep_fps_var = ctk.BooleanVar(value=modules.globals.keep_fps)
    ctk.CTkCheckBox(output_frame, text="Keep FPS", variable=keep_fps_var, width=100,
                    command=lambda: (setattr(modules.globals, "keep_fps", keep_fps_var.get()),
                                    save_switch_states())).pack(side="left")

    keep_audio_var = ctk.BooleanVar(value=modules.globals.keep_audio)
    ctk.CTkCheckBox(output_frame, text="Keep Audio", variable=keep_audio_var, width=100,
                    command=lambda: (setattr(modules.globals, "keep_audio", keep_audio_var.get()),
                                    save_switch_states())).pack(side="left")

    keep_frames_var = ctk.BooleanVar(value=modules.globals.keep_frames)
    ctk.CTkCheckBox(output_frame, text="Keep Frames", variable=keep_frames_var, width=100,
                    command=lambda: (setattr(modules.globals, "keep_frames", keep_frames_var.get()),
                                    save_switch_states())).pack(side="left")

    # === RIGHT MAIN AREA ===
    main_frame = ctk.CTkFrame(root)
    main_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    main_frame.grid_columnconfigure(0, weight=1)
    main_frame.grid_rowconfigure(0, weight=1)

    # Preview area (placeholder)
    preview_frame = ctk.CTkFrame(main_frame, fg_color="#1a1a1a")
    preview_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    preview_frame.grid_columnconfigure(0, weight=1)
    preview_frame.grid_rowconfigure(0, weight=1)

    preview_placeholder = ctk.CTkLabel(preview_frame, text="Preview will appear here\n\nSelect source and target, then click Preview or Live",
                                       font=("", 16), text_color="#666666")
    preview_placeholder.grid(row=0, column=0)

    # Action buttons
    buttons_frame = ctk.CTkFrame(main_frame, fg_color="transparent", height=50)
    buttons_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

    ctk.CTkButton(buttons_frame, text="▶ Start Processing", width=150, height=40, fg_color="#28a745",
                  hover_color="#218838", command=lambda: analyze_target(start, root)).pack(side="left", padx=5)

    ctk.CTkButton(buttons_frame, text="👁 Preview", width=120, height=40,
                  command=toggle_preview).pack(side="left", padx=5)

    ctk.CTkButton(buttons_frame, text="📹 Live", width=120, height=40, fg_color="#007bff",
                  hover_color="#0056b3",
                  command=lambda: webcam_preview(root, camera_indices[camera_names.index(camera_var.get())]
                                                 if camera_names and camera_var.get() in camera_names else None)
                  ).pack(side="left", padx=5)

    ctk.CTkButton(buttons_frame, text="✕ Exit", width=100, height=40, fg_color="#dc3545",
                  hover_color="#c82333", command=destroy).pack(side="right", padx=5)

    # Status bar
    status_frame = ctk.CTkFrame(main_frame, height=30)
    status_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=(0, 5))

    status_label = ctk.CTkLabel(status_frame, text="Ready", anchor="w")
    status_label.pack(side="left", padx=10)

    version_label = ctk.CTkLabel(status_frame, text=f"v{modules.metadata.version}", anchor="e",
                                 text_color="#666666")
    version_label.pack(side="right", padx=10)

    return root


def close_mapper_window():
    global POPUP, POPUP_LIVE
    if POPUP and POPUP.winfo_exists():
        POPUP.destroy()
        POPUP = None
    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.destroy()
        POPUP_LIVE = None


def create_modal_dialog(parent, title, width, height):
    """Create a modal dialog that appears in the foreground."""
    dialog = ctk.CTkToplevel(parent)
    dialog.title(title)
    dialog.geometry(f"{width}x{height}")

    # Make it modal and bring to front
    dialog.transient(parent)
    dialog.grab_set()
    dialog.focus_force()
    dialog.lift()

    # Center on parent
    dialog.update_idletasks()
    x = parent.winfo_x() + (parent.winfo_width() - width) // 2
    y = parent.winfo_y() + (parent.winfo_height() - height) // 2
    dialog.geometry(f"+{x}+{y}")

    return dialog


def analyze_target(start: Callable[[], None], root: ctk.CTk):
    if POPUP is not None and POPUP.winfo_exists():
        POPUP.focus_force()
        POPUP.lift()
        update_status("Please complete the popup or close it.")
        return

    if modules.globals.map_faces:
        modules.globals.source_target_map = []

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces...")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces...")
            get_unique_faces_from_target_video()

        if len(modules.globals.source_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.source_target_map)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def create_source_target_popup(start: Callable[[], None], root: ctk.CTk, map: list) -> None:
    global POPUP, popup_status_label

    POPUP = create_modal_dialog(root, _("Source ↔ Target Mapper"), 700, 600)

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            popup_status_label.configure(text="At least 1 source with target is required!")

    # Header
    ctk.CTkLabel(POPUP, text="Map source faces to target faces", font=("", 14)).pack(pady=10)

    # Scrollable content
    scrollable = ctk.CTkScrollableFrame(POPUP, width=660, height=450)
    scrollable.pack(fill="both", expand=True, padx=10, pady=5)

    def on_button_click(map, button_num):
        update_popup_source(scrollable, map, button_num)

    for item in map:
        id = item["id"]

        row_frame = ctk.CTkFrame(scrollable)
        row_frame.pack(fill="x", pady=5, padx=5)

        ctk.CTkButton(row_frame, text="Select Source", width=120,
                      command=lambda id=id: on_button_click(map, id)).pack(side="left", padx=5)

        ctk.CTkLabel(row_frame, text="→", width=30).pack(side="left")

        # Target preview
        target_img = Image.fromarray(cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        target_img = target_img.resize((80, 80), Image.LANCZOS)
        target_ctk = ctk.CTkImage(target_img, size=(80, 80))
        ctk.CTkLabel(row_frame, image=target_ctk, text="").pack(side="left", padx=5)

    # Footer
    footer = ctk.CTkFrame(POPUP, fg_color="transparent")
    footer.pack(fill="x", padx=10, pady=10)

    popup_status_label = ctk.CTkLabel(footer, text="", text_color="#ff6b6b")
    popup_status_label.pack(side="left")

    ctk.CTkButton(footer, text="Submit", width=100, command=lambda: on_submit_click(start)).pack(side="right")


def update_popup_source(scrollable_frame, map, item_id):
    global source_label_dict, RECENT_DIRECTORY_SOURCE, POPUP

    # Find item by ID, not by index
    item = next((x for x in map if x["id"] == item_id), None)
    if item is None:
        return map

    # Temporarily release grab so file dialog works properly
    if POPUP and POPUP.winfo_exists():
        POPUP.grab_release()

    source_path = ctk.filedialog.askopenfilename(
        title="Select source image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
        parent=POPUP,
    )

    # Restore grab
    if POPUP and POPUP.winfo_exists():
        POPUP.grab_set()

    if "source" in item:
        item.pop("source")
        if item_id in source_label_dict:
            source_label_dict[item_id].destroy()
            del source_label_dict[item_id]

    if source_path == "":
        return map

    RECENT_DIRECTORY_SOURCE = os.path.dirname(source_path)
    cv2_img = cv2.imread(source_path)
    face = get_one_face(cv2_img)

    if face:
        x_min, y_min, x_max, y_max = face["bbox"]
        item["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }
        update_status("Source face added")
    else:
        update_status("No face detected in source image")

    return map


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title("Preview")
    preview.protocol("WM_DELETE_WINDOW", toggle_preview)
    preview.resizable(True, True)
    preview.minsize(640, 480)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(preview, from_=0, to=0,
                                   command=lambda v: update_preview(int(v)))

    return preview


def update_status(text: str) -> None:
    if status_label:
        status_label.configure(text=text)
        # Don't call update_idletasks() here - let the event loop handle it naturally


def update_pop_status(text: str) -> None:
    if popup_status_label:
        popup_status_label.configure(text=text)


def update_pop_live_status(text: str) -> None:
    if popup_status_label_live:
        popup_status_label_live.configure(text=text)


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE

    if PREVIEW:
        PREVIEW.withdraw()

    source_path = ctk.filedialog.askopenfilename(
        title="Select source face image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(source_path)
        image = render_image_preview(source_path, (120, 120))
        source_label.configure(image=image, text="")
        update_status(f"Source: {os.path.basename(source_path)}")
    else:
        modules.globals.source_path = None
        source_label.configure(image=None, text="No Source")


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not is_image(source_path) or not is_image(target_path):
        update_status("Both source and target must be images to swap")
        return

    modules.globals.source_path = target_path
    modules.globals.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)

    if PREVIEW:
        PREVIEW.withdraw()

    source_image = render_image_preview(modules.globals.source_path, (120, 120))
    source_label.configure(image=source_image, text="")

    target_image = render_image_preview(modules.globals.target_path, (120, 120))
    target_label.configure(image=target_image, text="")

    update_status("Swapped source and target")


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET

    if PREVIEW:
        PREVIEW.withdraw()

    target_path = ctk.filedialog.askopenfilename(
        title="Select target image or video",
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )

    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(target_path)
        image = render_image_preview(target_path, (120, 120))
        target_label.configure(image=image, text="")
        update_status(f"Target: {os.path.basename(target_path)}")
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(target_path)
        video_frame = render_video_preview(target_path, (120, 120))
        target_label.configure(image=video_frame, text="")
        update_status(f"Target: {os.path.basename(target_path)}")
    else:
        modules.globals.target_path = None
        target_label.configure(image=None, text="No Target")


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="Save output image",
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title="Save output video",
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None

    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(output_path)
        start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame

    if type(target) is str:
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:
        check_nsfw = predict_frame

    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(to_quit=False)
        update_status("Processing ignored - NSFW content detected")
        return True
    return False


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
        return image
    h, w = image.shape[:2]
    ratio = min(width / w, height / h)
    new_size = (int(w * ratio), int(h * ratio))
    return cv2.resize(image, new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(video_path: str, size: Tuple[int, int], frame_number: int = 0) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    capture.release()

    if has_frame:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    return None


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()
    else:
        update_status("Please select both source and target first")


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    elif is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing preview...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)

        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return

        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )

        image = Image.fromarray(cv2.cvtColor(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (1200, 700), Image.LANCZOS)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Preview ready")
        PREVIEW.deiconify()


def get_available_cameras():
    """Returns lists of camera indices and names."""
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()
            if devices:
                return list(range(len(devices))), devices
        except:
            pass

    # Fallback: probe cameras
    indices = []
    names = []
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            indices.append(i)
            names.append(f"Camera {i}")
            cap.release()

    return (indices, names) if indices else ([], ["No cameras found"])


def webcam_preview(root: ctk.CTk, camera_index: int):
    global POPUP_LIVE

    if camera_index is None:
        update_status("No camera selected")
        return

    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.focus_force()
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        create_webcam_preview(camera_index)
    else:
        modules.globals.source_target_map = []
        create_source_target_popup_for_webcam(root, modules.globals.source_target_map, camera_index)


def create_webcam_preview(camera_index: int):
    """
    Non-blocking parallel GPU pipeline webcam preview.
    Uses after() callbacks instead of blocking while loop for responsive UI.
    """
    global preview_label, PREVIEW

    PREVIEW_WIDTH = 960
    PREVIEW_HEIGHT = 540

    cap = VideoCapturer(camera_index)
    if not cap.start(PREVIEW_WIDTH, PREVIEW_HEIGHT, 60):
        update_status("Failed to start camera")
        return

    # Configure preview window
    PREVIEW.geometry(f"{PREVIEW_WIDTH + 20}x{PREVIEW_HEIGHT + 50}")
    preview_label.configure(width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
    PREVIEW.deiconify()
    PREVIEW.lift()

    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)

    swap_processor = None
    enhance_processor = None
    for fp in frame_processors:
        if fp.NAME == "DLC.FACE-ENHANCER":
            enhance_processor = fp
        else:
            swap_processor = fp

    # State stored in a dict to allow modification in nested functions
    state = {
        'source_image': None,
        'prev_time': time.time(),
        'frame_count': 0,
        'processed_frame_count': 0,
        'processed_fps': 0,
        'last_processed_frame': None,
        'debug_count': 0,
        'swap_times': {'total': 0, 'count': 0},
        'enhance_times': {'total': 0, 'count': 0},
        'running': True,
        'after_id': None,
    }

    # Pipeline queues
    swap_queue = queue.Queue(maxsize=2)
    enhance_queue = queue.Queue(maxsize=2)
    display_queue = queue.Queue(maxsize=2)
    shutdown_event = threading.Event()

    def swap_worker():
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        while not shutdown_event.is_set():
            try:
                frame_data = swap_queue.get(timeout=0.05)
                if frame_data is None:
                    break

                temp_frame, use_map, source_img = frame_data
                t0 = time.perf_counter()

                if swap_processor:
                    if not use_map:
                        temp_frame = swap_processor.process_frame(source_img, temp_frame)
                    else:
                        temp_frame = swap_processor.process_frame_v2(temp_frame)

                state['swap_times']['total'] += time.perf_counter() - t0
                state['swap_times']['count'] += 1

                try:
                    enhance_queue.put_nowait(temp_frame)
                except queue.Full:
                    try:
                        enhance_queue.get_nowait()
                        enhance_queue.put_nowait(temp_frame)
                    except:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SWAP] Error: {e}")

    def enhance_worker():
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            torch.cuda.set_device(1)

        while not shutdown_event.is_set():
            try:
                temp_frame = enhance_queue.get(timeout=0.05)
                if temp_frame is None:
                    break

                t0 = time.perf_counter()

                if enhance_processor and modules.globals.fp_ui.get("face_enhancer", False):
                    if not modules.globals.map_faces:
                        temp_frame = enhance_processor.process_frame(None, temp_frame)
                    else:
                        temp_frame = enhance_processor.process_frame_v2(temp_frame)

                state['enhance_times']['total'] += time.perf_counter() - t0
                state['enhance_times']['count'] += 1

                try:
                    display_queue.put_nowait(temp_frame)
                except queue.Full:
                    try:
                        display_queue.get_nowait()
                        display_queue.put_nowait(temp_frame)
                    except:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ENHANCE] Error: {e}")

    # Start workers
    swap_thread = threading.Thread(target=swap_worker, daemon=True)
    enhance_thread = threading.Thread(target=enhance_worker, daemon=True)
    swap_thread.start()
    enhance_thread.start()

    # Reusable CTkImage to reduce object creation
    current_ctk_image = [None]

    def update_frame():
        """Non-blocking frame update using after()."""
        if not state['running'] or PREVIEW.state() == "withdrawn":
            cleanup()
            return

        ret, frame = cap.read()
        if not ret:
            cleanup()
            return

        temp_frame = frame.copy()
        if modules.globals.live_mirror:
            temp_frame = cv2.flip(temp_frame, 1)

        temp_frame = fit_image_to_size(temp_frame, PREVIEW_WIDTH, PREVIEW_HEIGHT)

        # Load source image once
        if state['source_image'] is None and modules.globals.source_path:
            state['source_image'] = get_one_face(cv2.imread(modules.globals.source_path))

        # Submit frame to processing pipeline
        try:
            swap_queue.put_nowait((temp_frame, modules.globals.map_faces, state['source_image']))
        except queue.Full:
            try:
                swap_queue.get_nowait()
                swap_queue.put_nowait((temp_frame, modules.globals.map_faces, state['source_image']))
            except:
                pass

        # Get processed frame if available
        try:
            new_frame = display_queue.get_nowait()
            state['last_processed_frame'] = new_frame
            state['processed_frame_count'] += 1
        except queue.Empty:
            pass

        # Display frame
        display_frame = state['last_processed_frame'] if state['last_processed_frame'] is not None else temp_frame

        # FPS calculation
        current_time = time.time()
        state['frame_count'] += 1
        if current_time - state['prev_time'] >= 0.5:
            state['processed_fps'] = state['processed_frame_count'] / (current_time - state['prev_time'])
            state['frame_count'] = 0
            state['processed_frame_count'] = 0
            state['prev_time'] = current_time

        if modules.globals.show_fps:
            cv2.putText(display_frame, f"FPS: {state['processed_fps']:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Efficient display update
        rgb_frame = display_frame[:, :, ::-1].copy()  # Ensure contiguous array
        image = Image.fromarray(rgb_frame)
        current_ctk_image[0] = ctk.CTkImage(light_image=image, size=(display_frame.shape[1], display_frame.shape[0]))
        preview_label.configure(image=current_ctk_image[0])

        # Debug output (less frequent)
        state['debug_count'] += 1
        if state['debug_count'] >= 60:  # Every ~2 seconds at 30fps
            avg_swap = (state['swap_times']['total'] / max(state['swap_times']['count'], 1)) * 1000
            avg_enhance = (state['enhance_times']['total'] / max(state['enhance_times']['count'], 1)) * 1000
            print(f"[TIMING] Swap: {avg_swap:.1f}ms, Enhance: {avg_enhance:.1f}ms, FPS: {state['processed_fps']:.1f}")
            state['debug_count'] = 0
            state['swap_times'] = {'total': 0, 'count': 0}
            state['enhance_times'] = {'total': 0, 'count': 0}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Schedule next frame (non-blocking) - ~33ms = 30fps display rate
        state['after_id'] = PREVIEW.after(16, update_frame)

    def cleanup():
        """Clean up resources."""
        state['running'] = False
        if state['after_id']:
            try:
                PREVIEW.after_cancel(state['after_id'])
            except:
                pass

        shutdown_event.set()
        swap_queue.put(None)
        enhance_queue.put(None)
        swap_thread.join(timeout=1.0)
        enhance_thread.join(timeout=1.0)
        cap.release()
        PREVIEW.withdraw()

    def on_preview_close():
        """Handle preview window close."""
        state['running'] = False

    # Bind close event
    PREVIEW.protocol("WM_DELETE_WINDOW", on_preview_close)

    # Start the non-blocking update loop
    update_frame()


def create_source_target_popup_for_webcam(root: ctk.CTk, map: list, camera_index: int) -> None:
    global POPUP_LIVE, popup_status_label_live

    # Create popup without grab_set since file dialogs need to work
    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title("Source ↔ Target Mapper (Live)")
    POPUP_LIVE.geometry("800x650")
    POPUP_LIVE.transient(root)
    POPUP_LIVE.focus_force()
    POPUP_LIVE.lift()

    # Center on parent
    POPUP_LIVE.update_idletasks()
    x = root.winfo_x() + (root.winfo_width() - 800) // 2
    y = root.winfo_y() + (root.winfo_height() - 650) // 2
    POPUP_LIVE.geometry(f"+{x}+{y}")

    def on_submit():
        if has_valid_map():
            simplify_maps()
            popup_status_label_live.configure(text="Mappings saved!", text_color="#28a745")
            POPUP_LIVE.after(500, lambda: create_webcam_preview(camera_index))
        else:
            popup_status_label_live.configure(text="At least 1 mapping required!", text_color="#dc3545")

    def on_add():
        add_blank_map()
        refresh_data(map)
        update_pop_live_status("Added new mapping row")

    def on_clear():
        clear_source_target_images(map)
        refresh_data(map)
        update_pop_live_status("All mappings cleared")

    # Status label at top
    popup_status_label_live = ctk.CTkLabel(POPUP_LIVE, text="Select source & target images, then click Start Live")
    popup_status_label_live.grid(row=0, column=0, pady=10, sticky="ew")

    # Add initial blank mapping so user sees the UI
    if len(map) == 0:
        add_blank_map()

    # Initial refresh to show mapping rows
    refresh_data(map)

    # Footer buttons
    button_frame = ctk.CTkFrame(POPUP_LIVE, fg_color="transparent")
    button_frame.grid(row=2, column=0, pady=10, sticky="ew")

    ctk.CTkButton(button_frame, text="Add Mapping", width=120, command=on_add).pack(side="left", padx=20)
    ctk.CTkButton(button_frame, text="Clear All", width=120, command=on_clear).pack(side="left", padx=20)
    ctk.CTkButton(button_frame, text="Start Live", width=120, fg_color="#28a745", command=on_submit).pack(side="left", padx=20)

    # Configure grid weights
    POPUP_LIVE.grid_columnconfigure(0, weight=1)


def clear_source_target_images(map: list):
    global source_label_dict_live, target_label_dict_live

    for item in map:
        if "source" in item:
            del item["source"]
        if "target" in item:
            del item["target"]

    for key in list(source_label_dict_live.keys()):
        source_label_dict_live[key].destroy()
        del source_label_dict_live[key]

    for key in list(target_label_dict_live.keys()):
        target_label_dict_live[key].destroy()
        del target_label_dict_live[key]


def refresh_data(map: list):
    """Refresh the mapping UI - creates a new scrollable frame each time (original approach)."""
    global POPUP_LIVE, source_label_dict_live, target_label_dict_live

    if not POPUP_LIVE or not POPUP_LIVE.winfo_exists():
        return

    # Create a new scrollable frame (this replaces any existing one)
    scrollable_frame = ctk.CTkScrollableFrame(POPUP_LIVE, width=760, height=480)
    scrollable_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

    def on_source_click(map, id):
        update_webcam_source(scrollable_frame, map, id)

    def on_target_click(map, id):
        update_webcam_target(scrollable_frame, map, id)

    # Create a row for each mapping
    for item in map:
        id = item["id"]

        # Source button (column 0)
        ctk.CTkButton(
            scrollable_frame,
            text="Select Source",
            command=lambda id=id: on_source_click(map, id),
            width=140,
            height=32,
        ).grid(row=id, column=0, padx=10, pady=10)

        # Source image preview (column 1)
        if "source" in item:
            src_img = Image.fromarray(cv2.cvtColor(item["source"]["cv2"], cv2.COLOR_BGR2RGB))
            src_img = src_img.resize((75, 75), Image.LANCZOS)
            src_ctk = ctk.CTkImage(src_img, size=(75, 75))
            src_label = ctk.CTkLabel(scrollable_frame, image=src_ctk, text="")
            src_label.grid(row=id, column=1, padx=10, pady=10)
            source_label_dict_live[id] = src_label

        # Arrow (column 2)
        ctk.CTkLabel(
            scrollable_frame,
            text="→",
            font=("", 20),
            width=40,
        ).grid(row=id, column=2, padx=5, pady=10)

        # Target button (column 3)
        ctk.CTkButton(
            scrollable_frame,
            text="Select Target",
            command=lambda id=id: on_target_click(map, id),
            width=140,
            height=32,
        ).grid(row=id, column=3, padx=10, pady=10)

        # Target image preview (column 4)
        if "target" in item:
            tgt_img = Image.fromarray(cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
            tgt_img = tgt_img.resize((75, 75), Image.LANCZOS)
            tgt_ctk = ctk.CTkImage(tgt_img, size=(75, 75))
            tgt_label = ctk.CTkLabel(scrollable_frame, image=tgt_ctk, text="")
            tgt_label.grid(row=id, column=4, padx=10, pady=10)
            target_label_dict_live[id] = tgt_label


def update_webcam_source(scrollable, map, item_id):
    global source_label_dict_live, RECENT_DIRECTORY_SOURCE, POPUP_LIVE

    # Find item by ID
    item = next((x for x in map if x["id"] == item_id), None)
    if item is None:
        return map

    # Remove old source if exists
    if "source" in item:
        item.pop("source")
        if item_id in source_label_dict_live:
            source_label_dict_live[item_id].destroy()
            del source_label_dict_live[item_id]

    source_path = ctk.filedialog.askopenfilename(
        title="Select source image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if not source_path:
        return map

    RECENT_DIRECTORY_SOURCE = os.path.dirname(source_path)
    cv2_img = cv2.imread(source_path)
    face = get_one_face(cv2_img)

    if face:
        x_min, y_min, x_max, y_max = face["bbox"]
        item["source"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }

        # Create and display the source image preview
        src_img = Image.fromarray(cv2.cvtColor(item["source"]["cv2"], cv2.COLOR_BGR2RGB))
        src_img = src_img.resize((75, 75), Image.LANCZOS)
        src_ctk = ctk.CTkImage(src_img, size=(75, 75))

        src_label = ctk.CTkLabel(scrollable, image=src_ctk, text="")
        src_label.grid(row=item_id, column=1, padx=10, pady=10)
        source_label_dict_live[item_id] = src_label

        update_pop_live_status("Source face added!")
    else:
        update_pop_live_status("No face detected in source image!")

    return map


def update_webcam_target(scrollable, map, item_id):
    global target_label_dict_live, RECENT_DIRECTORY_SOURCE, POPUP_LIVE

    # Find item by ID
    item = next((x for x in map if x["id"] == item_id), None)
    if item is None:
        return map

    # Remove old target if exists
    if "target" in item:
        item.pop("target")
        if item_id in target_label_dict_live:
            target_label_dict_live[item_id].destroy()
            del target_label_dict_live[item_id]

    target_path = ctk.filedialog.askopenfilename(
        title="Select target image",
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if not target_path:
        return map

    RECENT_DIRECTORY_SOURCE = os.path.dirname(target_path)
    cv2_img = cv2.imread(target_path)
    face = get_one_face(cv2_img)

    if face:
        x_min, y_min, x_max, y_max = face["bbox"]
        item["target"] = {
            "cv2": cv2_img[int(y_min):int(y_max), int(x_min):int(x_max)],
            "face": face,
        }

        # Create and display the target image preview
        tgt_img = Image.fromarray(cv2.cvtColor(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        tgt_img = tgt_img.resize((75, 75), Image.LANCZOS)
        tgt_ctk = ctk.CTkImage(tgt_img, size=(75, 75))

        tgt_label = ctk.CTkLabel(scrollable, image=tgt_ctk, text="")
        tgt_label.grid(row=item_id, column=4, padx=10, pady=10)
        target_label_dict_live[item_id] = tgt_label

        update_pop_live_status("Target face added!")
    else:
        update_pop_live_status("No face detected in target image!")

    return map
