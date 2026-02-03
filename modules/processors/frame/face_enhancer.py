# --- START OF FILE face_enhancer.py ---

from typing import Any, List
import cv2
import threading
import gfpgan
import onnxruntime
import os
import platform
import torch
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

FACE_ENHANCER = None  # GFPGAN PyTorch
GPEN_ENHANCER = None  # GPEN ONNX session
GFPGAN_ONNX_ENHANCER = None  # GFPGAN ONNX session (same as FaceFusion)
CODEFORMER_ENHANCER = None  # CodeFormer ONNX session
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

# Enhancement settings - can be modified via globals
DEFAULT_ENHANCE_SCALE = 0.5

# Available enhancer types:
# "gfpgan_onnx" - GFPGAN via ONNX (same as FaceFusion uses) ~80-120ms
# "codeformer" - CodeFormer ONNX - high quality ~80-120ms
# "gfpgan" - GFPGAN PyTorch - high quality but slower ~200-300ms
# "gpen" - GPEN ONNX - mixed results
# "fast" - CV operations ~5-10ms
# "none" - no enhancement

# Model URLs
GPEN_MODEL_URL = "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/facerestore_models/GPEN-BFR-512.onnx"
GPEN_MODEL_NAME = "GPEN-BFR-512.onnx"

GFPGAN_ONNX_URL = "https://huggingface.co/facefusion/models-3.0.0/resolve/main/gfpgan_1.4.onnx"
GFPGAN_ONNX_NAME = "gfpgan_1.4.onnx"

CODEFORMER_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/codeformer.onnx"
CODEFORMER_NAME = "codeformer.onnx"

# FFHQ 512x512 template for face alignment (scaled from ArcFace 112x112 template)
# This is the standard template used by FaceFusion and other face enhancement tools
FFHQ_512_TEMPLATE = np.array([
    [192.98138, 239.94708],  # left eye
    [318.90277, 240.19366],  # right eye
    [256.63416, 314.01935],  # nose
    [201.26117, 371.41043],  # left mouth corner
    [313.08905, 371.15118]   # right mouth corner
], dtype=np.float32)

# ArcFace 112x112 template (for reference)
ARCFACE_112_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


def warp_face_by_landmarks(frame: Frame, landmarks_5: np.ndarray, template: np.ndarray, size: tuple) -> tuple:
    """
    Warp face to standard template using affine transformation.

    Args:
        frame: Input frame
        landmarks_5: 5-point face landmarks (2 eyes, nose, 2 mouth corners)
        template: Target template landmarks
        size: Output size (width, height)

    Returns:
        (warped_face, affine_matrix)
    """
    # Estimate affine transformation matrix
    affine_matrix = cv2.estimateAffinePartial2D(landmarks_5, template, method=cv2.LMEDS)[0]

    # Warp the face
    warped = cv2.warpAffine(frame, affine_matrix, size, borderMode=cv2.BORDER_REPLICATE)

    return warped, affine_matrix


def paste_back(frame: Frame, enhanced_face: Frame, affine_matrix: np.ndarray, blend: float = 0.8) -> Frame:
    """
    Paste enhanced face back onto original frame with blending.

    Args:
        frame: Original frame
        enhanced_face: Enhanced face (512x512)
        affine_matrix: Affine matrix used for warping
        blend: Blend factor (0.0-1.0, default 0.8 like FaceFusion)

    Returns:
        Frame with enhanced face blended in
    """
    # Invert the affine matrix
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)

    # Get frame dimensions
    frame_h, frame_w = frame.shape[:2]
    face_h, face_w = enhanced_face.shape[:2]

    # Warp enhanced face back to original frame coordinates
    warped_face = cv2.warpAffine(
        enhanced_face, inverse_matrix, (frame_w, frame_h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    # Create mask: detect where warped_face has content (non-black)
    # Convert to grayscale and threshold
    warped_gray = cv2.cvtColor(warped_face, cv2.COLOR_BGR2GRAY)
    _, content_mask = cv2.threshold(warped_gray, 1, 255, cv2.THRESH_BINARY)

    # Erode mask slightly to remove edge artifacts, then blur for smooth blending
    kernel = np.ones((5, 5), np.uint8)
    content_mask = cv2.erode(content_mask, kernel, iterations=2)
    content_mask = cv2.GaussianBlur(content_mask, (21, 21), 0)

    # Normalize to 0-1 and apply blend factor
    mask_float = (content_mask / 255.0) * blend

    # Expand to 3 channels
    mask_3ch = mask_float[:, :, np.newaxis]

    # Blend
    result = frame.astype(np.float32) * (1 - mask_3ch) + warped_face.astype(np.float32) * mask_3ch

    return result.astype(np.uint8)


def enhance_fast(frame: Frame) -> Frame:
    """
    Fast CV-based face enhancement using sharpening and bilateral filter.
    Much faster than GFPGAN (~5-10ms vs ~200-300ms) but lower quality.
    """
    # Apply bilateral filter for smoothing while preserving edges
    smooth = cv2.bilateralFilter(frame, 5, 50, 50)

    # Unsharp masking for sharpening
    gaussian = cv2.GaussianBlur(smooth, (0, 0), 2.0)
    sharpened = cv2.addWeighted(smooth, 1.5, gaussian, -0.5, 0)

    # Slight contrast enhancement
    lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    result = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return result


def get_gpen_enhancer() -> Any:
    """Initialize and return GPEN ONNX session with TensorRT/CUDA optimization."""
    global GPEN_ENHANCER

    with THREAD_LOCK:
        if GPEN_ENHANCER is None:
            model_path = os.path.join(models_dir, GPEN_MODEL_NAME)

            # Download if not exists
            if not os.path.exists(model_path):
                print(f"{NAME}: Downloading GPEN model...")
                conditional_download(models_dir, [GPEN_MODEL_URL])

            if not os.path.exists(model_path):
                raise RuntimeError(f"{NAME}: Failed to download GPEN model")

            # Get optimized providers (TensorRT > CUDA > CPU)
            providers = get_execution_providers(device_id=1)
            session_options = get_onnx_session_options()

            print(f"{NAME}: Loading GPEN model from {model_path}")
            GPEN_ENHANCER = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
            print(f"{NAME}: GPEN loaded with providers: {GPEN_ENHANCER.get_providers()}")

    return GPEN_ENHANCER


def enhance_gpen(frame: Frame) -> Frame:
    """
    Enhance face using GPEN ONNX model.
    Faster than GFPGAN (~50-100ms) with good quality.
    """
    session = get_gpen_enhancer()

    try:
        # GPEN expects 512x512 input
        original_size = (frame.shape[1], frame.shape[0])
        input_size = 512

        # Resize to 512x512
        resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        # Preprocess: BGR to RGB, normalize to [-1, 1], NCHW format
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5  # Normalize to [-1, 1]
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img})[0]

        # Postprocess: denormalize, CHW to HWC, RGB to BGR
        result = np.squeeze(result, axis=0)  # Remove batch
        result = np.transpose(result, (1, 2, 0))  # CHW to HWC
        result = (result * 0.5 + 0.5) * 255.0  # Denormalize to [0, 255]
        result = np.clip(result, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        # Resize back to original size
        result = cv2.resize(result, original_size, interpolation=cv2.INTER_LINEAR)

        return result

    except Exception as e:
        print(f"{NAME}: GPEN error: {e}")
        return frame


abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)

# TensorRT cache directory for compiled engines
trt_cache_dir = os.path.join(models_dir, "trt_cache")


def get_onnx_session_options() -> onnxruntime.SessionOptions:
    """Get optimized ONNX session options."""
    options = onnxruntime.SessionOptions()
    options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Enable memory optimization
    options.enable_cpu_mem_arena = True
    options.enable_mem_pattern = True
    options.enable_mem_reuse = True

    return options


def get_execution_providers(device_id: int = 1) -> list:
    """
    Get optimized execution providers with TensorRT and FP16 support.

    Priority: TensorRT > CUDA > CPU
    TensorRT provides 30-50% speedup over CUDA for supported models.
    """
    providers = []
    available = onnxruntime.get_available_providers()
    use_fp16 = getattr(modules.globals, 'use_fp16', True)

    # Ensure TRT cache directory exists
    if not os.path.exists(trt_cache_dir):
        os.makedirs(trt_cache_dir, exist_ok=True)

    # Priority 1: TensorRT (fastest)
    if 'TensorrtExecutionProvider' in available:
        trt_options = {
            'device_id': device_id,
            'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,  # 2GB workspace
            'trt_fp16_enable': use_fp16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': trt_cache_dir,
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': trt_cache_dir,
        }
        providers.append(('TensorrtExecutionProvider', trt_options))
        print(f"{NAME}: TensorRT enabled (FP16={use_fp16}, cache={trt_cache_dir})")

    # Priority 2: CUDA with FP16
    if 'CUDAExecutionProvider' in available:
        cuda_options = {
            'device_id': device_id,
            'arena_extend_strategy': 'kSameAsRequested',
            'cudnn_conv_algo_search': 'HEURISTIC',
        }
        # Note: CUDA provider FP16 is controlled by model, not provider options
        providers.append(('CUDAExecutionProvider', cuda_options))

    # Fallback: CPU
    providers.append('CPUExecutionProvider')

    return providers


def get_gfpgan_onnx_enhancer() -> Any:
    """Initialize and return GFPGAN ONNX session with TensorRT/CUDA optimization."""
    global GFPGAN_ONNX_ENHANCER

    with THREAD_LOCK:
        if GFPGAN_ONNX_ENHANCER is None:
            model_path = os.path.join(models_dir, GFPGAN_ONNX_NAME)

            # Download if not exists
            if not os.path.exists(model_path):
                print(f"{NAME}: Downloading GFPGAN ONNX model (340MB)...")
                conditional_download(models_dir, [GFPGAN_ONNX_URL])

            if not os.path.exists(model_path):
                raise RuntimeError(f"{NAME}: Failed to download GFPGAN ONNX model")

            # Get optimized providers (TensorRT > CUDA > CPU)
            providers = get_execution_providers(device_id=1)
            session_options = get_onnx_session_options()

            print(f"{NAME}: Loading GFPGAN ONNX model from {model_path}")
            print(f"{NAME}: First run with TensorRT may take 1-2 min to compile engine...")
            GFPGAN_ONNX_ENHANCER = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
            print(f"{NAME}: GFPGAN ONNX loaded with providers: {GFPGAN_ONNX_ENHANCER.get_providers()}")

    return GFPGAN_ONNX_ENHANCER


def enhance_gfpgan_onnx(frame: Frame, face: Face = None) -> Frame:
    """
    Enhance face using GFPGAN ONNX model (same as FaceFusion).
    Uses proper FFHQ face alignment when face landmarks are provided.

    Args:
        frame: Input frame
        face: Face object with landmarks (optional, enables proper alignment)

    Returns:
        Enhanced frame
    """
    session = get_gfpgan_onnx_enhancer()
    blend = getattr(modules.globals, 'enhancer_blend', 0.8)

    try:
        input_size = 512
        use_alignment = face is not None and hasattr(face, 'landmark_2d_106')

        if use_alignment:
            # Get 5-point landmarks from 106-point landmarks
            # Indices: left eye, right eye, nose, left mouth, right mouth
            lm106 = face.landmark_2d_106
            landmarks_5 = np.array([
                np.mean(lm106[33:42], axis=0),   # left eye center
                np.mean(lm106[87:96], axis=0),   # right eye center
                lm106[72],                        # nose tip
                lm106[54],                        # left mouth corner
                lm106[76],                        # right mouth corner
            ], dtype=np.float32)

            # Warp face to FFHQ 512 template
            warped_face, affine_matrix = warp_face_by_landmarks(
                frame, landmarks_5, FFHQ_512_TEMPLATE, (input_size, input_size)
            )
            input_face = warped_face
        else:
            # Fallback: simple resize (less accurate)
            input_face = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        # Preprocess: BGR to RGB, normalize to [0, 1], NCHW format
        img = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: img})[0]

        # Postprocess: CHW to HWC, clip and convert
        result = np.squeeze(result, axis=0)  # Remove batch
        result = np.transpose(result, (1, 2, 0))  # CHW to HWC
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        if use_alignment:
            # Paste back with blending
            return paste_back(frame, result, affine_matrix, blend)
        else:
            # Simple resize back
            original_size = (frame.shape[1], frame.shape[0])
            return cv2.resize(result, original_size, interpolation=cv2.INTER_LINEAR)

    except Exception as e:
        print(f"{NAME}: GFPGAN ONNX error: {e}")
        import traceback
        traceback.print_exc()
        return frame


def get_codeformer_enhancer() -> Any:
    """Initialize and return CodeFormer ONNX session with TensorRT/CUDA optimization."""
    global CODEFORMER_ENHANCER

    with THREAD_LOCK:
        if CODEFORMER_ENHANCER is None:
            model_path = os.path.join(models_dir, CODEFORMER_NAME)

            # Download if not exists
            if not os.path.exists(model_path):
                print(f"{NAME}: Downloading CodeFormer ONNX model...")
                conditional_download(models_dir, [CODEFORMER_URL])

            if not os.path.exists(model_path):
                raise RuntimeError(f"{NAME}: Failed to download CodeFormer model")

            # Get optimized providers (TensorRT > CUDA > CPU)
            providers = get_execution_providers(device_id=1)
            session_options = get_onnx_session_options()

            print(f"{NAME}: Loading CodeFormer model from {model_path}")
            print(f"{NAME}: First run with TensorRT may take 1-2 min to compile engine...")
            CODEFORMER_ENHANCER = onnxruntime.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers
            )
            print(f"{NAME}: CodeFormer loaded with providers: {CODEFORMER_ENHANCER.get_providers()}")

    return CODEFORMER_ENHANCER


def enhance_codeformer(frame: Frame, face: Face = None) -> Frame:
    """
    Enhance face using CodeFormer ONNX model.
    Uses proper FFHQ face alignment when face landmarks are provided.

    Args:
        frame: Input frame
        face: Face object with landmarks (optional, enables proper alignment)

    Returns:
        Enhanced frame
    """
    session = get_codeformer_enhancer()
    blend = getattr(modules.globals, 'enhancer_blend', 0.8)

    try:
        input_size = 512
        use_alignment = face is not None and hasattr(face, 'landmark_2d_106')

        if use_alignment:
            # Get 5-point landmarks from 106-point landmarks
            lm106 = face.landmark_2d_106
            landmarks_5 = np.array([
                np.mean(lm106[33:42], axis=0),   # left eye center
                np.mean(lm106[87:96], axis=0),   # right eye center
                lm106[72],                        # nose tip
                lm106[54],                        # left mouth corner
                lm106[76],                        # right mouth corner
            ], dtype=np.float32)

            # Warp face to FFHQ 512 template
            warped_face, affine_matrix = warp_face_by_landmarks(
                frame, landmarks_5, FFHQ_512_TEMPLATE, (input_size, input_size)
            )
            input_face = warped_face
        else:
            # Fallback: simple resize (less accurate)
            input_face = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

        # Preprocess: BGR to RGB, normalize to [0, 1], NCHW format
        img = cv2.cvtColor(input_face, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC to CHW
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # CodeFormer requires a 'weight' input for fidelity control
        # Higher weight = more faithful to original, lower = more restoration
        # Range: 0.0 (full restoration) to 1.0 (full fidelity), default 0.7
        weight = np.array([0.7], dtype=np.float64)

        # Run inference with both inputs
        result = session.run(
            None,
            {
                'input': img,
                'weight': weight
            }
        )[0]

        # Postprocess: CHW to HWC, clip and convert
        result = np.squeeze(result, axis=0)  # Remove batch
        result = np.transpose(result, (1, 2, 0))  # CHW to HWC
        result = np.clip(result * 255.0, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

        if use_alignment:
            # Paste back with blending
            return paste_back(frame, result, affine_matrix, blend)
        else:
            # Simple resize back
            original_size = (frame.shape[1], frame.shape[0])
            return cv2.resize(result, original_size, interpolation=cv2.INTER_LINEAR)

    except Exception as e:
        print(f"{NAME}: CodeFormer error: {e}")
        import traceback
        traceback.print_exc()
        return frame


def pre_check() -> bool:
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    """
    Initializes and returns the GFPGAN face enhancer instance,
    prioritizing CUDA, then MPS (Mac), then CPU.
    """
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            device = None
            try:
                # Priority 1: CUDA - use second GPU if available for parallel processing
                if torch.cuda.is_available():
                    gpu_count = torch.cuda.device_count()
                    if gpu_count > 1:
                        device = torch.device("cuda:1")  # Use second GPU for face enhancer
                        print(f"{NAME}: Using CUDA device 1 (of {gpu_count} GPUs) for parallel processing.")
                    else:
                        device = torch.device("cuda:0")
                        print(f"{NAME}: Using CUDA device 0.")
                # Priority 2: MPS (Mac Silicon)
                elif platform.system() == "Darwin" and torch.backends.mps.is_available():
                    device = torch.device("mps")
                    print(f"{NAME}: Using MPS device.")
                # Priority 3: CPU
                else:
                    device = torch.device("cpu")
                    print(f"{NAME}: Using CPU device.")

                FACE_ENHANCER = gfpgan.GFPGANer(
                    model_path=model_path,
                    upscale=1,  # upscale=1 means enhancement only, no resizing
                    arch='clean',
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=device
                )
                print(f"{NAME}: GFPGANer initialized successfully on {device}.")

            except Exception as e:
                print(f"{NAME}: Error initializing GFPGANer: {e}")
                # Fallback to CPU if initialization with GPU fails for some reason
                if device is not None and device.type != 'cpu':
                    print(f"{NAME}: Falling back to CPU due to error.")
                    try:
                        device = torch.device("cpu")
                        FACE_ENHANCER = gfpgan.GFPGANer(
                            model_path=model_path,
                            upscale=1,
                            arch='clean',
                            channel_multiplier=2,
                            bg_upsampler=None,
                            device=device
                        )
                        print(f"{NAME}: GFPGANer initialized successfully on CPU after fallback.")
                    except Exception as fallback_e:
                         print(f"{NAME}: FATAL: Could not initialize GFPGANer even on CPU: {fallback_e}")
                         FACE_ENHANCER = None # Ensure it's None if totally failed
                else:
                    # If it failed even on the first CPU attempt or device was already CPU
                     print(f"{NAME}: FATAL: Could not initialize GFPGANer on CPU: {e}")
                     FACE_ENHANCER = None # Ensure it's None if totally failed


    # Check if enhancer is still None after attempting initialization
    if FACE_ENHANCER is None:
        raise RuntimeError(f"{NAME}: Failed to initialize GFPGANer. Check logs for errors.")

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame, scale: float = None, face: Face = None) -> Frame:
    """
    Enhances faces in a single frame.

    Uses enhancer_type from globals to determine method:
    - "gfpgan_onnx": GFPGAN ONNX (same as FaceFusion, ~80-120ms) - RECOMMENDED
    - "codeformer": CodeFormer ONNX (high quality, ~80-120ms)
    - "gfpgan": Full GFPGAN PyTorch model (high quality, ~200-300ms)
    - "gpen": GPEN ONNX model (mixed results, ~50-100ms)
    - "fast": CV-based enhancement (lower quality, ~5-10ms)
    - "none": No enhancement

    Args:
        temp_frame: Input frame
        scale: Processing scale (0.5 = half res for speed, 1.0 = full res for quality)
               If None, uses modules.globals.enhance_scale or DEFAULT_ENHANCE_SCALE
        face: Face object with landmarks (enables proper FFHQ alignment for ONNX enhancers)
    """
    enhancer_type = getattr(modules.globals, 'enhancer_type', 'gfpgan_onnx')

    # No enhancement
    if enhancer_type == "none":
        return temp_frame

    # Fast CV-based enhancement
    if enhancer_type == "fast":
        return enhance_fast(temp_frame)

    # GPEN ONNX enhancement
    if enhancer_type == "gpen":
        return enhance_gpen(temp_frame)

    # For ONNX enhancers, detect face if not provided (for proper alignment)
    if face is None and enhancer_type in ("gfpgan_onnx", "codeformer"):
        face = get_one_face(temp_frame)

    # GFPGAN ONNX enhancement (FaceFusion's default) - RECOMMENDED
    if enhancer_type == "gfpgan_onnx":
        return enhance_gfpgan_onnx(temp_frame, face)

    # CodeFormer ONNX enhancement
    if enhancer_type == "codeformer":
        return enhance_codeformer(temp_frame, face)

    # GFPGAN enhancement (default)
    if scale is None:
        scale = getattr(modules.globals, 'enhance_scale', DEFAULT_ENHANCE_SCALE)

    enhancer = get_face_enhancer()
    original_size = (temp_frame.shape[1], temp_frame.shape[0])  # (width, height)

    try:
        # Downscale for faster processing if scale < 1.0
        if scale < 1.0:
            new_width = int(original_size[0] * scale)
            new_height = int(original_size[1] * scale)
            processing_frame = cv2.resize(temp_frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            processing_frame = temp_frame

        # The enhance method returns: _, restored_faces, restored_img
        _, _, restored_img = enhancer.enhance(
            processing_frame,
            has_aligned=False,
            only_center_face=True,  # Only enhance center face for speed
            paste_back=True
        )

        # GFPGAN might return None if no face is detected
        if restored_img is None:
            return temp_frame

        # Upscale back to original size if we downscaled
        if scale < 1.0:
            restored_img = cv2.resize(restored_img, original_size, interpolation=cv2.INTER_LANCZOS4)

        return restored_img
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def process_frame(source_face: Face | None, temp_frame: Frame) -> Frame:
    """Processes a frame: enhances face if detected."""
    # We don't strictly need source_face for enhancement only
    # Check if any face exists to potentially save processing time, though GFPGAN also does detection.
    # For simplicity and ensuring enhancement is attempted if possible, we can rely on enhance_face.
    # target_face = get_one_face(temp_frame) # This gets only ONE face
    # If you want to enhance ONLY if a face is detected by your *own* analyser first:
    # has_face = get_one_face(temp_frame) is not None # Or use get_many_faces
    # if has_face:
    #     temp_frame = enhance_face(temp_frame)
    # else: # Enhance regardless, let GFPGAN handle detection
    temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """Processes multiple frames from file paths."""
    for temp_frame_path in temp_frame_paths:
        if not os.path.exists(temp_frame_path):
            print(f"{NAME}: Warning: Frame path not found {temp_frame_path}, skipping.")
            if progress:
                progress.update(1)
            continue

        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            print(f"{NAME}: Warning: Failed to read frame {temp_frame_path}, skipping.")
            if progress:
                progress.update(1)
            continue

        result_frame = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result_frame)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    """Processes a single image file."""
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    """Processes video frames using the frame processor core."""
    # source_path might be optional depending on how process_video is called
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)

def process_frame_v2(temp_frame: Frame) -> Frame:
    """Process frame for live webcam mode."""
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, face=target_face)
    return temp_frame


def process_frame_with_target(temp_frame: Frame, target_face: Face) -> Frame:
    """
    OPTIMIZED VERSION - Process frame with pre-detected target face.
    Skips face detection entirely when target_face is provided.
    """
    if target_face:
        temp_frame = enhance_face(temp_frame, face=target_face)
    return temp_frame


def process_frame_v2_with_targets(temp_frame: Frame, detected_faces: list) -> Frame:
    """
    OPTIMIZED VERSION for map_faces mode - enhance multiple pre-detected faces.
    Skips face detection entirely when faces are provided.
    """
    if detected_faces:
        for face in detected_faces:
            if face is not None:
                temp_frame = enhance_face(temp_frame, face=face)
    return temp_frame

# --- END OF FILE face_enhancer.py ---