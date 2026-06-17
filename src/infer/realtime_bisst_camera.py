# src/infer/realtime_bisst_camera.py

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.models.bisst import BiSSTModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime camera demo for BiSST style transfer."
    )

    # Checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained BiSST checkpoint.",
    )

    # Camera
    parser.add_argument(
        "--camera_id",
        type=int,
        default=0,
        help="Camera id. Usually 0 for default camera, 1 for external camera.",
    )
    parser.add_argument(
        "--camera_width",
        type=int,
        default=640,
        help="Camera capture width.",
    )
    parser.add_argument(
        "--camera_height",
        type=int,
        default=480,
        help="Camera capture height.",
    )

    # Inference image
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Input image size for generator.",
    )

    # Display
    parser.add_argument(
        "--display_width",
        type=int,
        default=512,
        help="Display width for each panel.",
    )
    parser.add_argument(
        "--display_height",
        type=int,
        default=512,
        help="Display height for each panel.",
    )
    parser.add_argument(
        "--show_fps",
        action="store_true",
        help="Show FPS on the output window.",
    )
    parser.add_argument(
        "--window_name",
        type=str,
        default="Realtime BiSST Camera Demo",
        help="OpenCV display window name.",
    )

    # Model architecture, must match training
    parser.add_argument(
        "--ngf",
        type=int,
        default=64,
        help="Number of generator filters. Must match training.",
    )
    parser.add_argument(
        "--ndf",
        type=int,
        default=64,
        help="Number of discriminator filters. Must match training.",
    )
    parser.add_argument(
        "--n_blocks",
        type=int,
        default=9,
        help="Number of ResNet blocks in generator. Must match training.",
    )

    # Dummy args needed by BiSSTModel constructor
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_gan", type=float, default=1.0)
    parser.add_argument("--lambda_nce", type=float, default=1.0)
    parser.add_argument("--lambda_idt", type=float, default=0.5)
    parser.add_argument("--nce_temperature", type=float, default=0.07)
    parser.add_argument("--num_patches", type=int, default=256)

    # Mask predictor Fmask
    mask_predictor_group = parser.add_mutually_exclusive_group()
    mask_predictor_group.add_argument(
        "--use_mask_predictor",
        dest="use_mask_predictor",
        action="store_true",
        help="Construct BiSST with mask predictor Fmask.",
    )
    mask_predictor_group.add_argument(
        "--no_mask_predictor",
        dest="use_mask_predictor",
        action="store_false",
        help="Construct BiSST without mask predictor Fmask.",
    )
    parser.set_defaults(use_mask_predictor=False)

    parser.add_argument("--mask_predictor_type", type=str, default="light_unet")
    parser.add_argument("--mask_base_channels", type=int, default=32)
    parser.add_argument("--mask_checkpoint", type=str, default=None)

    freeze_mask_group = parser.add_mutually_exclusive_group()
    freeze_mask_group.add_argument(
        "--freeze_mask_predictor",
        dest="freeze_mask_predictor",
        action="store_true",
    )
    freeze_mask_group.add_argument(
        "--train_mask_predictor",
        dest="freeze_mask_predictor",
        action="store_false",
    )
    parser.set_defaults(freeze_mask_predictor=True)

    parser.add_argument("--lambda_mask", type=float, default=0.0)
    parser.add_argument("--mask_loss_type", type=str, default="dice_consistency")

    # Embedding extractor Eh
    embedding_group = parser.add_mutually_exclusive_group()
    embedding_group.add_argument(
        "--use_embedding_extractor",
        dest="use_embedding_extractor",
        action="store_true",
        help="Construct BiSST with embedding extractor Eh.",
    )
    embedding_group.add_argument(
        "--no_embedding_extractor",
        dest="use_embedding_extractor",
        action="store_false",
        help="Construct BiSST without embedding extractor Eh.",
    )
    parser.set_defaults(use_embedding_extractor=True)

    parser.add_argument(
        "--embedding_extractor_type",
        type=str,
        default="conv",
        choices=["conv", "pool"],
    )
    parser.add_argument("--embedding_input_channels", type=int, default=None)
    parser.add_argument("--embedding_hidden_channels", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)

    parser.add_argument("--lambda_semantic", type=float, default=0.0)
    parser.add_argument("--lambda_semantic_neg", type=float, default=0.1)
    parser.add_argument("--semantic_loss_type", type=str, default="structural")

    detach_semantic_weight_group = parser.add_mutually_exclusive_group()
    detach_semantic_weight_group.add_argument(
        "--detach_semantic_negative_weight",
        dest="detach_semantic_negative_weight",
        action="store_true",
    )
    detach_semantic_weight_group.add_argument(
        "--no_detach_semantic_negative_weight",
        dest="detach_semantic_negative_weight",
        action="store_false",
    )
    parser.set_defaults(detach_semantic_negative_weight=True)

    # Runtime
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Use strict=True when loading generator weights.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 inference on CUDA. Faster on some GPUs, but may be unstable.",
    )

    return parser.parse_args()


def build_model(args):
    model = BiSSTModel(
        input_nc=3,
        output_nc=3,
        ngf=args.ngf,
        ndf=args.ndf,
        n_blocks=args.n_blocks,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        lambda_gan=args.lambda_gan,
        lambda_nce=args.lambda_nce,
        lambda_idt=args.lambda_idt,
        nce_temperature=args.nce_temperature,
        num_patches=args.num_patches,
        use_mask_predictor=args.use_mask_predictor,
        mask_predictor_type=args.mask_predictor_type,
        mask_base_channels=args.mask_base_channels,
        mask_checkpoint=args.mask_checkpoint,
        freeze_mask_predictor=args.freeze_mask_predictor,
        lambda_mask=args.lambda_mask,
        mask_loss_type=args.mask_loss_type,
        use_embedding_extractor=args.use_embedding_extractor,
        embedding_extractor_type=args.embedding_extractor_type,
        embedding_input_channels=args.embedding_input_channels,
        embedding_hidden_channels=args.embedding_hidden_channels,
        embedding_dim=args.embedding_dim,
        lambda_semantic=args.lambda_semantic,
        lambda_semantic_neg=args.lambda_semantic_neg,
        semantic_loss_type=args.semantic_loss_type,
        detach_semantic_negative_weight=args.detach_semantic_negative_weight,
        device=args.device,
    )

    return model


def load_generator_from_checkpoint(
    model,
    checkpoint_path,
    device,
    strict,
):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError("Checkpoint does not exist: {}".format(checkpoint_path))

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if "models" not in checkpoint:
        raise KeyError("Checkpoint does not contain key 'models'.")

    saved_models = checkpoint["models"]
    current_models = model.get_model_dict()

    if "G" not in current_models:
        raise KeyError("Current BiSSTModel does not contain model key 'G'.")

    if "G" not in saved_models:
        available_keys = sorted(list(saved_models.keys()))
        raise KeyError(
            "Checkpoint does not contain generator key 'G'. Available keys: {}".format(
                available_keys
            )
        )

    netG = current_models["G"]

    load_result = netG.load_state_dict(
        saved_models["G"],
        strict=strict,
    )

    netG.to(device)
    netG.eval()

    return netG, checkpoint, load_result


def frame_to_tensor(
    frame_bgr,
    image_size,
    device,
    use_half,
):
    """
    OpenCV BGR uint8 frame -> tensor [-1, 1], shape [1, 3, H, W].
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(
        frame_rgb,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA,
    )

    array = frame_rgb.astype(np.float32) / 255.0
    array = array * 2.0 - 1.0

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.to(device)

    if use_half:
        tensor = tensor.half()

    return tensor


def tensor_to_bgr_image(tensor):
    """
    Tensor [-1, 1] or [0, 1] -> OpenCV BGR uint8 image.
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.detach().float().cpu()

    if tensor.min().item() < 0.0:
        tensor = (tensor + 1.0) / 2.0

    tensor = tensor.clamp(0.0, 1.0)
    tensor = tensor * 255.0
    tensor = tensor.byte()
    tensor = tensor.permute(1, 2, 0).contiguous()

    image_rgb = tensor.numpy()
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr


def run_generator(netG, input_tensor):
    output = netG(input_tensor)

    if isinstance(output, tuple) or isinstance(output, list):
        output = output[0]

    return output


def draw_text(
    image,
    text,
    x,
    y,
):
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def open_camera(
    camera_id,
    camera_width,
    camera_height,
):
    cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        raise RuntimeError(
            "Failed to open camera_id {}. Try --camera_id 0, 1, or 2.".format(
                camera_id
            )
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    return cap


def main():
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Fall back to CPU.")
        args.device = "cpu"

    if args.half and args.device != "cuda":
        print("Warning: --half only works on CUDA. Disable half precision.")
        args.half = False

    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    print("========== Realtime BiSST Camera Demo ==========")
    print("Project root:", PROJECT_ROOT)
    print("Checkpoint:", args.checkpoint)
    print("Camera id:", args.camera_id)
    print("Camera size:", args.camera_width, args.camera_height)
    print("Image size:", args.image_size)
    print("Display size:", args.display_width, args.display_height)
    print("Device:", args.device)
    print("Half precision:", args.half)

    model = build_model(args)

    netG, checkpoint, load_result = load_generator_from_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=args.device,
        strict=args.strict,
    )

    if args.half:
        netG.half()

    print("")
    print("========== Checkpoint ==========")
    print("Loaded generator G from:", args.checkpoint)
    print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
    print("Checkpoint step:", checkpoint.get("step", "unknown"))

    if not args.strict:
        print("Missing keys:", load_result.missing_keys)
        print("Unexpected keys:", load_result.unexpected_keys)

    cap = open_camera(
        camera_id=args.camera_id,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
    )

    print("")
    print("========== Running ==========")
    print("Press q or ESC to quit.")

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

    last_time = time.time()
    fps = 0.0

    with torch.no_grad():
        while True:
            ret, frame_bgr = cap.read()

            if not ret:
                print("Warning: failed to read frame from camera.")
                break

            input_tensor = frame_to_tensor(
                frame_bgr=frame_bgr,
                image_size=args.image_size,
                device=args.device,
                use_half=args.half,
            )

            fake_tensor = run_generator(
                netG=netG,
                input_tensor=input_tensor,
            )

            fake_bgr = tensor_to_bgr_image(fake_tensor)

            input_display = cv2.resize(
                frame_bgr,
                (args.display_width, args.display_height),
                interpolation=cv2.INTER_AREA,
            )
            fake_display = cv2.resize(
                fake_bgr,
                (args.display_width, args.display_height),
                interpolation=cv2.INTER_AREA,
            )

            now = time.time()
            dt = now - last_time
            last_time = now

            if dt > 0:
                current_fps = 1.0 / dt
                fps = 0.9 * fps + 0.1 * current_fps

            draw_text(input_display, "Camera", 15, 35)
            draw_text(fake_display, "BiSST", 15, 35)

            if args.show_fps:
                draw_text(fake_display, "FPS: {:.1f}".format(fps), 15, 70)

            combined = np.concatenate(
                [input_display, fake_display],
                axis=1,
            )

            cv2.imshow(args.window_name, combined)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

    print("")
    print("Realtime demo finished.")


if __name__ == "__main__":
    
    main()



#python src/infer/realtime_bisst_camera.py --checkpoint checkpoints/bisst/20260617_143913_bisst_real2virtual/latest.pt --camera_id 1 --image_size 256 --show_fps
