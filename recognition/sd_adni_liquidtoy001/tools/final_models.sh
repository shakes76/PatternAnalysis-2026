# Weights of every model in docs/final-evaluation-protocol.md, relative to the
# final run directory (runs/final-20261019). Sourced by the final_*.sbatch jobs.
declare -A WEIGHTS=(
    [e0]=outputs/e0-adapter/lora-e0-step-20000.safetensors
    [e2]=../e2-20261013/outputs/e2-656467/weights/unet-ema-epoch-10.safetensors
    [r1e2]=../r1-20261017/outputs/r1-e2-669198/weights/unet-ema-epoch-10.safetensors
    [e3]=../e3-20261015/outputs/e3-660842/weights/lora-ema-epoch-10.safetensors
    [r1e3]=../r1-20261017/outputs/r1-e3-672769/weights/lora-ema-epoch-10.safetensors
    [b3]=../b3-20261015/outputs/b3-668934/weights/unet-ema-epoch-98.safetensors
)
E0_CHECKPOINT=../full20k-20261004/outputs/full-629931/step-20000/checkpoint.pt
B1_CHECKPOINT=../baseline128-20261005/outputs/baseline-vae-632570/epoch-5.pt
B2_CHECKPOINT=../b2-20261015/outputs/b2-663302/final.pt
B2B_AD=../b2b-20261017/outputs/b2b-AD-672812/final.pt
B2B_NC=../b2b-20261017/outputs/b2b-NC-672813/final.pt
# E2's private list of the two defective training scans it excluded.
EXCLUDED_SCANS=../e2-20261013/outputs/e2-656467/private-excluded-scans.json

# Arguments that load a diffusion model's weights in predict.py,
# membership.py and interpolate_final.py; empty for the SD control.
diffusion_args() {
    case "$1" in
        sd) model_args=() ;;
        b3) model_args=(--weights "${WEIGHTS[b3]}" --unet-init scratch) ;;
        e0 | e2 | r1e2 | e3 | r1e3) model_args=(--weights "${WEIGHTS[$1]}") ;;
        *) echo "not a diffusion model: $1" >&2; return 2 ;;
    esac
}
