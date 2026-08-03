#!/usr/bin/env bash
set -euo pipefail

# Install the frozen non-commercial DUSt3R confirmatory environment on my5090.
# Every network operation is wrapped by the process-scoped download proxy.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
dust3r_commit="4c24a6ebf04809f2cfe59915e51779c8984aaa40"
dust3r_repo="${repo_root}/third_party/dust3r"
dust3r_python="${repo_root}/.venv-dust3r/bin/python"
checkpoint_dir="${repo_root}/checkpoints/dust3r-512-dpt"
download_wrapper="${repo_root}/scripts/with_download_proxy.sh"

if [[ "${CAMCANON3R_ACCEPT_DUST3R_NONCOMMERCIAL:-}" != "1" ]]; then
  echo "DUSt3R is CC BY-NC-SA 4.0 and its weights have dataset-license terms." >&2
  echo "For this academic, non-commercial study, rerun with:" >&2
  echo "  CAMCANON3R_ACCEPT_DUST3R_NONCOMMERCIAL=1 $0" >&2
  exit 2
fi

mkdir -p "${repo_root}/third_party" "${repo_root}/checkpoints"
if [[ -e "${dust3r_repo}" && ! -d "${dust3r_repo}/.git" ]]; then
  echo "Refusing to replace non-Git path: ${dust3r_repo}" >&2
  exit 1
fi
if [[ ! -d "${dust3r_repo}/.git" ]]; then
  "${download_wrapper}" git clone --recursive \
    https://github.com/naver/dust3r.git "${dust3r_repo}"
fi
if [[ -n "$(git -C "${dust3r_repo}" status --short)" ]]; then
  echo "Refusing to change a dirty DUSt3R checkout." >&2
  exit 1
fi
"${download_wrapper}" git -C "${dust3r_repo}" fetch origin "${dust3r_commit}"
git -C "${dust3r_repo}" checkout --detach "${dust3r_commit}"
"${download_wrapper}" git -C "${dust3r_repo}" submodule update --init --recursive
actual_commit="$(git -C "${dust3r_repo}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${dust3r_commit}" ]]; then
  echo "DUSt3R checkout mismatch: ${actual_commit}" >&2
  exit 1
fi

if [[ ! -x "${dust3r_python}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 "${repo_root}/.venv-dust3r"
  else
    python3 -m venv "${repo_root}/.venv-dust3r"
  fi
fi
if command -v uv >/dev/null 2>&1; then
  "${download_wrapper}" uv pip install \
    --python "${dust3r_python}" \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision
  "${download_wrapper}" uv pip install \
    --python "${dust3r_python}" \
    -r "${dust3r_repo}/requirements.txt"
  uv pip install --python "${dust3r_python}" -e "${repo_root}"
else
  "${download_wrapper}" "${dust3r_python}" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision
  "${download_wrapper}" "${dust3r_python}" -m pip install \
    -r "${dust3r_repo}/requirements.txt"
  "${dust3r_python}" -m pip install -e "${repo_root}"
fi

mkdir -p "${checkpoint_dir}"
"${download_wrapper}" "${dust3r_python}" -c \
  "from huggingface_hub import snapshot_download; snapshot_download(repo_id='naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt', local_dir='${checkpoint_dir}')"

"${dust3r_python}" -c \
  "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
test -f "${checkpoint_dir}/config.json"
test -f "${checkpoint_dir}/model.safetensors"
echo "DUSt3R_READY commit=${actual_commit} checkpoint=${checkpoint_dir}"
