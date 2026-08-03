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
requirements_lock="${repo_root}/configs/dust3r_requirements_lock.txt"
download_priority=()
if command -v nice >/dev/null 2>&1; then
  download_priority+=(nice -n 15)
fi
if command -v ionice >/dev/null 2>&1; then
  download_priority+=(ionice -c 3)
fi

run_download() {
  "${download_wrapper}" "${download_priority[@]}" "$@"
}

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
  run_download git clone --recursive \
    https://github.com/naver/dust3r.git "${dust3r_repo}"
fi
if [[ -n "$(git -C "${dust3r_repo}" status --short)" ]]; then
  echo "Refusing to change a dirty DUSt3R checkout." >&2
  exit 1
fi
run_download git -C "${dust3r_repo}" fetch origin "${dust3r_commit}"
git -C "${dust3r_repo}" checkout --detach "${dust3r_commit}"
run_download git -C "${dust3r_repo}" submodule update --init --recursive
actual_commit="$(git -C "${dust3r_repo}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${dust3r_commit}" ]]; then
  echo "DUSt3R checkout mismatch: ${actual_commit}" >&2
  exit 1
fi

if [[ ! -x "${dust3r_python}" ]]; then
  if command -v uv >/dev/null 2>&1; then
    run_download uv venv --python 3.11 "${repo_root}/.venv-dust3r"
  else
    python3 -m venv "${repo_root}/.venv-dust3r"
  fi
fi
if command -v uv >/dev/null 2>&1; then
  run_download uv pip install \
    --python "${dust3r_python}" \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
  run_download uv pip install \
    --python "${dust3r_python}" \
    -r "${requirements_lock}"
  run_download uv pip install --python "${dust3r_python}" -e "${repo_root}"
else
  run_download "${dust3r_python}" -m pip install \
    --index-url https://download.pytorch.org/whl/cu128 \
    'torch==2.11.0+cu128' 'torchvision==0.26.0+cu128'
  run_download "${dust3r_python}" -m pip install \
    -r "${requirements_lock}"
  run_download "${dust3r_python}" -m pip install -e "${repo_root}"
fi

mkdir -p "${checkpoint_dir}"
config_url="https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt/resolve/main/config.json"
config_sha="3a95c4ea45381e13e998ec059e91f641e3d43b3230e85afb29e46b47e76c1ba5"
if [[ ! -f "${checkpoint_dir}/config.json" ]]; then
  run_download curl -fL --retry 10 --retry-all-errors \
    -o "${checkpoint_dir}/config.json.part" "${config_url}"
  mv -- "${checkpoint_dir}/config.json.part" "${checkpoint_dir}/config.json"
fi
actual_config_sha="$(sha256sum "${checkpoint_dir}/config.json" | awk '{print $1}')"
if [[ "${actual_config_sha}" != "${config_sha}" ]]; then
  echo "DUSt3R config checksum mismatch: ${actual_config_sha}" >&2
  exit 1
fi

model_url="https://huggingface.co/naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt/resolve/main/model.safetensors"
model_path="${checkpoint_dir}/model.safetensors"
model_part="${model_path}.part"
model_size=2284790056
model_sha="7c300a89534113436bde52732d3151212bcbd90f0aa3c8d1496f86d84bfe4b42"
if [[ ! -f "${model_path}" ]]; then
  run_download curl -fL --retry 30 --retry-all-errors --retry-delay 2 \
    --connect-timeout 30 --speed-time 90 --speed-limit 1024 \
    -C - -o "${model_part}" "${model_url}"
  actual_size="$(stat -c %s "${model_part}")"
  if [[ "${actual_size}" -ne "${model_size}" ]]; then
    echo "DUSt3R checkpoint size mismatch: ${actual_size}" >&2
    exit 1
  fi
  actual_model_sha="$(sha256sum "${model_part}" | awk '{print $1}')"
  if [[ "${actual_model_sha}" != "${model_sha}" ]]; then
    echo "DUSt3R checkpoint checksum mismatch: ${actual_model_sha}" >&2
    exit 1
  fi
  mv -- "${model_part}" "${model_path}"
fi
actual_size="$(stat -c %s "${model_path}")"
actual_model_sha="$(sha256sum "${model_path}" | awk '{print $1}')"
if [[ "${actual_size}" -ne "${model_size}" || "${actual_model_sha}" != "${model_sha}" ]]; then
  echo "Existing DUSt3R checkpoint failed size or checksum verification." >&2
  exit 1
fi

"${dust3r_python}" -c \
  "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
test -f "${checkpoint_dir}/config.json"
test -f "${model_path}"
echo "DUSt3R_READY commit=${actual_commit} checkpoint=${checkpoint_dir}"
