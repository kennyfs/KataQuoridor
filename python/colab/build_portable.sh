#!/bin/bash -eu
# Builds a KataQuoridor CUDA binary for Colab GPU VMs (Ubuntu 24.04, glibc 2.39, driver 580 / CUDA 13.0, with cuBLAS 13,
# NVRTC 13 and cuDNN 9 installed: checked 2026-10-17) in a CUDA 13.0 + cuDNN 9 docker image, for T4 (75), A100 (80),
# RTX 30xx (86), L4 (89), H100 (90) and RTX PRO 6000 Blackwell (120). Bundles only libzip (Colab's CUDA libraries are
# used); BUNDLE_CUDA=1 also bundles cuBLAS / cuDNN / NVRTC (~1.3 GB) for machines without them.
# Run on a machine with docker (ws4: rootless docker, PATH=~/bin:$PATH):
#   [JOBS=32] build_portable.sh SRC_CPP_DIR OUT_DIR    (JOBS: make jobs; ws4 is shared, at most 32)
# OUT_DIR gets katago, lib/ and run_katago.sh (sets LD_LIBRARY_PATH), and katago-kq-colab.tar.gz of them.
SRC="$(realpath "$1")"; OUT="$(realpath -m "$2")"
IMAGE=${IMAGE:-nvidia/cuda:13.0.2-cudnn-devel-ubuntu24.04}
mkdir -p "$OUT"
docker run --rm -e BUNDLE_CUDA="${BUNDLE_CUDA:-0}" -e JOBS="${JOBS:-32}" -v "$SRC":/src:ro -v "$OUT":/out "$IMAGE" bash -euc '
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq && apt-get install -y -qq cmake g++ make zlib1g-dev libzip-dev > /dev/null
  cp -r /src /build && cd /build
  rm -rf CMakeCache.txt CMakeFiles
  cmake . -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=RelWithDebInfo -DNO_GIT_REVISION=1 -DUSE_AVX2=0 \
    -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90;120" > /out/cmake.log
  make -j"$JOBS" > /out/make.log
  rm -rf /out/lib && mkdir -p /out/lib
  cp katago /out/katago
  # the shared libraries outside the base system, and cuDNN / NVRTC parts loaded at run time
  for l in $(ldd katago | awk "/=>/ {print \$3}" | grep -E "libzip"); do cp -L "$l" /out/lib/; done
  if [ "$BUNDLE_CUDA" = 1 ]; then
    for l in $(ldd katago | awk "/=>/ {print \$3}" | grep -E "cublas|cudnn|nvrtc|cudart"); do cp -L "$l" /out/lib/; done
    for p in /usr/lib/x86_64-linux-gnu/libcudnn*.so.9 /usr/local/cuda/lib64/libnvrtc-builtins.so.*; do cp -L "$p" /out/lib/ 2>/dev/null || true; done
  fi
  chmod -R a+rX /out
'
cat > "$OUT/run_katago.sh" <<'EOS'
#!/bin/bash
# Runs the bundled KataQuoridor binary with its own CUDA / cuDNN libraries.
D="$(dirname "$(realpath "$0")")"
export LD_LIBRARY_PATH="$D/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$D/katago" "$@"
EOS
chmod +x "$OUT/run_katago.sh"
tar -C "$OUT" -czf "$OUT/katago-kq-colab.tar.gz" katago lib run_katago.sh
ls -la "$OUT" "$OUT/lib"; du -sh "$OUT/katago-kq-colab.tar.gz"
