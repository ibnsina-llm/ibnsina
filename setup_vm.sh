#!/bin/bash
set -eux
[ -f /var/tmp/startup-done ] && exit 0
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y build-essential python3 python3-pip python3-venv p7zip-full poppler-utils tesseract-ocr tesseract-ocr-fas tesseract-ocr-eng zstd pigz pbzip2 xz-utils git tmux jq curl unzip pv htop
python3 -m venv /opt/pipe
/opt/pipe/bin/pip install -U pip wheel
/opt/pipe/bin/pip install "datatrove[processing]" fasttext-wheel tokenizers pyarrow zstandard orjson xxhash regex tiktoken wikiextractor trafilatura pymupdf numpy pandas tqdm huggingface_hub
mkdir -p /data /models && chmod 1777 /data
curl -sL -o /models/lid.176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
chmod -R a+rX /opt/pipe /models
cat > /etc/profile.d/pipe.sh <<'EOP'
export PATH=/opt/pipe/bin:$PATH
export CORPUS_BUCKET=${CORPUS_BUCKET:-gs://YOUR-BUCKET}
export LID_MODEL=/models/lid.176.bin
EOP
touch /var/tmp/startup-done
