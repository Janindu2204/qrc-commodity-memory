#!/usr/bin/env bash
# Check every registration hash file. Prints only problems; see README "Known hash exceptions".
cd "$(dirname "$0")/notebooks/qlstm_copper" || exit 1
for f in *.sha256 hw/*.sha256; do
  d=$(dirname "$f"); b=$(basename "$f")
  if [ "$(awk '{print NF}' "$f" | head -1)" = "1" ]; then           # bare hash of the matching .json
    j="${b%.sha256}.json"
    [ "$(tr -d ' \n' < "$f")" = "$(cd "$d" && sha256sum "$j" | cut -c1-64)" ] && echo "$f: OK" || echo "$f: FAILED"
    continue
  fi
  # one hashed theory note lived outside the code tree; it is in notes/
  (cd "$d" && sed 's#\.\./\.\./\.\./\.\./PROGRESS/#../../notes/#' "$b" | sha256sum -c --quiet - 2>&1 | sed "s#^#$f: #") || true
done
