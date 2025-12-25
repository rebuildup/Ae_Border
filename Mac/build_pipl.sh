#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
  echo "Usage: $0 <plugin_dir> <plugin_path> <sdk_root>" >&2
  exit 1
fi

PLUGIN_DIR="$1"
PLUGIN_PATH="$2"
SDK_ROOT="$3"

PIPL_SOURCE="$PLUGIN_DIR/BorderPiPL.r"
PIPL_OUTPUT="$PLUGIN_PATH/Contents/Resources/Border.rsrc"

echo "=== PiPL Resource Generation ==="
echo "  PLUGIN_DIR: $PLUGIN_DIR"
echo "  PLUGIN_PATH: $PLUGIN_PATH"
echo "  SDK_ROOT: $SDK_ROOT"
echo "  PIPL_SOURCE: $PIPL_SOURCE"
echo "  PIPL_OUTPUT: $PIPL_OUTPUT"

if [ ! -f "$PIPL_SOURCE" ]; then
  echo "Error: PiPL source not found: $PIPL_SOURCE" >&2
  exit 1
fi

if [ ! -d "$SDK_ROOT/Headers" ]; then
  echo "Error: SDK Headers not found at: $SDK_ROOT/Headers" >&2
  exit 1
fi

# Find AE_General.r location
echo ""
echo "Searching for AE_General.r..."
AE_GENERAL_R=""
for search_path in "$SDK_ROOT/Resources" "$SDK_ROOT/Headers" "$SDK_ROOT"; do
  if [ -f "$search_path/AE_General.r" ]; then
    AE_GENERAL_R="$search_path/AE_General.r"
    echo "  ✓ Found: $AE_GENERAL_R"
    break
  fi
done

if [ -z "$AE_GENERAL_R" ]; then
  echo "  ⚠ AE_General.r not found in standard locations"
  echo "  Searching recursively..."
  AE_GENERAL_R=$(find "$SDK_ROOT" -name "AE_General.r" -type f 2>/dev/null | head -1)
  if [ -n "$AE_GENERAL_R" ]; then
    echo "  ✓ Found: $AE_GENERAL_R"
  else
    echo "  Error: AE_General.r not found anywhere in SDK" >&2
    find "$SDK_ROOT" -name "*.r" -type f | head -10
    exit 1
  fi
fi

AE_GENERAL_DIR=$(dirname "$AE_GENERAL_R")

# Verify required SDK files exist
echo ""
echo "Verifying SDK structure..."
for required_file in "AEConfig.h" "AE_EffectVers.h"; do
  if [ -f "$SDK_ROOT/Headers/$required_file" ]; then
    echo "  ✓ Found: Headers/$required_file"
  else
    echo "  ⚠ Missing: Headers/$required_file"
  fi
done

mkdir -p "$(dirname "$PIPL_OUTPUT")"

# Remove existing resource file to ensure fresh generation
rm -f "$PIPL_OUTPUT"

SYSROOT=$(xcrun --sdk macosx --show-sdk-path 2>/dev/null || xcrun --show-sdk-path 2>/dev/null || true)
STDINC=""
if [ -n "$SYSROOT" ] && [ -d "$SYSROOT/usr/include" ]; then
  STDINC="$SYSROOT/usr/include"
else
  echo "::warning::macOS SDK sysroot include path not found; Rez may not locate stdint.h" >&2
fi

echo ""
echo "Running Rez compiler..."
echo "Include paths:"
if [ -n "$SYSROOT" ]; then
  echo "  -isysroot $SYSROOT"
fi
echo "  -i $SDK_ROOT/Headers"
echo "  -i $SDK_ROOT/Headers/SP"
echo "  -i $SDK_ROOT/Resources"
echo "  -i $AE_GENERAL_DIR"
if [ -n "$STDINC" ]; then
  echo "  -i $STDINC (sysroot std headers)"
fi

# Run Rez with verbose output
REZ_CMD="xcrun Rez -useDF -d AE_OS_MAC -d __MACH__ -d __APPLE__=1 -d __LP64__=1 -d __GNUC__=1 -d __clang__=1 -d A_INTERNAL_TEST_ONE=0 -d TARGET_OS_MAC=1 -d TARGET_OS_IPHONE=0 -d TARGET_OS_IOS=0 -d TARGET_OS_SIMULATOR=0 -d TARGET_OS_WATCH=0 -d TARGET_OS_TV=0 -d TARGET_OS_MACCATALYST=0 -d TARGET_CPU_PPC=0 -d TARGET_CPU_PPC64=0 -d TARGET_CPU_X86=0 -d TARGET_CPU_X86_64=1 -d TARGET_CPU_ARM=0 -d TARGET_CPU_ARM64=0 ${SYSROOT:+-isysroot \"$SYSROOT\"} ${STDINC:+-i \"$STDINC\"} -i \"$SDK_ROOT/Headers\" -i \"$SDK_ROOT/Headers/SP\" -i \"$SDK_ROOT/Resources\" -i \"$AE_GENERAL_DIR\" -o \"$PIPL_OUTPUT\" \"$PIPL_SOURCE\""
echo "Command: $REZ_CMD"

set +e
REZ_OUTPUT=$(xcrun Rez -useDF \
  -d AE_OS_MAC \
  -d __MACH__ -d __APPLE__=1 -d __LP64__=1 -d __GNUC__=1 -d __clang__=1 -d A_INTERNAL_TEST_ONE=0 \
  -d TARGET_OS_MAC=1 -d TARGET_OS_IPHONE=0 -d TARGET_OS_IOS=0 -d TARGET_OS_SIMULATOR=0 \
  -d TARGET_OS_WATCH=0 -d TARGET_OS_TV=0 -d TARGET_OS_MACCATALYST=0 \
  -d TARGET_CPU_PPC=0 -d TARGET_CPU_PPC64=0 -d TARGET_CPU_X86=0 -d TARGET_CPU_X86_64=1 \
  -d TARGET_CPU_ARM=0 -d TARGET_CPU_ARM64=0 \
  ${SYSROOT:+-isysroot "$SYSROOT"} \
  ${STDINC:+-i "$STDINC"} \
  -i "$SDK_ROOT/Headers" \
  -i "$SDK_ROOT/Headers/SP" \
  -i "$SDK_ROOT/Resources" \
  -i "$AE_GENERAL_DIR" \
  -o "$PIPL_OUTPUT" \
  "$PIPL_SOURCE" 2>&1)
REZ_EXIT=$?
set -e

if [ $REZ_EXIT -ne 0 ]; then
  echo "Error: Rez compilation failed (exit code: $REZ_EXIT)" >&2
  echo "Rez output: $REZ_OUTPUT" >&2
  exit 1
fi

if [ -n "$REZ_OUTPUT" ]; then
  echo "Rez output: $REZ_OUTPUT"
fi

if [ ! -f "$PIPL_OUTPUT" ]; then
  echo "Error: Failed to generate PiPL resource" >&2
  exit 1
fi

RSRC_SIZE=$(stat -f%z "$PIPL_OUTPUT" 2>/dev/null || stat -c%s "$PIPL_OUTPUT" 2>/dev/null || ls -la "$PIPL_OUTPUT" | awk '{print $5}')
echo ""
echo "PiPL resource generated successfully: $PIPL_OUTPUT"
echo "  Size: $RSRC_SIZE bytes"

# Verify minimum size (a valid PiPL with both architectures should be ~600+ bytes)
if [ "$RSRC_SIZE" -lt 500 ]; then
  echo "::warning::PiPL resource seems too small ($RSRC_SIZE bytes). Expected ~600+ bytes for dual-architecture."
  echo "This may indicate missing architecture entries."
fi

# Verify the resource contains expected content
echo ""
echo "Verifying PiPL content..."
PIPL_STRINGS=$(strings "$PIPL_OUTPUT" 2>/dev/null || true)
echo "PiPL strings content:"
echo "$PIPL_STRINGS"

FOUND_EFFECT_MAIN=false
FOUND_MI64=false
FOUND_MA64=false

if echo "$PIPL_STRINGS" | grep -q "EffectMain"; then
  FOUND_EFFECT_MAIN=true
  echo "  ✓ Found 'EffectMain' entry point"
fi

if echo "$PIPL_STRINGS" | grep -qi "mi64"; then
  FOUND_MI64=true
  echo "  ✓ Found Intel 64-bit entry (mi64)"
fi

if echo "$PIPL_STRINGS" | grep -qi "ma64"; then
  FOUND_MA64=true
  echo "  ✓ Found ARM 64-bit entry (ma64)"
fi

if [ "$FOUND_EFFECT_MAIN" = false ]; then
  echo "::error::EffectMain not found in PiPL resource!"
  exit 1
fi

if [ "$FOUND_MI64" = false ] || [ "$FOUND_MA64" = false ]; then
  echo "::warning::One or more architecture entries may be missing from PiPL"
fi

echo ""
echo "Hex dump of first 512 bytes:"
xxd "$PIPL_OUTPUT" | head -32

echo ""
echo "=== PiPL Generation Complete ==="
