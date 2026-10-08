#!/usr/bin/env bash
# Native macOS XcodeGen driver, adapted from rebuildup/ae-plugin-template.
# Never changes the AE host during a build or verification.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${AE_PLUGIN_NAME:?Run via mise, or set AE_PLUGIN_NAME}"
kind="${AE_PLUGIN_KIND:-effect}"
action="${1:-}"
configuration="${2:-Debug}"
build_dir="$root/build/$configuration"
bundle="$build_dir/$name.plugin"
ae_dir="$HOME/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
installed="$ae_dir/$name.plugin"

die() { printf 'error: %s\n' "$*" >&2; exit 1; }

resolve_sdk() {
  local value="" f candidate
  if [[ -n "${AE_SDK_ROOT:-}" ]]; then
    value="$AE_SDK_ROOT"
  elif [[ -f "$root/.env.local" ]]; then
    value="$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$root/.env.local" | head -n 1)"
  elif [[ -n "${AE_SDK_ROOT_FILE:-}" ]]; then
    [[ -f "$AE_SDK_ROOT_FILE" ]] || die "AE_SDK_ROOT_FILE not found: $AE_SDK_ROOT_FILE"
    value="$(head -n 1 "$AE_SDK_ROOT_FILE")"
  fi
  if [[ -n "$value" ]]; then
    value="${value/#\~/$HOME}"
    [[ -f "$value/Examples/Headers/AE_Effect.h" ]] ||
      die "SDK header missing: $value/Examples/Headers/AE_Effect.h"
    (cd "$value" && pwd)
    return
  fi
  for candidate in \
    "$HOME/Documents/projects/adobe/AeSDK" \
    "$HOME/Developer/AfterEffectsSDK" \
    "$HOME/Developer/AE_SDK" \
    "$HOME/sdk/AfterEffectsSDK" \
    "/opt/AfterEffectsSDK"
  do
    if [[ -f "$candidate/Examples/Headers/AE_Effect.h" ]]; then
      (cd "$candidate" && pwd)
      return
    fi
  done
  die "Set AE_SDK_ROOT or put the absolute SDK path in .env.local"
}

generate() {
  export AE_SDK_ROOT="$(resolve_sdk)"
  command -v xcodegen >/dev/null || die "xcodegen missing (run mise install)"
  xcodegen generate --spec "$root/project.yml" --project "$root"
}

build() {
  [[ "$configuration" == Debug || "$configuration" == Release ]] ||
    die "Configuration must be Debug or Release"
  export AE_SDK_ROOT="$(resolve_sdk)"
  generate
  mkdir -p "$build_dir"
  xcodebuild \
    -project "$root/$name.xcodeproj" \
    -scheme "$name" \
    -configuration "$configuration" \
    -derivedDataPath "$root/build/DerivedData" \
    CONFIGURATION_BUILD_DIR="$build_dir" \
    ONLY_ACTIVE_ARCH="$([[ "$configuration" == Debug ]] && echo YES || echo NO)" \
    build
  [[ -d "$bundle" ]] || die "Build succeeded but bundle is missing: $bundle"
  echo "Built: $bundle"
}

verify() {
  [[ -d "$bundle" ]] || die "Missing bundle: $bundle; run mise run build first"
  local plist="$bundle/Contents/Info.plist"
  local binary="$bundle/Contents/MacOS/$name"
  local rsrc="$bundle/Contents/Resources/$name.rsrc"
  [[ -f "$plist" ]] || die "Info.plist missing"
  [[ -f "$binary" ]] || die "Executable missing: $binary"
  [[ -s "$rsrc" ]] || die "PiPL resource missing: $rsrc"
  /usr/libexec/PlistBuddy -c "Print :CFBundleExecutable" "$plist" | grep -Fxq "$name" ||
    die "CFBundleExecutable mismatch"
  local package_type signature entry
  if [[ "$kind" == aegp ]]; then
    package_type=AEgx
    signature=FXFL
    entry=EntryPointFunc
  else
    package_type=eFKT
    signature=FXTC
    entry=EffectMain
  fi
  [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundlePackageType' "$plist")" == "$package_type" ]] ||
    die "CFBundlePackageType does not match $kind"
  [[ "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleSignature' "$plist")" == "$signature" ]] ||
    die "CFBundleSignature does not match $kind"
  lipo -verify_arch arm64 "$binary" || die "Missing arm64 binary slice"
  if [[ "$configuration" == Release ]]; then
    lipo -verify_arch x86_64 "$binary" || die "Missing x86_64 binary slice"
  fi
  nm -gU "$binary" | grep -Eq "_${entry}$" ||
    die "Expected exported symbol $entry not found"
  grep -aq 'ma64' "$rsrc" || die "PiPL is missing CodeMacARM64"
  codesign --verify --strict "$bundle" || die "Bundle signature invalid"
  if otool -L "$binary" | sed '1d' | grep -q '^[[:space:]]*/Users/'; then
    die "Absolute /Users link dependency in bundle"
  fi
  echo "Verified bundle (not AE-host-tested): $bundle"
}

install_plugin() {
  verify
  mkdir -p "$ae_dir"
  if [[ -e "$installed" ]]; then
    mv "$installed" "$installed.backup-$(date +%Y%m%d%H%M%S)"
  fi
  ditto "$bundle" "$installed"
  codesign --force --sign - --timestamp=none "$installed"
  codesign --verify --strict "$installed"
  echo "Installed: $installed (restart After Effects)"
}

case "$action" in
  sdk-path) resolve_sdk ;;
  generate) generate ;;
  build) build ;;
  verify) verify ;;
  install) install_plugin ;;
  uninstall)
    if [[ -e "$installed" ]]; then
      mv "$installed" "$installed.backup-$(date +%Y%m%d%H%M%S)"
      echo "Uninstalled $name (backup retained)"
    else
      echo "Not installed: $name"
    fi
    ;;
  reveal) mkdir -p "$ae_dir"; open "$ae_dir" ;;
  clean)
    rm -rf "$root/build" "$root/$name.xcodeproj"
    echo "Removed generated build artifacts"
    ;;
  *) die "Usage: $0 {sdk-path|generate|build|verify|install|uninstall|reveal|clean} [Debug|Release]" ;;
esac
