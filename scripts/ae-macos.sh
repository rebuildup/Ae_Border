#!/usr/bin/env bash
#
# ae-macos.sh — native macOS build/verify/install driver for this plug-in.
#
# Derived from rebuildup/ae-plugin-template. The template owns the *practice*
# (external SDK reference, generated Xcode project, build separated from
# install, bundle verification); this script is the per-repository expression
# of it, because each plug-in has its own source layout.
#
# Invariants this script exists to keep:
#
#   * The Adobe SDK is referenced from outside the repository. Nothing is copied.
#   * `project.yml` is the source of truth for build settings; the generated
#     .xcodeproj is an artifact.
#   * Building and verifying never touch the After Effects installation.
#   * Installing is explicit, backs up whatever it replaces, and re-signs after
#     copying (copying invalidates a signature).
#   * "builds" and "the host accepts it" are separate claims. verify.sh-family
#     checks never claim the latter.

set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${AE_PLUGIN_NAME:=}"
if [[ -z "$AE_PLUGIN_NAME" ]]; then
  printf 'error: AE_PLUGIN_NAME is not set. Run through mise, or export it.\n' >&2
  exit 1
fi
: "${AE_PLUGIN_KIND:=effect}"   # effect | aegp

action="${1:-}"
configuration="${2:-Debug}"
sdk_marker="Examples/Headers/AE_Effect.h"

build_dir="$root/build/$configuration"
bundle="$build_dir/$AE_PLUGIN_NAME.plugin"
ae_dir="$HOME/Library/Application Support/Adobe/Common/Plug-ins/7.0/MediaCore"
installed="$ae_dir/$AE_PLUGIN_NAME.plugin"

die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
note() { printf '  %s\n' "$*"; }

# --------------------------------------------------------------------------
# SDK resolution
# --------------------------------------------------------------------------

# _first_value <file>
_first_value() {
  sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$1" | head -n1 | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

resolve_sdk() {
  local value=""

  if [[ -n "${AE_SDK_ROOT:-}" ]]; then
    value="$AE_SDK_ROOT"
  elif [[ -f "$root/.env.local" ]]; then
    value="$(_first_value "$root/.env.local" || true)"
  elif [[ -n "${AE_SDK_ROOT_FILE:-}" ]]; then
    [[ -f "$AE_SDK_ROOT_FILE" ]] || die "AE_SDK_ROOT_FILE is not a file: $AE_SDK_ROOT_FILE"
    value="$(_first_value "$AE_SDK_ROOT_FILE" || true)"
  fi

  if [[ -n "$value" ]]; then
    value="${value/#\~/$HOME}"
    [[ -f "$value/$sdk_marker" ]] ||
      die "SDK header missing: $value/$sdk_marker
The path you configured does not point at an unpacked After Effects SDK."
    (cd "$value" && pwd)
    return
  fi

  local candidate
  for candidate in \
    "$HOME/Documents/projects/adobe/AeSDK" \
    "$HOME/Developer/AfterEffectsSDK" \
    "$HOME/Developer/AE_SDK" \
    "$HOME/sdk/AfterEffectsSDK" \
    "/opt/AfterEffectsSDK" \
    "/usr/local/share/AfterEffectsSDK"
  do
    if [[ -f "$candidate/$sdk_marker" ]]; then
      (cd "$candidate" && pwd)
      return
    fi
  done

  die "Could not locate the After Effects SDK.
Set AE_SDK_ROOT, or write the absolute SDK path to $root/.env.local:
    /absolute/path/to/AdobeAfterEffectsSDK_26.5_MacOS"
}

# --------------------------------------------------------------------------
# Identity (what the plug-in says it is — read from the PiPL source)
# --------------------------------------------------------------------------

pipl_source() {
  local f
  f="$(find "$root" -maxdepth 2 -name '*PiPL.r' -not -path '*/build/*' -print -quit 2>/dev/null || true)"
  [[ -n "$f" ]] || die "No PiPL resource source found at the repository root"
  printf '%s\n' "$f"
}

# Entry point named in the PiPL for macOS, e.g. EffectMain / EntryPointFunc.
pipl_entry_points() {
  grep -oE 'CodeMac(ARM64|Intel64)[[:space:]]*\{[[:space:]]*"[A-Za-z_][A-Za-z_0-9]*"' "$1" \
    | sed -E 's/.*"([^"]+)"/\1/' | sort -u
}

# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------

do_generate() {
  export AE_SDK_ROOT="$(resolve_sdk)"
  command -v xcodegen >/dev/null 2>&1 ||
    die "xcodegen not found. Run 'mise install' (mise puts it on PATH for tasks)."
  xcodegen generate --spec "$root/project.yml" --project "$root"
}

do_build() {
  [[ "$configuration" == Debug || "$configuration" == Release ]] ||
    die "configuration must be Debug or Release (got: '$configuration')"
  export AE_SDK_ROOT="$(resolve_sdk)"

  do_generate >/dev/null
  mkdir -p "$build_dir"

  local only_active=NO
  [[ "$configuration" == Debug ]] && only_active=YES

  xcodebuild \
    -project "$root/$AE_PLUGIN_NAME.xcodeproj" \
    -scheme "$AE_PLUGIN_NAME" \
    -configuration "$configuration" \
    -derivedDataPath "$root/build/DerivedData" \
    CONFIGURATION_BUILD_DIR="$build_dir" \
    ONLY_ACTIVE_ARCH="$only_active" \
    build

  [[ -d "$bundle" ]] || die "xcodebuild reported success but $bundle is missing"
  printf 'Built %s -> %s\n' "$configuration" "$bundle"
}

_plist() { /usr/libexec/PlistBuddy -c "Print $1" "$bundle/Contents/Info.plist" 2>/dev/null || true; }

do_verify() {
  [[ -d "$bundle" ]] || die "Missing $bundle. Run 'mise run build${configuration:+ }' first."

  local plist="$bundle/Contents/Info.plist"
  local binary="$bundle/Contents/MacOS/$AE_PLUGIN_NAME"
  local rsrc="$bundle/Contents/Resources/$AE_PLUGIN_NAME.rsrc"
  local failures=0

  ok()   { printf '  ok    %s\n' "$*"; }
  bad()  { printf '  FAIL  %s\n' "$*" >&2; failures=$((failures + 1)); }
  must() { if "$@"; then :; else bad "$*"; fi; }

  printf 'Verifying %s (%s, kind=%s)\n\n' "$AE_PLUGIN_NAME" "$configuration" "$AE_PLUGIN_KIND"

  # -- layout -------------------------------------------------------------
  must test -f "$plist"    || true
  must test -f "$binary"   || true
  must test -s "$rsrc"     || true
  [[ -f "$plist" && -f "$binary" ]] || {
    printf 'verify: bundle is incomplete; aborting further checks\n' >&2
    exit 1
  }

  # -- Info.plist identity ------------------------------------------------
  local expect_type expect_sig
  if [[ "$AE_PLUGIN_KIND" == aegp ]]; then
    expect_type=AEgx; expect_sig=FXFL
  else
    expect_type=eFKT; expect_sig=FXTC
  fi

  if [[ "$(_plist :CFBundleExecutable)" == "$AE_PLUGIN_NAME" ]]; then
    ok "CFBundleExecutable matches the bundle name"
  else
    bad "CFBundleExecutable is '$(_plist :CFBundleExecutable)', expected '$AE_PLUGIN_NAME'"
  fi

  if [[ "$(_plist :CFBundlePackageType)" == "$expect_type" ]]; then
    ok "CFBundlePackageType is $expect_type"
  else
    bad "CFBundlePackageType is '$(_plist :CFBundlePackageType)', expected '$expect_type' ($AE_PLUGIN_KIND)"
  fi

  if [[ "$(_plist :CFBundleSignature)" == "$expect_sig" ]]; then
    ok "CFBundleSignature is $expect_sig"
  else
    bad "CFBundleSignature is '$(_plist :CFBundleSignature)', expected '$expect_sig' ($AE_PLUGIN_KIND)"
  fi

  # -- architectures ------------------------------------------------------
  if lipo -verify_arch arm64 "$binary" >/dev/null 2>&1; then
    ok "arm64 slice present"
  else
    bad "no arm64 slice in $(lipo -archs "$binary" 2>/dev/null)"
  fi

  if [[ "$configuration" == Release ]]; then
    if lipo -verify_arch x86_64 "$binary" >/dev/null 2>&1; then
      ok "x86_64 slice present"
    else
      bad "Release has no x86_64 slice; the PiPL advertises CodeMacIntel64"
    fi
  else
    ok "Debug is single-architecture by design; x86_64 is checked in Release"
  fi

  # -- symbols ------------------------------------------------------------
  local symbols
  symbols="$(nm -gU "$binary" 2>/dev/null || true)"

  if [[ "$AE_PLUGIN_KIND" == aegp ]]; then
    if grep -Eq '_EntryPointFunc$' <<<"$symbols"; then
      ok "exports EntryPointFunc"
    else
      bad "does not export EntryPointFunc"
    fi
  else
    if grep -Eq '_EffectMain$' <<<"$symbols"; then ok "exports EffectMain"; else bad "does not export EffectMain"; fi
    if grep -Eq '_PluginDataEntryFunction2$' <<<"$symbols"; then
      ok "exports PluginDataEntryFunction2"
    else
      bad "does not export PluginDataEntryFunction2"
    fi
  fi

  # -- PiPL <-> binary ----------------------------------------------------
  local pipl
  pipl="$(pipl_source)"

  local declared
  declared="$(pipl_entry_points "$pipl")"
  if [[ -z "$declared" ]]; then
    bad "PiPL declares no CodeMacARM64/CodeMacIntel64 entry point"
  else
    local entry
    while IFS= read -r entry; do
      [[ -n "$entry" ]] || continue
      if grep -Eq "_${entry}\$" <<<"$symbols"; then
        ok "PiPL entry point $entry is exported"
      else
        bad "PiPL names $entry but the binary does not export it"
      fi
    done <<<"$declared"
  fi

  for atom in ma64 mi64; do
    if grep -aq "$atom" "$rsrc"; then ok "compiled PiPL contains $atom"; else bad "compiled PiPL is missing $atom"; fi
  done

  # Match names identify effects inside a saved .aep. AEGPs are registered by
  # bundle, so they legitimately have none.
  if [[ "$AE_PLUGIN_KIND" == effect ]]; then
    if grep -aq 'eMNA' "$rsrc"; then
      ok "compiled PiPL contains a match name (eMNA)"
    else
      bad "compiled PiPL is missing the match name; the effect will not be findable in a saved project"
    fi
  else
    if grep -aq 'eMNA' "$rsrc"; then
      note "AEGP PiPL contains a match name (eMNA); unexpected for an AEGP, worth a look"
    else
      ok "no match name, as expected for an AEGP"
    fi
  fi

  # -- outflags consistency ----------------------------------------------
  # AE reads Global_OutFlags from the PiPL for host compatibility and compares
  # them against what PF_Cmd_GLOBAL_SETUP reports. A mismatch does not fail
  # loudly; it degrades behaviour. Checking it needs the SDK's bit values.
  if [[ "$AE_PLUGIN_KIND" == effect ]]; then
    local sdk="${AE_SDK_ROOT:-}"
    if [[ -z "$sdk" ]] || [[ ! -f "$sdk/$sdk_marker" ]]; then
      sdk="$(resolve_sdk)"
    fi
    if command -v python3 >/dev/null 2>&1; then
      AE_SDK_ROOT="$sdk" python3 - "$root" "$pipl" <<'PY'
import os, re, sys

repo, pipl_path = sys.argv[1], sys.argv[2]
sdk = os.environ["AE_SDK_ROOT"]
hdr = os.path.join(sdk, "Examples/Headers/AE_Effect.h")
src_files = []
for dirpath, dirnames, filenames in os.walk(repo):
    dirnames[:] = [d for d in dirnames if d not in ("build", ".git", "Mac", "Win", "x64")]
    src_files += [os.path.join(dirpath, f) for f in filenames if f.endswith((".cpp", ".h"))]

bits = {}
with open(hdr, encoding="utf-8", errors="replace") as fh:
    for line in fh:
        m = re.search(r"\b(PF_OutFlag2?_[A-Z0-9_]+)\s*=\s*([0-9]+)L?\s*<<\s*(\d+)", line)
        if m:
            bits[m.group(1)] = int(m.group(2)) << int(m.group(3))

def declared_ids(prefix, text):
    out = []
    for m in re.finditer(re.escape(prefix) + r"\s*=\s*([^;]+);", text):
        for tok in re.findall(r"PF_OutFlag2?_[A-Z0-9_]+", m.group(1)):
            out.append(tok)
    return out

with open(pipl_path, encoding="utf-8", errors="replace") as fh:
    pipl = fh.read()

globaltext = None
for path in src_files:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    if "PF_Cmd_GLOBAL_SETUP" in text or "out_flags" in text:
        globaltext = text
        break

def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return text

# Concatenate every source so that macros defined in a header can be resolved
# while reading an assignment in a .cpp. Reading only the .cpp that contains the
# assignment hides the version macros it depends on.
chunks = []
for path in src_files:
    with open(path, encoding="utf-8", errors="replace") as fh:
        chunks.append(strip_comments(fh.read()))
globaltext = "\n".join(chunks)

if "out_data->out_flags" not in globaltext:
    print("  note  could not locate a PF_Cmd_GLOBAL_SETUP source; PiPL cross-check skipped")
    sys.exit(2)

def resolve(expr, depth=0):
    """Turn a C++ expression into an integer, resolving named flags and macros."""
    expr = expr.strip()
    m = re.search(r"0x[0-9a-fA-F]+", expr)
    if m:
        return int(m.group(0), 16)
    m = re.search(r"(?<![A-Za-z_0-9])(\d+)(?![A-Za-z_0-9])", expr)
    if m and depth == 0:
        # A plain decimal literal, e.g. a precomputed version constant.
        return int(m.group(1))
    if depth < 3:
        # A bare identifier or symbol: find where it is defined and resolve that.
        for ident in re.findall(r"\b[A-Za-z_][A-Za-z_0-9]*\b", expr):
            if ident in ("PF_OutFlag", "PF_OutFlag2"):
                continue
            dm = re.search(r"\b" + re.escape(ident) + r"\b[^=;]*=\s*([^;]+);", globaltext)
            if dm:
                value = resolve(dm.group(1), depth + 1)
                if value is not None:
                    return value
    total = 0
    found = False
    for ident in re.findall(r"PF_OutFlag2?_[A-Z0-9_]+", expr):
        if ident not in bits:
            return None
        total |= bits[ident]
        found = True
    return total if found else None

# The PiPL carries /* ... */ comments between a property name and its value.
# Parsing the raw text silently fails to match those, which previously turned a
# real mismatch into a skipped check reported as success.
pipl_plain = strip_comments(pipl)

def encode_version(vers, subvers, bugfix, stage, build):
    return ((((vers >> 3) & 0xF) << 26) | ((vers & 0x7) << 19) | (subvers << 15)
            | (bugfix << 11) | (stage << 9) | build)

STAGES = {"PF_Stage_DEVELOP": 0, "PF_Stage_ALPHA": 1,
          "PF_Stage_BETA": 2, "PF_Stage_RELEASE": 3}

problems = []
undecided = []

# --- PiPL effect version vs the version the code reports -------------------
vm = re.search(r"AE_Effect_Version\s*\{\s*([A-Za-z_][A-Za-z_0-9]*|0x[0-9a-fA-F]+|\d+)", pipl_plain)
sm = re.search(r"\bout_data\s*->\s*my_version\s*=\s*PF_VERSION\s*\(([^)]*)\);", globaltext)
literal_code_version = None
if not sm:
    # my_version is often assigned through a macro, e.g.
    #   out_data->my_version = BORDER_VERSION_VALUE;
    #   #define BORDER_VERSION_VALUE PF_VERSION(MAJOR, MINOR, ...)
    alias = re.search(r"\bout_data\s*->\s*my_version\s*=\s*([A-Za-z_][A-Za-z_0-9]*)\s*;", globaltext)
    if alias:
        inner = re.search(r"#define\s+" + re.escape(alias.group(1)) + r"\s+PF_VERSION\s*\(([^)]*)\)", globaltext)
        if inner:
            sm = inner
        else:
            # A precomputed literal, e.g. #define FOO_VERSION_VALUE 524289.
            # Comparing it against the copy in the PiPL still catches drift
            # between the two definitions.
            lit = re.search(r"#define\s+" + re.escape(alias.group(1)) + r"\s+(0x[0-9a-fA-F]+|\d+)", globaltext)
            if lit:
                literal_code_version = int(lit.group(1), 0)
if vm and (sm is not None or literal_code_version is not None):
    raw = vm.group(1).strip()
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", raw):
        # A symbolic value: either the source's macro, or the literal the PiPL
        # defines for itself (Rez cannot expand macros, so the literal is
        # duplicated in the .r file and must match the header).
        # Look for a #define first. A looser "NAME ... = ...;" pattern also
        # matches enum members that merely reference the symbol.
        dm = re.search(r"#define\s+" + re.escape(raw) + r"\s+([^\n]+)", globaltext)
        if dm:
            pv = resolve(dm.group(1))
        else:
            pm = re.search(r"#define\s+" + re.escape(raw) + r"\s+(0x[0-9a-fA-F]+|-?\d+)", pipl_plain)
            if pm:
                pv = int(pm.group(1), 0)
            else:
                dm = re.search(r"\b" + re.escape(raw) + r"\b\s*=\s*([^;]+);", globaltext)
                pv = None if not dm else resolve(dm.group(1))
    else:
        pv = int(raw, 16) if raw.startswith("0x") else int(raw)

    def resolve_number(tok):
        """Resolve a version component, following #define indirection."""
        tok = tok.strip()
        if tok in STAGES:
            return STAGES[tok]
        try:
            return int(tok, 0)
        except ValueError:
            pass
        # Follow one level of #define indirection, e.g.
        #   #define STAGE_VERSION PF_Stage_DEVELOP
        #   #define PF_Stage_DEVELOP 0   (or the name in STAGES)
        dm = re.search(r"#define\s+" + re.escape(tok) + r"\s+([^\n]+)", globaltext)
        if not dm:
            return None
        value = dm.group(1).strip()
        if value in STAGES:
            return STAGES[value]
        try:
            return int(value, 0)
        except ValueError:
            return None

    if literal_code_version is not None:
        cv = literal_code_version
        how = "the literal assigned to out_data->my_version"
    else:
        parts = [p.strip() for p in sm.group(1).split(",")]
        try:
            nums = [resolve_number(part) for part in parts]
            cv = None if any(n is None for n in nums) else encode_version(*nums)
        except Exception as exc:
            cv = None
            undecided.append(f"could not evaluate my_version ({exc})")
        if cv is None and not undecided:
            undecided.append("my_version components could not be resolved")
        how = "PF_VERSION(%s)" % sm.group(1).strip()

    if pv is None or cv is None:
        undecided.append("PiPL effect version could not be evaluated on both sides")
    elif pv != cv:
        problems.append(
            f"AE_Effect_Version: PiPL says 0x{pv:08X} but PF_Cmd_GLOBAL_SETUP "
            f"reports 0x{cv:08X} via {how}")
else:
    undecided.append("PiPL effect version or my_version assignment not found")

# --- PiPL outflags vs PF_Cmd_GLOBAL_SETUP ---------------------------------
for pipl_prop, prefix in (("AE_Effect_Global_OutFlags", "out_flags"),
                          ("AE_Effect_Global_OutFlags_2", "out_flags2")):
    pm = re.search(re.escape(pipl_prop) + r"\s*\{([^}]*)\}", pipl_plain, re.S)
    om = re.search(r"\bout_data\s*->\s*" + re.escape(prefix) + r"(?!\d)\s*=\s*([^;]+);", globaltext)
    if not pm:
        undecided.append(f"{pipl_prop} not found in the PiPL")
        continue
    if not om:
        undecided.append(f"{prefix} is never assigned in PF_Cmd_GLOBAL_SETUP")
        continue
    hv = re.search(r"0x[0-9a-fA-F]+|\d+", pm.group(1))
    if not hv:
        undecided.append(f"{pipl_prop} has no literal value")
        continue
    pipl_value = int(hv.group(0), 16) if hv.group(0).startswith("0x") else int(hv.group(0))
    source_value = resolve(om.group(1))
    if source_value is None:
        undecided.append(f"could not evaluate {prefix} = {om.group(1).strip()}")
        continue
    if source_value != pipl_value:
        problems.append(
            f"{pipl_prop}: PiPL says 0x{pipl_value:08x} but PF_Cmd_GLOBAL_SETUP "
            f"sets 0x{source_value:08x} ({om.group(1).strip()})")

for u in undecided:
    print(f"  note  {u}")
for p in problems:
    print(f"  FAIL  {p}")
if problems:
    sys.exit(1)
if undecided:
    print("  note  PiPL/source cross-check incomplete; agreement NOT established")
    sys.exit(2)
print("  ok    PiPL version and outflags agree with PF_Cmd_GLOBAL_SETUP")
sys.exit(0)
PY
      # The snippet exits 0 on agreement, 1 on disagreement, 2 when it cannot
      # decide. Only a disagreement is a failure; an undecidable check is
      # reported but must not silently pass as agreement.
      rc=$?
      if [[ $rc -eq 1 ]]; then
        bad "PiPL version or outflags disagree with PF_Cmd_GLOBAL_SETUP"
      elif [[ $rc -eq 2 ]]; then
        note "PiPL cross-check could not decide; agreement is NOT established"
      fi
    else
      note "python3 unavailable; the PiPL outflags cross-check did not run"
    fi
  fi

  # -- signature ----------------------------------------------------------
  if codesign --verify --strict "$bundle" >/dev/null 2>&1; then
    ok "signature is valid"
  else
    bad "signature does not verify; macOS 15+ will refuse to load this bundle"
  fi

  local siginfo
  siginfo="$(codesign -dv "$bundle" 2>&1 || true)"
  if grep -q 'Signature=adhoc' <<<"$siginfo"; then
    ok "ad-hoc signed, as expected for local development"
  else
    bad "expected an ad-hoc signature for local development"
  fi

  # -- link dependencies --------------------------------------------------
  # otool -L prints one "path (architecture N):" header per slice, and that
  # header contains this machine's absolute path to the bundle. Filtering on the
  # leading tab keeps only real dependency lines; dropping just the first line
  # would leave the second architecture's header and look like a /Users link.
  local deps
  deps="$(otool -L "$binary" | grep -E '^[[:space:]]' || true)"
  if grep -q '^[[:space:]]*/Users/' <<<"$deps"; then
    bad "links an absolute /Users path; the bundle will not load on another machine"
  else
    ok "no absolute /Users paths in link dependencies"
  fi
  if grep -q '@executable_path\|@loader_path' <<<"$deps"; then
    bad "links through @executable_path/@loader_path; AE does not provide those inside a plug-in bundle"
  else
    ok "no @executable_path/@loader_path dependencies"
  fi
  if grep -q 'OpenMP\|libomp\|libiomp' <<<"$deps"; then
    note "depends on an OpenMP runtime; confirm it is present wherever this plug-in ships"
  fi

  printf '\n'
  if [[ $failures -gt 0 ]]; then
    printf 'verify: %d check(s) failed\n' "$failures" >&2
    exit 1
  fi
  printf 'verify: all checks passed (%s)\n' "$configuration"
  printf 'This is bundle-level verification. It is NOT evidence that After Effects\n'
  printf 'loads the plug-in or that the effect behaves correctly.\n'
}

do_install() {
  do_verify

  # Match only the AE application binary. A plain `pgrep -f 'Adobe After
  # Effects'` also matches the helper processes that outlive the app
  # (crashpad_handler, dynamiclinkmanager), which would block installation for no
  # reason.
  if pgrep -f 'Adobe After Effects[^/]*/Contents/MacOS/After Effects$' >/dev/null 2>&1 \
     || pgrep -f '/Adobe After Effects[^/]*.app/Contents/MacOS/After Effects$' >/dev/null 2>&1; then
    die "After Effects appears to be running.
Quit it before installing. An in-place bundle swap while AE holds the old
image leaves AE with a deleted plug-in and a confusing error."
  fi

  mkdir -p "$ae_dir"
  if [[ -e "$installed" ]]; then
    local backup="$installed.backup-$(date +%Y%m%d%H%M%S)"
    mv "$installed" "$backup"
    note "replaced existing plug-in; backup at $backup"
  fi

  ditto "$bundle" "$installed"
  # Copying invalidates the signature, so sign after the copy, never before.
  codesign --force --sign - --timestamp=none "$installed"
  codesign --verify --strict "$installed" || die "post-copy signature verification failed"

  printf 'Installed %s (%s) -> %s\n' "$AE_PLUGIN_NAME" "$configuration" "$installed"
  note 'Restart After Effects, then apply the effect to a test composition.'
}

do_uninstall() {
  if [[ -e "$installed" ]]; then
    local backup="$installed.backup-$(date +%Y%m%d%H%M%S)"
    mv "$installed" "$backup"
    printf 'Moved the installed plug-in to %s\n' "$backup"
    note 'Restart After Effects for the removal to take effect.'
  else
    printf 'Not installed: %s\n' "$AE_PLUGIN_NAME"
  fi
}

do_info() {
  local pipl
  pipl="$(pipl_source)"
  printf 'name:        %s\n' "$AE_PLUGIN_NAME"
  printf 'kind:        %s\n' "$AE_PLUGIN_KIND"
  printf 'match name:  %s\n' "$(grep -A3 'AE_Effect_Match_Name' "$pipl" | grep -oE '"[^"]+"' | tail -1 | tr -d '"' || echo '(not found)')"
  printf 'entry point: %s\n' "$(pipl_entry_points "$pipl" | tr '\n' ' ')"
  printf 'pipl source: %s\n' "${pipl#"$root"/}"
  printf 'sdk:         %s\n' "$(resolve_sdk)"
}

do_clean() {
  rm -rf "$root/build" "$root/$AE_PLUGIN_NAME.xcodeproj"
  printf 'Removed generated build artifacts (legacy Mac/ project untouched)\n'
}

case "$action" in
  sdk-path) resolve_sdk ;;
  info)     do_info ;;
  generate) do_generate ;;
  build)    do_build ;;
  verify)   do_verify ;;
  install)  do_install ;;
  uninstall) do_uninstall ;;
  reveal)   mkdir -p "$ae_dir"; open "$ae_dir" ;;
  clean)    do_clean ;;
  *)
    cat >&2 <<USAGE
Usage: scripts/ae-macos.sh <action> [Debug|Release]

  sdk-path    print the resolved Adobe After Effects SDK path
  info        print the plug-in identity declared in the PiPL
  generate    regenerate $AE_PLUGIN_NAME.xcodeproj from project.yml
  build       compile into build/<configuration>/
  verify      check the built bundle (layout, PiPL, symbols, flags, signature)
  install     verify, then install into the per-user AE plug-in folder
  uninstall   move the installed bundle to a timestamped backup
  reveal      open the AE plug-in folder in Finder
  clean       remove generated artifacts
USAGE
    exit 1
    ;;
esac