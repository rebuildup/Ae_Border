# macOS SDK 26.5 development

This migration follows [ae-plugin-template](https://github.com/rebuildup/ae-plugin-template).
The sources and legacy `Mac/` Xcode project remain in place; the generated
root-level `Border.xcodeproj` is NOT authoritative.

## Local setup (macOS, Xcode, After Effects 2026)

```sh
# Path must contain Examples/Headers/AE_Effect.h.
printf '%s\n' '/absolute/path/to/AfterEffectsSDK' > .env.local
mise install
mise run sdk-path
mise run build
mise run verify
# Only with After Effects closed and after inspecting the build:
mise run install
```

Build and verification do not modify the After Effects installation.
Installation uses the per-user MediaCore plug-in directory, keeps a backup
of any existing plug-in, and re-signs the copied bundle ad-hoc.
Restart After Effects, apply **Border** to a test composition, verify output
and parameter behavior, and record the host version. A successful build or
bundle verification is **not** evidence that the host works.

- `mise run generate`: rebuild the generated Xcode project from `project.yml`.
- `mise run build-release`: build both arm64 and x86_64.
- `mise run verify-release`: inspect the Release bundle.
- `mise run uninstall`: move the installed bundle to a timestamped backup.
- `mise run clean`: delete generated files, not the legacy Mac project.

Keep PiPL match name, registration identifiers and parameter IDs stable.
SDK headers remain external. Developer ID signing and notarization are out of scope.
